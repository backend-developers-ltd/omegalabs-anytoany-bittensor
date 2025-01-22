import json
from abc import abstractmethod, ABC
from dataclasses import dataclass

import bittensor as bt
from compute_horde.base.volume import HuggingfaceVolume
from compute_horde.miner_client.organic import OrganicMinerClient, run_organic_job
from datasets import Dataset

from constants import EXECUTOR_CLASS, VIDEOBIND_HF_REPO_ID, CHECKPOINTS_RELATIVE_PATH
from model.data import ModelMetadata
from model.model_tracker import ModelTracker
from neurons.job_generator import ValidationJobGenerator
from neurons.model_scoring import pull_latest_omega_dataset, get_model_score, get_model_files_from_hf
from neurons.s3 import upload_data_to_s3
from neurons.v2v_scoring import pull_latest_diarization_dataset, compute_s2s_metrics
from utilities.temp_dir_cache import TempDirCache


class DatasetNotAvailable(Exception):
    """
    No data is currently available to evaluate miner models on.
    """


@dataclass
class TrustedMiner:
    address: str
    port: int
    hotkey: str


class AbstractComputationProvider(ABC):
    @abstractmethod
    async def score_model(self, competition_id: str, hotkey: str, model_metadata: ModelMetadata) -> float:
        pass


class LocalComputationProvider(AbstractComputationProvider):
    """
    Runs computations on the same machine that the validator is running on.
    """

    def __init__(self, temp_dir_cache: TempDirCache, model_tracker: ModelTracker | None):
        self.temp_dir_cache = temp_dir_cache
        self.model_tracker = model_tracker

    async def score_model(self, competition_id: str, hotkey: str, model_metadata: ModelMetadata) -> float:
        # TODO: Move this somewhere else (model scoring)? so that it can be reused by trusted miner entrypoint.
        hf_repo_id = model_metadata.id.namespace + "/" + model_metadata.id.name
        local_dir = self.temp_dir_cache.get_temp_dir(hf_repo_id)

        if competition_id == "o1":
            eval_data = pull_latest_omega_dataset()
            if eval_data is None:
                raise DatasetNotAvailable()

            model_files = get_model_files_from_hf(hf_repo_id, local_dir)

            score = get_model_score(
                hotkey=hotkey,
                model_metadata=model_metadata,
                model_files=model_files,
                mini_batch=eval_data,
                model_tracker=self.model_tracker,
            )
        elif competition_id == "v1":
            eval_data_v2v = pull_latest_diarization_dataset()
            if eval_data_v2v is None:
                raise DatasetNotAvailable()

            score = compute_s2s_metrics(
                model_id="moshi",  # update this to the model id as we support more models.
                hf_repo_id=hf_repo_id,
                mini_batch=Dataset.from_dict(eval_data_v2v),
                local_dir=self.temp_dir_cache.get_temp_dir(hf_repo_id),
                hotkey=hotkey,
                block=model_metadata.block,
                model_tracker=self.model_tracker
            )
        else:
            raise ValueError(f"Invalid competition ID: {competition_id}")

        return score


class ComputeHordeComputationProvider(AbstractComputationProvider):
    """
    Runs computations on the Compute Horde subnet.
    """

    def __init__(self, wallet: bt.Wallet, trusted_miner: TrustedMiner):
        self.wallet = wallet
        self.trusted_miner = trusted_miner

    async def score_model(self, competition_id: str, hotkey: str, model_metadata: ModelMetadata) -> float:
        assert competition_id == 'o1'

        if competition_id == 'o1':
            data_sample = pull_latest_omega_dataset(shuffle_seed=0)
            if data_sample is None:
                raise RuntimeError("Could not load data sample.")

            bt.logging.info("Uploading data sample to S3.")

            data_sample_url = upload_data_to_s3(json.dumps(data_sample))

            job_generator = ValidationJobGenerator(
                competition_id=competition_id,
                hotkey=hotkey,
                model_metadata=model_metadata,
                data_sample_url=data_sample_url,
                docker_image_name='backenddevelopersltd/slawek-test:v0-latest',
                executor_class=EXECUTOR_CLASS,
                hf_volumes=[
                    HuggingfaceVolume(
                        repo_id=VIDEOBIND_HF_REPO_ID,
                        revision=None,
                        relative_path=CHECKPOINTS_RELATIVE_PATH,
                    ),
                ],
            )

            score = await self.run_validation_job(job_generator)
        elif competition_id == 'v1':
            data_sample = pull_latest_diarization_dataset()
            if data_sample is None:
                raise RuntimeError("Could not load data sample.")

            bt.logging.info("Uploading data sample to S3.")

            data_sample_url = upload_data_to_s3(json.dumps(data_sample))

            job_generator = ValidationJobGenerator(
                competition_id=competition_id,
                hotkey=hotkey,
                model_metadata=model_metadata,
                data_sample_url=data_sample_url,
                docker_image_name='backenddevelopersltd/slawek-test:v0-latest',
                executor_class=EXECUTOR_CLASS,
                hf_volumes=[
                    HuggingfaceVolume(
                        repo_id='openai/whisper-large-v2',
                        revision=None,
                        relative_path='openai/whisper-large-v2',
                    ),
                    HuggingfaceVolume(
                        repo_id='jungjee/RawNet3',
                        revision=None,
                        relative_path='jungjee/RawNet3',
                    ),
                    HuggingfaceVolume(
                        repo_id='tezuesh/mimi',
                        revision=None,
                        relative_path='tezuesh/mimi',
                    ),
                ],
            )

            score = await self.run_validation_job(job_generator)
        else:
            raise ValueError(f"Invalid competition ID: {competition_id}")

        return score


    async def run_validation_job(self, job_generator: ValidationJobGenerator) -> float:
        job_details = job_generator.get_job_details()

        miner_client = OrganicMinerClient(
            miner_hotkey=self.trusted_miner.hotkey,
            miner_address=self.trusted_miner.address,
            miner_port=self.trusted_miner.port,
            job_uuid=str(job_details.job_uuid),
            my_keypair=self.wallet.get_hotkey(),
        )

        bt.logging.info(f"Starting organic job: {job_details}")
        stdout, stderr = await run_organic_job(
            miner_client,
            job_details,
            wait_timeout=job_generator.wait_timeout_seconds(),
        )
        bt.logging.info(f"Job completed with stdout: {stdout}, stderr: {stderr}")

        output = await job_generator.download_output()

        return output["score"]
