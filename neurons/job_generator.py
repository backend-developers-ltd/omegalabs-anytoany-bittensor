from compute_horde.base.volume import Volume, HuggingfaceVolume
import json
from compute_horde.base.output_upload import (
    MultiUpload,
    OutputUpload,
    SingleFilePutUpload,
)
from compute_horde.base.volume import SingleFileVolume, MultiVolume

import uuid
from compute_horde.executor_class import ExecutorClass, DEFAULT_EXECUTOR_CLASS
from compute_horde.miner_client.organic import OrganicJobDetails

from model.data import ModelId
from model.storage.disk.utils import get_local_model_snapshot_dir
from constants import (
    VIDEOBIND_HF_REPO_ID,
    VOLUME_DIR,
    MODELS_RELATIVE_PATH,
    CHECKPOINTS_RELATIVE_PATH,
    DATASET_FILENAME,
    OUTPUT_FILENAME,
)


class ValidationJobGenerator:
    def __init__(
        self,
        competition_id: str,
        hotkey: str,
        model_id: ModelId,
        block: int,
        data_sample_url: str,
        docker_image_name: str,
        executor_class: str | None = None,
    ) -> None:
        self.competition_id = competition_id
        self.hotkey = hotkey
        self.model_id = model_id
        self.block = block
        self.data_sample_url = data_sample_url
        self._docker_image_name = docker_image_name
        self._executor_class = (
            ExecutorClass(executor_class) if executor_class else DEFAULT_EXECUTOR_CLASS
        )

        self.s3_output_filename = f"{hotkey}_output.json"

    def total_timeout_seconds(self) -> int:
        return 60 * 40

    def wait_timeout_seconds(self) -> int:
        return 60 * 60

    def docker_image_name(self) -> str:
        return self._docker_image_name

    def executor_class(self) -> ExecutorClass:
        return self._executor_class

    def docker_run_cmd(self) -> list[str]:
        args = [
            "python",
            "trusted_miner_entrypoint.py",
            "--competition-id",
            self.competition_id,
            "--hotkey",
            self.hotkey,
            "--model-id",
            self.model_id.to_compressed_str(),
            "--block",
            str(self.block),
            "--models-dir",
            str(VOLUME_DIR / MODELS_RELATIVE_PATH),
        ]

        return args

    def docker_run_options_preset(self) -> str:
        return "nvidia_all"

    def raw_script(self) -> str | None:
        return None

    def volume(self) -> Volume | None:
        return MultiVolume(
            volumes=[
                HuggingfaceVolume(
                    repo_id=self.model_id.namespace + "/" + self.model_id.name,
                    revision=self.model_id.commit,
                    relative_path=get_local_model_snapshot_dir(
                        MODELS_RELATIVE_PATH, self.hotkey, self.model_id
                    ),
                ),
                HuggingfaceVolume(
                    repo_id=VIDEOBIND_HF_REPO_ID,
                    revision=None,
                    relative_path=CHECKPOINTS_RELATIVE_PATH,
                ),
                SingleFileVolume(
                    url=self.data_sample_url,
                    relative_path=DATASET_FILENAME,
                ),
            ]
        )

    def output_upload(self) -> OutputUpload | None:
        return MultiUpload(
            uploads=[
                SingleFilePutUpload(
                    url=generate_upload_url(self.s3_output_filename),
                    relative_path=OUTPUT_FILENAME,
                )
            ]
        )

    def get_job_details(self) -> OrganicJobDetails:
        job_uuid = uuid.uuid4()
        return OrganicJobDetails(
            job_uuid=str(job_uuid),
            executor_class=self.executor_class(),
            docker_image=self.docker_image_name(),
            raw_script=self.raw_script(),
            docker_run_options_preset=self.docker_run_options_preset(),
            docker_run_cmd=self.docker_run_cmd(),
            total_job_timeout=self.total_timeout_seconds(),
            volume=self.volume(),
            output=self.output_upload(),
        )

    async def download_output(self):
        response = await download_file_content(
            get_public_url(key=self.s3_output_filename)
        )
        return json.loads(response)
