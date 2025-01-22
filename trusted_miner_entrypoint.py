import argparse
import json
import logging
import time

from datasets import Dataset

from constants import (
    VIDEOBIND_FILENAME,
    ORIGINAL_COMPETITION_ID,
    VOLUME_DIR,
    CHECKPOINTS_RELATIVE_PATH,
    DATASET_FILENAME,
    OUTPUT_DIR,
    OUTPUT_FILENAME,
)
from model.data import ModelId, ModelMetadata
from model.storage.disk.utils import get_local_model_snapshot_dir
from neurons.model_scoring import get_model_score, get_model_files_from_disk
from neurons.v2v_scoring import compute_s2s_metrics


def parse_arguments():
    parser = argparse.ArgumentParser(description="Argument parser for the script")
    parser.add_argument(
        "--competition-id",
        type=str,
        required=True,
        help="Competition ID.",
    )
    parser.add_argument(
        "--hotkey",
        type=str,
        required=True,
        help="Hotkey to be used (of the miner to validate).",
    )
    parser.add_argument(
        "--model-id",
        type=str,
        required=True,
        help="Model ID to validate, in the compressed string format.",
    )
    parser.add_argument(
        "--models-dir", type=str, required=True, help="Directory containing the models."
    )
    parser.add_argument(
        "--block",
        type=int,
        required=True,
        help="Block number.",
    )

    return parser.parse_args()


def score_model(
    competition_id: str, hotkey: str, model_metadata: ModelMetadata, models_dir: str
) -> tuple[float, float]:
    st = time.time()
    logging.info("Loading data sample")

    with open(VOLUME_DIR / DATASET_FILENAME) as f:
        dataset_raw = f.read()

    data_sample = json.loads(dataset_raw)

    model_dir = get_local_model_snapshot_dir(models_dir, hotkey, model_metadata.id)

    if competition_id == 'o1':
        model_files = get_model_files_from_disk(model_dir)

        videobind_path = VOLUME_DIR / CHECKPOINTS_RELATIVE_PATH / VIDEOBIND_FILENAME

        logging.info("Scoring model")

        score = get_model_score(
            hotkey=hotkey,
            model_metadata=model_metadata,
            model_files=model_files,
            mini_batch=data_sample,
            model_tracker=None,
            videobind_path=videobind_path,
        )
    elif competition_id == 'v1':
        logging.info("Scoring model")

        score = compute_s2s_metrics(
            model_id='moshi',  # update this to the model id as we support more models.
            hf_repo_id=model_metadata.id.hf_repo_id(),
            local_dir=get_local_model_snapshot_dir(
                models_dir, hotkey, model_metadata.id
            ),
            mini_batch=Dataset.from_dict(data_sample),
            hotkey=hotkey,
            block=model_metadata.block,
            model_tracker=None,
            mimi_weight_path=VOLUME_DIR / 'tezuesh/mimi/tokenizer-e351c8d8-checkpoint125.safetensors',
            whisper_model_dir_path=VOLUME_DIR / 'openai/whisper-large-v2',
            rawnet3_model_path=VOLUME_DIR / 'jungjee/RawNet3/model.pt',
        )
    else:
        raise ValueError(f"Invalid competition ID: {competition_id}")

    logging.info(f"Returned score: {score}")

    total_s = time.time() - st

    return score, total_s


if __name__ == "__main__":
    args = parse_arguments()

    model_id = ModelId.from_compressed_str(args.model_id)
    model_metadata = ModelMetadata(id=model_id, block=args.block)

    score, total_s = score_model(
        args.competition_id, args.hotkey, model_metadata, args.models_dir
    )
    result = {"score": score, "total_s": total_s}

    with open(OUTPUT_DIR / OUTPUT_FILENAME, "w+") as f:
        json.dump(result, f)
