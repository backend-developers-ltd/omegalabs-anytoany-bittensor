import argparse
import asyncio
import logging
import time

from model.data import ModelId, ModelMetadata
from model.model_tracker import ModelTracker
from neurons.computation_providers import LocalComputationProvider
from neurons.model_scoring import pull_latest_omega_dataset
from utilities.temp_dir_cache import TempDirCache


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
        help="Hotkey of the miner being validated (not trusted miner).",
    )
    parser.add_argument(
        "--model-id",
        type=str,
        required=True,
        help="Model ID to validate, in the compressed string format.",
    )
    parser.add_argument(
        "--block",
        type=int,
        required=True,
        help="Block number.",
    )

    return parser.parse_args()


async def score_model(competition_id: str, hotkey: str, model_metadata: ModelMetadata) -> tuple[float, float]:
    start = time.time()
    temp_dir_cache = TempDirCache(10)
    model_tracker = ModelTracker()

    computation_provider = LocalComputationProvider(temp_dir_cache, model_tracker)

    logging.info("Scoring model %s", model_metadata.id.hf_repo_id())

    score = await computation_provider.score_model(
        competition_id=competition_id,
        hotkey=hotkey,
        model_metadata=model_metadata,
    )

    logging.info(f"Returned score: {score}")

    return score, time.time() - start


if __name__ == "__main__":
    args = parse_arguments()
    model_id = ModelId.from_compressed_str(args.model_id)
    model_metadata = ModelMetadata(id=model_id, block=args.block)
    # These are from get_models.py.
    score, took_s = asyncio.run(score_model(args.competition_id, args.hotkey, model_metadata))
    print(f"Scored model {model_id.hf_repo_id()}, score={score}, took {took_s}s")
