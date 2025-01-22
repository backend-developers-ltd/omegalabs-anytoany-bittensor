import argparse
import asyncio
import logging
import os
import time

import bittensor

from constants import TRUSTED_MINER_ADDRESS, TRUSTED_MINER_PORT, TRUSTED_MINER_HOTKEY
from model.data import ModelId, ModelMetadata
from neurons.computation_providers import ComputeHordeComputationProvider, TrustedMiner

BITTENSOR_WALLET = os.getenv('BITTENSOR_WALLET', 'default')
BITTENSOR_HOTKEY = os.getenv('BITTENSOR_HOTKEY', 'default')

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
    wallet = bittensor.wallet(name=BITTENSOR_WALLET, hotkey=BITTENSOR_HOTKEY)

    start = time.time()

    computation_provider = ComputeHordeComputationProvider(wallet=wallet, trusted_miner=TrustedMiner(
        address=TRUSTED_MINER_ADDRESS,
        port=TRUSTED_MINER_PORT,
        hotkey=TRUSTED_MINER_HOTKEY,
    ))

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
