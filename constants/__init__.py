import os
from pathlib import Path
from dataclasses import dataclass
from typing import List
import math

from dotenv import load_dotenv

load_dotenv()


@dataclass
class CompetitionParameters:
    """Class defining model parameters"""

    # Reward percentage
    reward_percentage: float
    # Competition id
    competition_id: str


# ---------------------------------
# Project Constants.
# ---------------------------------

# The uid for this subnet.
SUBNET_UID = 21
# The start block of this subnet
SUBNET_START_BLOCK = 2635801
# The root directory of this project.
ROOT_DIR = Path(__file__).parent.parent
# The maximum bytes for the hugging face repo
MAX_HUGGING_FACE_BYTES: int = 18 * 1024 * 1024 * 1024
# Schedule of model architectures
COMPETITION_SCHEDULE: List[CompetitionParameters] = [
    CompetitionParameters(
        reward_percentage=0.5,
        competition_id="o1",
    ),
    CompetitionParameters(
        reward_percentage=0.5,
        competition_id="v1",
    ),
]
ORIGINAL_COMPETITION_ID = "o1"
BLOCK_DURATION = 12  # 12 seconds


assert math.isclose(sum(x.reward_percentage for x in COMPETITION_SCHEDULE), 1.0)

# ---------------------------------
# Miner/Validator Model parameters.
# ---------------------------------

weights_version_key = 1

# validator weight moving average term
alpha = 0.9
# validator scoring exponential temperature
temperature = 0.08
# validator score boosting for earlier models.
timestamp_epsilon = 0.01

# ---------------------------------
# Constants for running with a trusted miner.
# ---------------------------------

USE_COMPUTE_HORDE_TRUSTED_MINER = os.getenv("USE_COMPUTE_HORDE_TRUSTED_MINER") in ("1", "true", "yes")

VIDEOBIND_HF_REPO_ID = 'jondurbin/videobind-v0.2'
VIDEOBIND_FILENAME = 'videobind.pth'

VOLUME_DIR = Path('/volume')
OUTPUT_DIR = Path('/output')

MODELS_RELATIVE_PATH = '.'
CHECKPOINTS_RELATIVE_PATH = 'checkpoints'

OUTPUT_FILENAME = 'output.json'
DATASET_FILENAME = 'dataset.json'

TRUSTED_MINER_ADDRESS = os.getenv("TRUSTED_MINER_ADDRESS")
miner_port_str = os.getenv("TRUSTED_MINER_PORT")
TRUSTED_MINER_PORT = int(miner_port_str) if miner_port_str else None
TRUSTED_MINER_HOTKEY = os.getenv("TRUSTED_MINER_HOTKEY")

EXECUTOR_CLASS = os.getenv("EXECUTOR_CLASS")

COMPUTE_HORDE_JOB_STDOUT_MARKER = 'COMPUTE_HORDE_JOB_STDOUT'

# TODO: Remove S3 stuff

AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
AWS_ENDPOINT_URL = os.getenv("AWS_ENDPOINT_URL", "")
COMPUTE_HORDE_VALIDATION_S3_BUCKET = os.getenv("COMPUTE_HORDE_VALIDATION_S3_BUCKET", default="https://s3.amazonaws.com")
