from pathlib import Path
from dataclasses import dataclass
from typing import Type, Optional, Any, List, Tuple
import math


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

VIDEOBIND_HF_REPO_ID = 'jondurbin/videobind-v0.2'
VIDEOBIND_FILENAME = 'videobind.pth'

VOLUME_DIR = Path('/volume')
OUTPUT_DIR = Path('/output')

MODELS_RELATIVE_PATH = '.'
CHECKPOINTS_RELATIVE_PATH = 'checkpoints'

OUTPUT_FILENAME = 'output.json'
DATASET_FILENAME = 'dataset.json'
