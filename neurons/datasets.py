import os
import time

import huggingface_hub
import ulid
from datasets import Dataset


def get_timestamp_from_filename(filename: str) -> float:
    return ulid.from_str(os.path.splitext(filename.split("/")[-1])[0]).timestamp().timestamp


def get_recent_omega_dataset_files(hf_repo_id: str, data_files_prefix: str, min_age: float, max_files: int) -> list[str]:
    omega_ds_files = huggingface_hub.repo_info(repo_id=hf_repo_id, repo_type="dataset").siblings
    recent_files = [
        f.rfilename
        for f in omega_ds_files if
        f.rfilename.startswith(data_files_prefix) and
        time.time() - get_timestamp_from_filename(f.rfilename) < min_age
    ][:max_files]
    return recent_files


def shuffle_omega_dataset(dataset: Dataset, shuffle_seed: int | None = None) -> dict:
    return next(dataset.shuffle(seed=shuffle_seed).iter(batch_size=64))


def get_huggingface_file_url(hf_repo_id: str, revision: str, filename: str) -> str:
    return f"https://huggingface.co/datasets/{hf_repo_id}/resolve/{revision}/{filename}"
