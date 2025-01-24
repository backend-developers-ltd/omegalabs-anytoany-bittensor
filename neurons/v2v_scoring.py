import os
from datasets import load_dataset, Audio, DownloadConfig
import huggingface_hub
from tempfile import TemporaryDirectory
import time
from typing import Optional
from datasets import Dataset
import ulid
import numpy as np
import random
import subprocess
import torch
import bittensor as bt

import sys
from pathlib import Path

from neurons.datasets import shuffle_omega_dataset, get_recent_omega_dataset_files, get_huggingface_file_url

sys.path.append(str(Path(__file__).parent.parent))

from models.S2S import inference as s2s_inference
from evaluation.S2S.distance import S2SMetrics

HF_DATASET = "omegalabsinc/omega-voice"
DATA_FILES_PREFIX = "default/train/"
MIN_AGE = 8 * 60 * 60  # 8 hours
MAX_FILES = 8


def get_recent_omega_voice_dataset_files() -> list[str]:
    return get_recent_omega_dataset_files(HF_DATASET, DATA_FILES_PREFIX, MIN_AGE, MAX_FILES)


def get_recent_omega_voice_dataset_urls() -> list[str]:
    recent_files = get_recent_omega_voice_dataset_files()
    return [
        get_huggingface_file_url(HF_DATASET, 'main', filename)
        for filename in recent_files
    ]


def process_diarization_dataset(dataset: Dataset) -> dict | None:
    dataset.cast_column("audio", Audio(sampling_rate=16000))
    omega_dataset = shuffle_omega_dataset(dataset)

    # Initialize dictionary to store processed samples
    overall_dataset = {k: [] for k in omega_dataset.keys()}

    # Process each audio sample
    for i in range(len(omega_dataset['audio'])):
        # Extract raw audio array
        audio_array = omega_dataset['audio'][i]

        # Get speaker timestamps and IDs
        diar_timestamps_start = np.array(omega_dataset['diar_timestamps_start'][i])
        diar_speakers = np.array(omega_dataset['diar_speakers'][i])

        # Skip samples with only 1 speaker
        if len(set(diar_speakers)) == 1:
            continue

        # Add all fields for this sample
        for k in omega_dataset.keys():
            value = audio_array if k == 'audio' else omega_dataset[k][i]
            overall_dataset[k].append(value)

        # Stop after collecting 16 valid samples
        if len(overall_dataset['audio']) >= 8:
            break

    # Check if we found enough valid samples
    if len(overall_dataset['audio']) < 1:
        return None

    return overall_dataset


def pull_latest_diarization_dataset() -> Optional[dict]:
    recent_files = get_recent_omega_voice_dataset_files()
    if len(recent_files) == 0:
        return None

    download_config = DownloadConfig(download_desc="Downloading Omega Voice Dataset")
    with TemporaryDirectory(dir='./data_cache') as temp_dir:
        # Load the dataset from HuggingFace using the recent files
        dataset = load_dataset(HF_DATASET, data_files=recent_files, cache_dir=temp_dir, download_config=download_config)["train"]
        return process_diarization_dataset(dataset)


def get_diarization_dataset_from_disk(dir_path) -> dict | None:
    dataset = load_dataset(dir_path)['train']
    return process_diarization_dataset(dataset)


def get_gpu_memory():
    # system-level
    output = subprocess.check_output(['nvidia-smi', '--query-gpu=memory.total,memory.used', '--format=csv,nounits,noheader'])
    total_gb, used_gb = map(lambda s: int(s) / 1e3, output.decode('utf-8').split(','))
    return total_gb, used_gb, total_gb - used_gb

def log_gpu_memory(msg=''):
    # process-level
    t = torch.cuda.get_device_properties(0).total_memory
    r = torch.cuda.memory_reserved(0)
    a = torch.cuda.memory_allocated(0)
    bt.logging.info(f"GPU-MEM {msg} Total: {t/1e9:.2f}GB, Reserved: {r/1e9:.2f}GB, Allocated: {a/1e9:.2f}GB")

def cleanup_gpu_memory():
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()


def get_model_files_from_hf(hf_repo_id: str, local_dir: str) -> str:
    repo_dir = Path(local_dir) / hf_repo_id
    huggingface_hub.snapshot_download(hf_repo_id, local_dir=repo_dir)
    return str(repo_dir)


def get_hotkey_file_contents(model_dir: str, target_file: str = "hotkey.txt") -> str | None:
    # Download if not exists and read the target file
    hf_api = huggingface_hub.HfApi()
    target_file_contents = None
    try:
        target_file_path = Path(model_dir) / target_file
        if not target_file_path.exists():
            target_file_path = hf_api.hf_hub_download(repo_id=hf_repo_id, filename=target_file, local_dir=model_dir)
        with open(target_file_path, 'r') as file:
            target_file_contents = file.read().strip()
    except huggingface_hub.utils._errors.EntryNotFoundError:
        print(f"Warning: File '{target_file}' not found in the repository.")
    except Exception as e:
        print(f"An error occurred while trying to read '{target_file}': {str(e)}")

    return target_file_contents


def compute_s2s_metrics(
    model_id: str,
    hf_repo_id: str,
    local_dir: str,
    mini_batch: Dataset,
    hotkey: str,
    block,
    model_tracker,
    device: str='cuda',
    mimi_weight_path: str | None = None,
    whisper_model_dir_path: str | None = None,
    rawnet3_model_path: str | None = None
) -> float:
    cleanup_gpu_memory()
    log_gpu_memory('before model load')

    inference_recipe = s2s_inference(model_id, hf_repo_id, local_dir, device)
    hotkey_file_contents = get_hotkey_file_contents(local_dir)

    # inference_recipe, hotkey_file_contents = load_ckpt_from_hf(model_id, hf_repo_id, local_dir=local_dir, device=device, target_file="hotkey.txt")

    # Check if the contents of license file are the same as the hotkey if in repo
    if hotkey_file_contents is not None and hotkey_file_contents != hotkey:
        bt.logging.warning(f"*** Hotkey file contents {hotkey_file_contents[:48]} do not match hotkey {hotkey}. Returning score of 0. ***")
        cleanup_gpu_memory()
        log_gpu_memory('after model clean-up')
        return 0
    elif hotkey_file_contents is not None and hotkey_file_contents == hotkey:
        bt.logging.info(f"Hotkey file contents match hotkey {hotkey}")
    
    # Check if the model is unique. Calculates the model's checkpoint (.pt) file hash for storage.
    if model_tracker is not None:
        model_paths = inference_recipe.model_paths
        for path in model_paths:
            is_model_unique, model_hash = model_tracker.is_model_unique(
                hotkey, 
                block, 
                path
            )
            if is_model_unique:
                bt.logging.info(f"Model {model_id} with hash {model_hash} on block {block} is unique.")
            else:
                bt.logging.warning(f"*** Model {model_id} with hash {model_hash} on block {block} is not unique. Returning score of 0. ***")
                cleanup_gpu_memory()
                log_gpu_memory('after model clean-up')
                return 0
        
    log_gpu_memory('after model load')
    cache_dir = "./model_cache"
    s2s_metrics = S2SMetrics(cache_dir=cache_dir, mimi_weight_path=mimi_weight_path, whisper_model_dir_path=whisper_model_dir_path, rawnet3_model_path=rawnet3_model_path)
    metrics = {'mimi_score': [],
                'wer_score': [],
                'length_penalty': [],
                'pesq_score': [],
                'anti_spoofing_score': [],
                'combined_score': [],
                'total_samples': 0}
    
   
    for i in range(len(mini_batch['youtube_id'])):
        youtube_id = mini_batch['youtube_id'][i]
        audio_array = np.array(mini_batch['audio'][i]['array'])
        sample_rate = mini_batch['audio'][i]['sampling_rate']

        diar_timestamps_start = np.array(mini_batch['diar_timestamps_start'][i])
        diar_timestamps_end = np.array(mini_batch['diar_timestamps_end'][i])
        diar_speakers = np.array(mini_batch['diar_speakers'][i])
        if len(diar_timestamps_start) == 1:
            continue
        

        test_idx = random.randint(0, len(diar_timestamps_start) - 2)
        diar_sample = audio_array[int(diar_timestamps_start[test_idx] * sample_rate):int(diar_timestamps_end[test_idx] * sample_rate)]
        diar_gt = audio_array[int(diar_timestamps_start[test_idx+1] * sample_rate):int(diar_timestamps_end[test_idx+1] * sample_rate)]
        speaker = diar_speakers[test_idx]

        # Add minimum length check (250ms = 0.25 seconds)
        min_samples = int(0.25 * sample_rate)
        if len(diar_sample) < min_samples or len(diar_gt) < min_samples:
            continue

        # Perform inference
        result = inference_recipe.inference(audio_array=diar_sample, sample_rate=sample_rate)
        pred_audio = result['audio']
        
        # Check if inference produced valid audio
        if pred_audio is None or len(pred_audio) == 0 or pred_audio.shape[-1] == 0:
            for k, v in metrics.items():
                if k == 'total_samples':
                    metrics[k] += 1
                else:
                    metrics[k].append([0])
            continue

            
        metrics['total_samples'] += 1

        metrics_dict = s2s_metrics.compute_distance(gt_audio_arrs=[[diar_gt, sample_rate]], generated_audio_arrs=[[pred_audio.squeeze(0).squeeze(0), sample_rate]])
        
        for key, value in metrics_dict.items():
            metrics[key].append(value)
    
    mean_score = np.mean(metrics['combined_score'])
    bt.logging.info(f"Scoring {model_id} {hf_repo_id} complete: {mean_score:0.5f}")
    cleanup_gpu_memory()
    log_gpu_memory('after model clean-up')

    return mean_score


if __name__ == "__main__":
    # this snippet below is necessary to avoid double logging and extraneous logging caused by torchtune
    # import logging
    # import torchtune
    #
    # logging.root.handlers = []
    # logging.root.manager.loggerDict = {}
    # logging.root.level = logging.root.level
    #
    # bt.logging.set_info()

    from utilities.temp_dir_cache import TempDirCache

    temp_dir_cache = TempDirCache(10)
    for epoch in range(2):
        for hf_repo_id in ["tezuesh/moshi1", "tezuesh/moshi7"]:
            start_time = time.time()
            diar_time = time.time()
            mini_batch = pull_latest_diarization_dataset()
            bt.logging.info(f"Time taken for diarization dataset: {time.time() - diar_time:.2f} seconds")
            # local_dir = temp_dir_cache.get_temp_dir(hf_repo_id)
            local_dir = Path('./model_cache') / hf_repo_id #temp_dir_cache.get_temp_dir(hf_repo_id)
            get_model_files_from_hf(hf_repo_id, local_dir)

            hotkey = '5FeqmebkCWfepQPgSkrEHRwtpUmHGASF4BNERZDs9pvKFtcD'
            block = 1
            model_tracker = None
            whisper_model_dir_path = '../models/openai/whisper-large-v2'
            rawnet3_model_path = '../models/jungjee/RawNet3/model.pt'
            mimi_weight_path = '../models/tezuesh/mimi/tokenizer-e351c8d8-checkpoint125.safetensors'
            vals = compute_s2s_metrics(model_id="moshi", hf_repo_id=hf_repo_id, mini_batch=mini_batch, local_dir=local_dir, hotkey=hotkey, block=block, model_tracker=model_tracker, whisper_model_dir_path=whisper_model_dir_path, rawnet3_model_path=rawnet3_model_path)
            end_time = time.time()
            bt.logging.info(f"I am here {hf_repo_id} Time taken: {end_time - start_time:.2f} seconds")
            bt.logging.info(f"Combined score: {vals}")
            print(f"Score={vals}, took {end_time - start_time}s.")
            exit(0)
