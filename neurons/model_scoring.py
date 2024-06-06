import os
from tempfile import TemporaryDirectory
import time
from typing import List, Optional

import torch
import ulid
from omegaconf import OmegaConf, DictConfig
import huggingface_hub
from datasets import load_dataset, Dataset
from imagebind.models.multimodal_preprocessors import SimpleTokenizer
from imagebind.models.imagebind_model import ModalityType

from torchtune.data import (
    CROSS_ENTROPY_IGNORE_IDX,
    Message,
)
import numpy as np

from tune_recipes.gen import InferenceRecipe as Recipe


HF_DATASET = "omegalabsinc/omega-multimodal"
DATA_FILES_PREFIX = "default/train/"
DATA_FILES_SUFFIX = ".parquet"
MIN_AGE = 6 * 60 * 60  # 6 hours
MIN_AGE = 30 * 60 * 60  # 30 hours
MAX_FILES = 1#8
MODEL_FILE_PREFIX = "meta_model"
CONFIG_FILE = "training_config.yml"
BPE_PATH = "./models/bpe_simple_vocab_16e6.txt.gz"


@torch.no_grad()
def inference_loss(recipe, prompt, expected_output, media_embedding):
    loss_fn = torch.nn.CrossEntropyLoss()

    messages = [
        Message(
            role="user",
            content=prompt,
            masked=True,
        ),
        Message(
            role="assistant",
            content=expected_output,
        )
    ]

    input_ids, mask = recipe._tokenizer.tokenize_messages(messages)
    mm_context = [recipe.extract_mm_context(media_embedding, input_ids)] # context should be a list, batch-id indexed

    # Wherever mask == True, set to CROSS_ENTROPY_IGNORE_IDX. Otherwise keep as input_ids
    labels = list(np.where(mask, CROSS_ENTROPY_IGNORE_IDX, input_ids))
    assert len(input_ids) == len(labels)

    input_ids = torch.tensor(input_ids, device=recipe._device).unsqueeze(0)
    labels = torch.tensor(labels, device=recipe._device).unsqueeze(0)

    recipe._model.tok_embeddings.set_context(mm_context)

    logits = recipe._model(input_ids)
    # Shift so that tokens < n predict n
    logits = logits[..., :-1, :].contiguous()
    labels = labels[..., 1:].contiguous()
    logits = logits.transpose(1, 2)

    return loss_fn(logits, labels)


def get_video_understanding_score(inference_recipe, config, mini_batch):
    scores = []
    for video_embed, caption in zip(mini_batch["video_embed"], mini_batch["description"]):
        prompt = inference_recipe.mm_process_prompt(inference_recipe.prompt_template)
        loss = inference_loss(inference_recipe, prompt, caption, video_embed)
        score = (10 - loss).clamp(0, 1)
        scores.append(score.item())
    return torch.tensor(scores).mean().item()


metric_weights = {
    get_video_understanding_score: 1.0
}


def get_timestamp_from_filename(filename: str):
    return ulid.from_str(filename[len(DATA_FILES_PREFIX):filename.find(DATA_FILES_SUFFIX)]).timestamp().timestamp


def pull_latest_omega_dataset() -> Optional[Dataset]:
    omega_ds_files = huggingface_hub.repo_info(repo_id=HF_DATASET, repo_type="dataset").siblings
    now_s = time.time()
    age_filenames = sorted([
        (now_s - get_timestamp_from_filename(f.rfilename), f.rfilename)
        for f in omega_ds_files if
        f.rfilename.startswith(DATA_FILES_PREFIX)
    ])
    recent_files = [
        filename
        for age_s, filename in age_filenames
        if age_s < MIN_AGE
    ][:MAX_FILES]
    if len(recent_files) == 0:
        return None
    omega_dataset = load_dataset(HF_DATASET, data_files=recent_files)["train"]
    omega_dataset = next(omega_dataset.shuffle().iter(batch_size=64))
    return omega_dataset


def get_config(config_path, ckpt_path):
    cfg = OmegaConf.load(config_path)
    cfg.model = DictConfig({
        "_component_": "models.mmllama3_8b",
        "use_clip": False,
        "perception_tokens": cfg.model.perception_tokens,
    })
    cfg.checkpointer.checkpoint_dir = os.path.dirname(ckpt_path)
    cfg.checkpointer.checkpoint_files = [os.path.basename(ckpt_path)]
    cfg.inference.max_new_tokens = 300
    return cfg


def load_ckpt_from_hf(hf_repo_id: str) -> Recipe:
    # assert False, "make sure not to cache downloaded checkpoints"
    hf_api = huggingface_hub.HfApi()
    ckpt_files = [f for f in hf_api.list_repo_files(repo_id=hf_repo_id) if f.startswith(MODEL_FILE_PREFIX)]
    if len(ckpt_files) == 0:
        raise ValueError(f"No checkpoint files found in {hf_repo_id}")
    with TemporaryDirectory() as temp_dir:
        config_path = hf_api.hf_hub_download(repo_id=hf_repo_id, filename=CONFIG_FILE, local_dir=temp_dir)
        ckpt_path = hf_api.hf_hub_download(repo_id=hf_repo_id, filename=ckpt_files[0], local_dir=temp_dir)
        cfg = get_config(config_path, ckpt_path)
        recipe = Recipe(cfg)
        recipe.setup(cfg=cfg)
    return recipe, cfg


def load_and_transform_text(text, device):
    if text is None:
        return None
    tokenizer = SimpleTokenizer(bpe_path=BPE_PATH)
    tokens = [tokenizer(t).unsqueeze(0).to(device) for t in text]
    tokens = torch.cat(tokens, dim=0)
    return tokens


def embed_text(imagebind, texts: List[str], device) -> List[torch.FloatTensor]:
    return imagebind({ModalityType.TEXT: load_and_transform_text(texts, device)})[ModalityType.TEXT]


def get_model_score(hf_repo_id, mini_batch):
    inference_recipe, config = load_ckpt_from_hf(hf_repo_id)
    video_understanding_score = get_video_understanding_score(inference_recipe, config, mini_batch)
    return video_understanding_score * metric_weights[get_video_understanding_score]



def scores_to_weights(uids, scores_per_uid, uid_to_block, constants, competition_parameters):
    from validator import compute_wins
    # Compute wins and win rates per uid.
    wins, win_rate = compute_wins(uids, scores_per_uid, uid_to_block)
    print(win_rate)
    # Compute softmaxed weights based on win rate.
    model_weights = torch.tensor(
        [win_rate[uid] for uid in uids], dtype=torch.float32
    )
    step_weights = torch.softmax(model_weights / constants.temperature, dim=0)
    # Update weights based on moving average.
    new_weights = torch.zeros((len(uids),))#self.metagraph.S)
    for i, uid_i in enumerate(uids):
        new_weights[uid_i] = step_weights[i]
    print(new_weights)
    scale = (
        len(constants.COMPETITION_SCHEDULE)
        * competition_parameters.reward_percentage
    )
    new_weights *= scale / new_weights.sum()
    return new_weights


if __name__ == "__main__":
    from types import SimpleNamespace
    from pathlib import Path

    hf_repo_id = "a-good-gpt2/experiment_1"
    mini_batch = pull_latest_omega_dataset()
    # print(get_model_score(hf_repo_id, mini_batch))

    constants = SimpleNamespace(**{
        'temperature': 2.0,
        'COMPETITION_SCHEDULE': [SimpleNamespace(**{'reward_percentage': 100})],
    })
    competition_parameters = constants.COMPETITION_SCHEDULE[0]

    model_md = {
        0: {
            'cfg-path': 'output_checkpoints/experiment_1/training_config.yml',
            'ckpt-path': 'output_checkpoints/experiment_1/meta_model_latest.pt',
            'block': 1,
        },
        1: {
            'cfg-path': 'output_checkpoints/experiment_2/training_config.yml',
            'ckpt-path': 'output_checkpoints/experiment_2/meta_model_0.pt',
            'block': 1,
        }
    }

    snbp = Path('scores_n_blocks.pt')
    if snbp.exists() and False:
        scores_per_uid, uid_to_block = torch.load(snbp)
    else:
        scores_per_uid = {}
        uid_to_block = {}
        for uid, md in model_md.items():
            config = get_config(md['cfg-path'], md['ckpt-path'])
            recipe = Recipe(config)
            recipe.setup(cfg=config)
            score = get_video_understanding_score(recipe, config, mini_batch)
            scores_per_uid[uid] = score
            uid_to_block[uid] = md['block']
        torch.save((scores_per_uid, uid_to_block), snbp)

    print(scores_per_uid)
    weights = scores_to_weights(model_md.keys(), scores_per_uid, uid_to_block, constants, competition_parameters)
    print(weights)


