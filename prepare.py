"""Data preparation using nampdn-ai/tiny-codes dataset.
Do NOT modify this file -- the agent only edits train.py.
"""

import math
import os
import json

import torch
import torch.nn.functional as F

DATA_DIR = "data"
EVAL_BATCHES = 10
TIME_BUDGET_SECONDS = 5 * 60

MAX_EXAMPLES = 100_000


def download_data():
    """Download and cache the tiny-codes dataset as plain text."""
    cache_file = os.path.join(DATA_DIR, "tiny_codes_text.txt")
    if os.path.exists(cache_file):
        with open(cache_file, "r") as f:
            return f.read()

    from datasets import load_dataset

    os.makedirs(DATA_DIR, exist_ok=True)
    print("Downloading nampdn-ai/tiny-codes from HuggingFace...")
    ds = load_dataset("nampdn-ai/tiny-codes", split="train")

    if MAX_EXAMPLES is not None and len(ds) > MAX_EXAMPLES:
        ds = ds.select(range(MAX_EXAMPLES))

    texts = []
    for row in ds:
        prompt = row.get("prompt", "") or ""
        response = row.get("response", "") or ""
        texts.append(f"{prompt}\n{response}\n\n")

    text = "".join(texts)

    with open(cache_file, "w") as f:
        f.write(text)

    print(f"Cached {len(texts)} examples ({len(text):,} chars) to {cache_file}")
    return text


def build_tokenizer(text: str):
    chars = sorted(set(text))
    stoi = {c: i for i, c in enumerate(chars)}
    itos = {i: c for c, i in stoi.items()}
    return stoi, itos, len(chars)


def encode(text: str, stoi: dict) -> torch.Tensor:
    return torch.tensor([stoi[c] for c in text], dtype=torch.long)


def decode(ids, itos: dict) -> str:
    return "".join(itos.get(t, "?") for t in ids)


def prepare():
    text = download_data()
    stoi, itos, vocab_size = build_tokenizer(text)
    data = encode(text, stoi)
    split = int(0.9 * len(data))
    train_data, val_data = data[:split], data[split:]
    print(f"Dataset: {len(text):,} chars, {vocab_size} unique tokens")
    print(f"Train: {len(train_data):,} tokens, Val: {len(val_data):,} tokens")
    return train_data, val_data, vocab_size, stoi, itos


def get_batch(data: torch.Tensor, batch_size: int, seq_len: int, device: torch.device):
    ix = torch.randint(0, len(data) - seq_len - 1, (batch_size,))
    x = torch.stack([data[i : i + seq_len] for i in ix]).to(device)
    y = torch.stack([data[i + 1 : i + 1 + seq_len] for i in ix]).to(device)
    return x, y


@torch.no_grad()
def evaluate(model, val_data: torch.Tensor, batch_size: int, seq_len: int, device: torch.device):
    model.eval()
    losses = []
    for _ in range(EVAL_BATCHES):
        x, y = get_batch(val_data, batch_size, seq_len, device)
        logits = model(x)
        if isinstance(logits, tuple):
            logits = logits[0]
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
        losses.append(loss.item())
    model.train()
    val_loss = sum(losses) / len(losses)
    val_bpb = val_loss / math.log(2)
    return val_loss, val_bpb


def generate(
    model,
    stoi: dict,
    itos: dict,
    prompt: str,
    max_new: int = 200,
    device: torch.device = torch.device("cpu"),
    max_seq_len: int = 128,
):
    model.eval()
    ids = torch.tensor(
        [stoi.get(c, 0) for c in prompt], dtype=torch.long, device=device
    ).unsqueeze(0)
    for _ in range(max_new):
        ctx = ids[:, -max_seq_len:]
        logits = model(ctx)
        if isinstance(logits, tuple):
            logits = logits[0]
        probs = F.softmax(logits[:, -1, :], dim=-1)
        nxt = torch.multinomial(probs, 1)
        ids = torch.cat([ids, nxt], dim=1)
    model.eval()
    return decode(ids[0].tolist(), itos)


def save_checkpoint(
    model, config, stoi: dict, itos: dict, out_dir: str = "checkpoint"
):
    from safetensors.torch import save_model

    os.makedirs(out_dir, exist_ok=True)
    save_model(model, os.path.join(out_dir, "model.safetensors"))
    with open(os.path.join(out_dir, "config.json"), "w") as f:
        json.dump(vars(config), f, indent=2)
    with open(os.path.join(out_dir, "tokenizer.json"), "w") as f:
        json.dump(
            {"stoi": stoi, "itos": {str(k): v for k, v in itos.items()}}, f
        )
    print(f"Saved checkpoint to {out_dir}/")


def load_checkpoint(
    model_cls,
    config_cls,
    out_dir: str = "checkpoint",
    device: torch.device = torch.device("cpu"),
):
    from safetensors.torch import load_model

    with open(f"{out_dir}/config.json") as f:
        config = config_cls(**json.load(f))
    with open(f"{out_dir}/tokenizer.json") as f:
        tok = json.load(f)
    stoi = tok["stoi"]
    itos = {int(k): v for k, v in tok["itos"].items()}

    model = model_cls(config).to(device)
    load_model(model, f"{out_dir}/model.safetensors", device=str(device))
    return model, config, stoi, itos


if __name__ == "__main__":
    train_data, val_data, vocab_size, stoi, itos = prepare()
    print("Data preparation complete.")
