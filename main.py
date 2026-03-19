from dataclasses import dataclass
import json
import os
import time

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class ModelConfig:
    d_model: int = 512
    n_layers: int = 6
    n_dense_layers: int = 1
    n_heads: int = 8
    d_head: int = 64
    d_kv_latent: int = 128
    d_q_latent: int = 256
    d_head_rope: int = 32
    d_ff_dense: int = 2048
    d_ff_expert: int = 512
    n_routed_experts: int = 8
    n_shared_experts: int = 1
    top_k: int = 2
    vocab_size: int = 8192
    max_seq_len: int = 5120
    gamma: float = 1e-3
    balance_alpha: float = 1e-4
    lr: float = 3e-4
    weight_decay: float = 0.1
    batch_size: int = 8
    train_steps: int = 500
    log_interval: int = 25


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)

# http://arxiv.org/pdf/2104.09864
def precompute_rope_freqs(dim: int, max_seq_len: int, theta: float = 10000.0):
    if dim % 2 != 0:
        raise ValueError("RoPE dim must be even")
    inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
    positions = torch.arange(max_seq_len).float()
    angles = torch.outer(positions, inv_freq)
    emb = torch.cat((angles, angles), dim=-1)
    return emb.cos(), emb.sin()


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    seq_len = x.shape[1]
    cos = cos[:seq_len].unsqueeze(0).unsqueeze(2).to(dtype=x.dtype, device=x.device)
    sin = sin[:seq_len].unsqueeze(0).unsqueeze(2).to(dtype=x.dtype, device=x.device)
    return (x * cos) + (rotate_half(x) * sin)


class SwiGLU(nn.Module):
    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        self.w_gate = nn.Linear(d_model, d_ff, bias=False)
        self.w_up = nn.Linear(d_model, d_ff, bias=False)
        self.w_down = nn.Linear(d_ff, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w_down(F.silu(self.w_gate(x)) * self.w_up(x))


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-8):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm = torch.sqrt(torch.mean(x * x, dim=-1, keepdim=True) + self.eps)
        return x / norm * self.weight


class MLA(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_head: int,
        d_kv_latent: int,
        d_q_latent: int,
        d_head_rope: int,
        max_seq_len: int,
    ):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_head
        self.d_head_rope = d_head_rope

        self.w_dq = nn.Linear(d_model, d_q_latent, bias=False)
        self.w_uq = nn.Linear(d_q_latent, n_heads * d_head, bias=False)
        self.w_qr = nn.Linear(d_q_latent, n_heads * d_head_rope, bias=False)

        self.w_dkv = nn.Linear(d_model, d_kv_latent, bias=False)
        self.w_uk = nn.Linear(d_kv_latent, n_heads * d_head, bias=False)
        self.w_uv = nn.Linear(d_kv_latent, n_heads * d_head, bias=False)
        self.w_kr = nn.Linear(d_model, d_head_rope, bias=False)

        self.w_o = nn.Linear(n_heads * d_head, d_model, bias=False)

        cos, sin = precompute_rope_freqs(dim=d_head_rope, max_seq_len=max_seq_len)
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz, seq_len, _ = x.shape

        c_q = self.w_dq(x)
        q_c = self.w_uq(c_q).view(bsz, seq_len, self.n_heads, self.d_head)
        q_r = self.w_qr(c_q).view(bsz, seq_len, self.n_heads, self.d_head_rope)
        q_r = apply_rope(q_r, self.rope_cos, self.rope_sin)

        c_kv = self.w_dkv(x)
        k_c = self.w_uk(c_kv).view(bsz, seq_len, self.n_heads, self.d_head)
        v_c = self.w_uv(c_kv).view(bsz, seq_len, self.n_heads, self.d_head)

        k_r = self.w_kr(x).view(bsz, seq_len, 1, self.d_head_rope)
        k_r = apply_rope(k_r, self.rope_cos, self.rope_sin).expand(-1, -1, self.n_heads, -1)

        q = torch.cat([q_c, q_r], dim=-1).transpose(1, 2)
        k = torch.cat([k_c, k_r], dim=-1).transpose(1, 2)
        v = v_c.transpose(1, 2)

        scale = (self.d_head + self.d_head_rope) ** -0.5
        attn_scores = torch.matmul(q, k.transpose(-1, -2)) * scale

        causal_mask = torch.triu(
            torch.ones(seq_len, seq_len, device=x.device, dtype=torch.bool),
            diagonal=1,
        )
        attn_scores = attn_scores.masked_fill(causal_mask, float("-inf"))

        attn_probs = F.softmax(attn_scores, dim=-1)
        attn_out = torch.matmul(attn_probs, v)
        attn_out = attn_out.transpose(1, 2).contiguous().view(bsz, seq_len, self.n_heads * self.d_head)
        return self.w_o(attn_out)

class DeepSeekMoE(nn.Module):
    def __init__(
        self,
        d_model: int,
        d_ff_expert: int,
        n_routed: int,
        top_k: int,
        n_shared: int = 1,
        gamma: float = 1e-3,
        balance_alpha: float = 1e-4,
    ):
        super().__init__()
        self.n_routed = n_routed
        self.top_k = top_k
        self.gamma = gamma
        self.balance_alpha = balance_alpha

        self.router = nn.Linear(d_model, n_routed, bias=False)
        self.shared_experts = nn.ModuleList([SwiGLU(d_model, d_ff_expert) for _ in range(n_shared)])
        self.routed_experts = nn.ModuleList([SwiGLU(d_model, d_ff_expert) for _ in range(n_routed)])

        self.register_buffer("expert_bias", torch.zeros(n_routed))
        self.last_scores: torch.Tensor | None = None
        self.last_topk_idx: torch.Tensor | None = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape
        N = B * T
        x_flat = x.reshape(N, D)

        scores = torch.sigmoid(self.router(x_flat))                      # (N, n_routed)
        biased = scores + self.expert_bias.unsqueeze(0)                  # (N, n_routed)
        _, topk_idx = torch.topk(biased, self.top_k, dim=-1)            # (N, K)

        selected = torch.gather(scores, dim=1, index=topk_idx)          # (N, K)
        gates = selected / (selected.sum(dim=-1, keepdim=True) + 1e-9)  # (N, K)

        shared_out = sum(expert(x_flat) for expert in self.shared_experts)

        routed_out = torch.zeros_like(x_flat)
        for e in range(self.n_routed):
            token_pos, slot_pos = torch.where(topk_idx == e)
            if token_pos.numel() == 0:
                continue
            y_e = self.routed_experts[e](x_flat[token_pos])
            routed_out[token_pos] += gates[token_pos, slot_pos].unsqueeze(-1) * y_e

        self.last_scores = scores.detach()
        self.last_topk_idx = topk_idx.detach()

        return (x_flat + shared_out + routed_out).view(B, T, D)

    def balance_loss(self) -> torch.Tensor:
        if self.last_scores is None or self.last_topk_idx is None:
            return torch.tensor(0.0)
        N = self.last_scores.shape[0]

        f = torch.zeros(self.n_routed, device=self.last_scores.device)
        for e in range(self.n_routed):
            f[e] = (self.last_topk_idx == e).sum().float()
        f = f * self.n_routed / (self.top_k * N)

        s_norm = self.last_scores / (self.last_scores.sum(dim=-1, keepdim=True) + 1e-9)
        P = s_norm.mean(dim=0)

        return self.balance_alpha * (f * P).sum()

    @torch.no_grad()
    def update_expert_bias(self):
        if self.last_scores is None or self.last_topk_idx is None:
            return
        load = torch.zeros(self.n_routed, device=self.expert_bias.device)
        for e in range(self.n_routed):
            load[e] = (self.last_topk_idx == e).sum().float()
        target = load.mean()
        self.expert_bias[load > target] -= self.gamma
        self.expert_bias[load < target] += self.gamma


class TransformerBlock(nn.Module):
    def __init__(self, config: ModelConfig, use_moe: bool):
        super().__init__()
        self.attn_norm = RMSNorm(config.d_model)
        self.attn = MLA(
            d_model=config.d_model,
            n_heads=config.n_heads,
            d_head=config.d_head,
            d_kv_latent=config.d_kv_latent,
            d_q_latent=config.d_q_latent,
            d_head_rope=config.d_head_rope,
            max_seq_len=config.max_seq_len,
        )
        self.ffn_norm = RMSNorm(config.d_model)
        if use_moe:
            self.ffn = DeepSeekMoE(
                d_model=config.d_model,
                d_ff_expert=config.d_ff_expert,
                n_routed=config.n_routed_experts,
                top_k=config.top_k,
                n_shared=config.n_shared_experts,
                gamma=config.gamma,
                balance_alpha=config.balance_alpha,
            )
        else:
            self.ffn = SwiGLU(config.d_model, config.d_ff_dense)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.attn_norm(x))
        x = x + self.ffn(self.ffn_norm(x))
        return x


class DeepSeekV3(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.layers = nn.ModuleList([
            TransformerBlock(config, use_moe=(i >= config.n_dense_layers))
            for i in range(config.n_layers)
        ])
        self.final_norm = RMSNorm(config.d_model)
        self.output_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.embedding.weight = self.output_head.weight

    def forward(self, token_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.embedding(token_ids)
        for layer in self.layers:
            x = layer(x)
        hidden = self.final_norm(x)
        logits = self.output_head(hidden)
        return logits, hidden

    def moe_layers(self):
        for layer in self.layers:
            if isinstance(layer.ffn, DeepSeekMoE):
                yield layer.ffn

    def balance_loss(self) -> torch.Tensor:
        losses = [moe.balance_loss() for moe in self.moe_layers()]
        if not losses:
            return torch.tensor(0.0)
        return torch.stack(losses).sum()

    @torch.no_grad()
    def update_expert_biases(self):
        for moe in self.moe_layers():
            moe.update_expert_bias()


def save_checkpoint(model: DeepSeekV3, config: ModelConfig, stoi: dict, itos: dict, out_dir: str = "checkpoint"):
    import json, os
    from safetensors.torch import save_model

    os.makedirs(out_dir, exist_ok=True)
    save_model(model, os.path.join(out_dir, "model.safetensors"))
    with open(os.path.join(out_dir, "config.json"), "w") as f:
        json.dump(vars(config), f, indent=2)
    with open(os.path.join(out_dir, "tokenizer.json"), "w") as f:
        json.dump({"stoi": stoi, "itos": {str(k): v for k, v in itos.items()}}, f)
    print(f"Saved checkpoint to {out_dir}/")


def load_checkpoint(out_dir: str = "checkpoint", device: torch.device = torch.device("cpu")):
    import json
    from safetensors.torch import load_model

    with open(f"{out_dir}/config.json") as f:
        config = ModelConfig(**json.load(f))
    with open(f"{out_dir}/tokenizer.json") as f:
        tok = json.load(f)
    stoi = tok["stoi"]
    itos = {int(k): v for k, v in tok["itos"].items()}

    model = DeepSeekV3(config).to(device)
    load_model(model, f"{out_dir}/model.safetensors", device=str(device))
    return model, config, stoi, itos


def push_to_hub(repo_id: str, out_dir: str = "checkpoint"):
    from huggingface_hub import HfApi
    api = HfApi(token=os.getenv("HF_TOKEN"))
    api.create_repo(repo_id, exist_ok=True)
    api.upload_folder(
        folder_path=out_dir,
        repo_id=repo_id,
        allow_patterns=["model.safetensors", "config.json", "tokenizer.json"],
    )
    print(f"Pushed to https://huggingface.co/{repo_id}")


MAX_EXAMPLES = 100_000


def load_dataset(config: ModelConfig):
    data_dir = "data"
    cache_file = os.path.join(data_dir, "tiny_codes_text.txt")

    if os.path.exists(cache_file):
        with open(cache_file, "r") as f:
            text = f.read()
    else:
        from datasets import load_dataset as hf_load

        os.makedirs(data_dir, exist_ok=True)
        print("Downloading nampdn-ai/tiny-codes from HuggingFace...")
        ds = hf_load("nampdn-ai/tiny-codes", split="train")
        if len(ds) > MAX_EXAMPLES:
            ds = ds.select(range(MAX_EXAMPLES))

        texts = []
        for row in ds:
            prompt = row.get("prompt", "") or ""
            response = row.get("response", "") or ""
            texts.append(f"{prompt}\n{response}\n\n")
        text = "".join(texts)

        with open(cache_file, "w") as f:
            f.write(text)
        print(f"Cached {len(texts)} examples ({len(text):,} chars)")

    chars = sorted(set(text))
    stoi = {c: i for i, c in enumerate(chars)}
    itos = {i: c for c, i in stoi.items()}
    actual_vocab = len(chars)
    print(f"Dataset: {len(text):,} chars, {actual_vocab} unique tokens")

    data = torch.tensor([stoi[c] for c in text], dtype=torch.long)
    split = int(0.9 * len(data))
    return data[:split], data[split:], actual_vocab, stoi, itos


def get_batch(data: torch.Tensor, config: ModelConfig, device: torch.device):
    ix = torch.randint(0, len(data) - config.max_seq_len - 1, (config.batch_size,))
    x = torch.stack([data[i : i + config.max_seq_len] for i in ix]).to(device)
    y = torch.stack([data[i + 1 : i + 1 + config.max_seq_len] for i in ix]).to(device)
    return x, y


@torch.no_grad()
def estimate_loss(model: DeepSeekV3, val_data: torch.Tensor, config: ModelConfig, device: torch.device, n_batches: int = 10):
    model.eval()
    losses = []
    for _ in range(n_batches):
        x, y = get_batch(val_data, config, device)
        logits, _ = model(x)
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
        losses.append(loss.item())
    model.train()
    return sum(losses) / len(losses)


def generate(model: DeepSeekV3, itos: dict, stoi: dict, prompt: str, max_new: int = 200, device: torch.device = torch.device("cpu")):
    model.eval()
    ids = torch.tensor([stoi.get(c, 0) for c in prompt], dtype=torch.long, device=device).unsqueeze(0)
    for _ in range(max_new):
        ctx = ids[:, -model.config.max_seq_len:]
        logits, _ = model(ctx)
        probs = F.softmax(logits[:, -1, :], dim=-1)
        nxt = torch.multinomial(probs, 1)
        ids = torch.cat([ids, nxt], dim=1)
    return "".join(itos.get(t, "?") for t in ids[0].tolist())


def train():
    import math

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = ModelConfig(max_seq_len=128, vocab_size=256)

    train_data, val_data, actual_vocab, stoi, itos = load_dataset(config)
    config.vocab_size = actual_vocab

    model = DeepSeekV3(config).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    dense_count = sum(1 for l in model.layers if not isinstance(l.ffn, DeepSeekMoE))
    moe_count = sum(1 for l in model.layers if isinstance(l.ffn, DeepSeekMoE))
    print(f"Model: {total_params / 1e6:.1f}M params, {dense_count} dense + {moe_count} MoE layers, device={device}")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.lr,
        betas=(0.9, 0.95),
        weight_decay=config.weight_decay,
    )

    training_log = {
        "config": vars(config),
        "total_params": total_params,
        "device": str(device),
        "dataset": "nampdn-ai/tiny-codes",
        "steps": [],
    }

    model.train()
    t0 = time.time()

    for step in range(1, config.train_steps + 1):
        x, y = get_batch(train_data, config, device)
        logits, _ = model(x)

        main_loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
        bal_loss = model.balance_loss()
        total_loss = main_loss + bal_loss

        optimizer.zero_grad()
        total_loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        model.update_expert_biases()

        if step % config.log_interval == 0 or step == 1:
            val_loss = estimate_loss(model, val_data, config, device)
            val_bpb = val_loss / math.log(2)
            elapsed = time.time() - t0
            print(f"step {step:>4d} | {elapsed:6.1f}s | train {main_loss.item():.3f} | val {val_loss:.3f} | val_bpb {val_bpb:.3f} | bal {bal_loss.item():.6f}")

            training_log["steps"].append({
                "step": step,
                "elapsed_s": round(elapsed, 2),
                "train_loss": round(main_loss.item(), 4),
                "val_loss": round(val_loss, 4),
                "val_bpb": round(val_bpb, 4),
                "balance_loss": round(bal_loss.item(), 6),
            })

    final_val = estimate_loss(model, val_data, config, device)
    final_bpb = final_val / math.log(2)
    elapsed = time.time() - t0
    print(f"\n=== FINAL | {config.train_steps} steps in {elapsed:.1f}s | val_loss {final_val:.4f} | val_bpb {final_bpb:.4f} ===")

    training_log["final"] = {
        "total_steps": config.train_steps,
        "elapsed_s": round(elapsed, 2),
        "val_loss": round(final_val, 4),
        "val_bpb": round(final_bpb, 4),
    }

    with open("training_log.json", "w") as f:
        json.dump(training_log, f, indent=2)
    print("Saved training_log.json")

    print("\n--- Sample ---")
    print(generate(model, itos, stoi, "def ", device=device))

    save_checkpoint(model, config, stoi, itos)
    print("Training complete. Model saved to checkpoint/")


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "generate":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model, config, stoi, itos = load_checkpoint(device=device)
        prompt = sys.argv[2] if len(sys.argv) > 2 else "ROMEO:"
        print(generate(model, itos, stoi, prompt, device=device))
    elif len(sys.argv) > 1 and sys.argv[1] == "push":
        repo_id = sys.argv[2]
        push_to_hub(repo_id)
    else:
        train()