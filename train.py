"""DeepSeek-V3 toy model + training loop.
This is the ONLY file the agent should edit.
"""

import json
import time
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from prepare import prepare, get_batch, evaluate, generate, save_checkpoint, TIME_BUDGET_SECONDS

# ---------------------------------------------------------------------------
# Config -- agent can tune these
# ---------------------------------------------------------------------------

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
    vocab_size: int = 65
    max_seq_len: int = 128
    gamma: float = 1e-3
    balance_alpha: float = 1e-4
    lr: float = 3e-4
    weight_decay: float = 0.1
    batch_size: int = 8
    log_interval: int = 25

# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------

def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def precompute_rope_freqs(dim: int, max_seq_len: int, theta: float = 10000.0):
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
        norm = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return x * norm * self.weight

# ---------------------------------------------------------------------------
# Multi-Head Latent Attention (MLA)
# ---------------------------------------------------------------------------

class MLA(nn.Module):
    def __init__(self, d_model, n_heads, d_head, d_kv_latent, d_q_latent, d_head_rope, max_seq_len):
        super().__init__()
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

        cos, sin = precompute_rope_freqs(d_head_rope, max_seq_len)
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)

    def forward(self, x):
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
        attn = torch.matmul(q, k.transpose(-1, -2)) * scale
        mask = torch.triu(torch.ones(seq_len, seq_len, device=x.device, dtype=torch.bool), diagonal=1)
        attn = attn.masked_fill(mask, float("-inf"))
        attn = F.softmax(attn, dim=-1)
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(bsz, seq_len, self.n_heads * self.d_head)
        return self.w_o(out)

# ---------------------------------------------------------------------------
# DeepSeekMoE
# ---------------------------------------------------------------------------

class DeepSeekMoE(nn.Module):
    def __init__(self, d_model, d_ff_expert, n_routed, top_k, n_shared=1, gamma=1e-3, balance_alpha=1e-4):
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

    def forward(self, x):
        B, T, D = x.shape
        N = B * T
        x_flat = x.reshape(N, D)

        scores = torch.sigmoid(self.router(x_flat))
        biased = scores + self.expert_bias.unsqueeze(0)
        _, topk_idx = torch.topk(biased, self.top_k, dim=-1)

        selected = torch.gather(scores, dim=1, index=topk_idx)
        gates = selected / (selected.sum(dim=-1, keepdim=True) + 1e-9)

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

    def balance_loss(self):
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

# ---------------------------------------------------------------------------
# Transformer block + full model
# ---------------------------------------------------------------------------

class TransformerBlock(nn.Module):
    def __init__(self, config: ModelConfig, use_moe: bool):
        super().__init__()
        self.attn_norm = RMSNorm(config.d_model)
        self.attn = MLA(
            config.d_model, config.n_heads, config.d_head,
            config.d_kv_latent, config.d_q_latent, config.d_head_rope, config.max_seq_len,
        )
        self.ffn_norm = RMSNorm(config.d_model)
        if use_moe:
            self.ffn = DeepSeekMoE(
                config.d_model, config.d_ff_expert, config.n_routed_experts,
                config.top_k, config.n_shared_experts, config.gamma, config.balance_alpha,
            )
        else:
            self.ffn = SwiGLU(config.d_model, config.d_ff_dense)

    def forward(self, x):
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

    def forward(self, token_ids):
        x = self.embedding(token_ids)
        for layer in self.layers:
            x = layer(x)
        x = self.final_norm(x)
        return self.output_head(x)

    def moe_layers(self):
        for layer in self.layers:
            if isinstance(layer.ffn, DeepSeekMoE):
                yield layer.ffn

    def balance_loss(self):
        losses = [moe.balance_loss() for moe in self.moe_layers()]
        if not losses:
            return torch.tensor(0.0)
        return torch.stack(losses).sum()

    @torch.no_grad()
    def update_expert_biases(self):
        for moe in self.moe_layers():
            moe.update_expert_bias()

# ---------------------------------------------------------------------------
# Training -- agent can change anything above or below
# ---------------------------------------------------------------------------

LOG_FILE = "training_log.json"


def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_data, val_data, vocab_size, stoi, itos = prepare()

    config = ModelConfig(vocab_size=vocab_size)
    model = DeepSeekV3(config).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    dense_n = sum(1 for l in model.layers if not isinstance(l.ffn, DeepSeekMoE))
    moe_n = sum(1 for l in model.layers if isinstance(l.ffn, DeepSeekMoE))
    print(f"Model: {total_params/1e6:.1f}M params, {dense_n} dense + {moe_n} MoE layers, device={device}")

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.lr, betas=(0.9, 0.95), weight_decay=config.weight_decay,
    )

    training_log = {
        "config": vars(config),
        "total_params": total_params,
        "device": str(device),
        "dataset": "nampdn-ai/tiny-codes",
        "steps": [],
    }

    model.train()
    step = 0
    t0 = time.time()

    while True:
        elapsed = time.time() - t0
        if elapsed >= TIME_BUDGET_SECONDS:
            break
        step += 1

        x, y = get_batch(train_data, config.batch_size, config.max_seq_len, device)
        logits = model(x)
        main_loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
        bal_loss = model.balance_loss()
        total_loss = main_loss + bal_loss

        optimizer.zero_grad()
        total_loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        model.update_expert_biases()

        if step % config.log_interval == 0 or step == 1:
            val_loss, val_bpb = evaluate(model, val_data, config.batch_size, config.max_seq_len, device)
            elapsed = time.time() - t0
            print(f"step {step:>5d} | {elapsed:6.1f}s | train {main_loss.item():.3f} | val {val_loss:.3f} | val_bpb {val_bpb:.3f} | bal {bal_loss.item():.6f}")

            training_log["steps"].append({
                "step": step,
                "elapsed_s": round(elapsed, 2),
                "train_loss": round(main_loss.item(), 4),
                "val_loss": round(val_loss, 4),
                "val_bpb": round(val_bpb, 4),
                "balance_loss": round(bal_loss.item(), 6),
            })

    val_loss, val_bpb = evaluate(model, val_data, config.batch_size, config.max_seq_len, device)
    elapsed = time.time() - t0
    print(f"\n=== FINAL | {step} steps in {elapsed:.1f}s | val_loss {val_loss:.4f} | val_bpb {val_bpb:.4f} ===")

    training_log["final"] = {
        "total_steps": step,
        "elapsed_s": round(elapsed, 2),
        "val_loss": round(val_loss, 4),
        "val_bpb": round(val_bpb, 4),
    }

    with open(LOG_FILE, "w") as f:
        json.dump(training_log, f, indent=2)
    print(f"Saved {LOG_FILE}")

    print("\n--- Sample ---")
    print(generate(model, stoi, itos, "def ", device=device, max_seq_len=config.max_seq_len))

    save_checkpoint(model, config, stoi, itos)
    print("Training complete. Model saved to checkpoint/")


if __name__ == "__main__":
    train()
