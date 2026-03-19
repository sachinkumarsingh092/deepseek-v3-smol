"""Generate training visualizations from training_log.json.

Usage:
    uv run plot.py                              # defaults: training_log.json -> training_plots.png
    uv run plot.py training_log.json output.png  # custom paths
"""

import json
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


COLORS = {
    "train":     "#58a6ff",
    "val":       "#f78166",
    "bpb":       "#7ee787",
    "balance":   "#d2a8ff",
    "accent":    "#ff7b72",
    "highlight": "#ffa657",
}

BG       = "#0d1117"
PANEL_BG = "#161b22"
BORDER   = "#30363d"
GRID     = "#21262d"
TEXT     = "#c9d1d9"
TEXT_HI  = "#f0f6fc"
TEXT_DIM = "#8b949e"


def setup_style():
    plt.rcParams.update({
        "figure.facecolor":   BG,
        "axes.facecolor":     PANEL_BG,
        "axes.edgecolor":     BORDER,
        "axes.labelcolor":    TEXT,
        "axes.titlecolor":    TEXT_HI,
        "text.color":         TEXT,
        "xtick.color":        TEXT_DIM,
        "ytick.color":        TEXT_DIM,
        "grid.color":         GRID,
        "grid.linestyle":     "--",
        "grid.alpha":         0.6,
        "font.family":        "monospace",
        "font.size":          10,
        "axes.titlesize":     13,
        "axes.labelsize":     11,
        "legend.fontsize":    9,
        "legend.facecolor":   PANEL_BG,
        "legend.edgecolor":   BORDER,
        "legend.labelcolor":  TEXT,
        "savefig.dpi":        200,
        "savefig.bbox":       "tight",
        "savefig.facecolor":  BG,
    })


def ema(values: list[float], alpha: float = 0.75) -> list[float]:
    """Exponential moving average for smoother curves."""
    out = []
    s = values[0]
    for v in values:
        s = alpha * s + (1 - alpha) * v
        out.append(s)
    return out


def plot_curve(ax, steps, raw, smoothed, color, label, ylabel, title, sci_y=False):
    ax.plot(steps, raw, color=color, alpha=0.25, linewidth=0.8)
    ax.plot(steps, smoothed, color=color, linewidth=2.2, label=label)
    ax.set_xlabel("Step")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(loc="upper right")
    ax.grid(True)
    if sci_y:
        ax.ticklabel_format(style="scientific", axis="y", scilimits=(-3, 3))


def plot_training(log: dict, output_path: str = "training_plots.png"):
    steps_data = log["steps"]
    if not steps_data:
        print("No training steps found in log.")
        return

    steps      = [s["step"]         for s in steps_data]
    train_loss = [s["train_loss"]   for s in steps_data]
    val_loss   = [s["val_loss"]     for s in steps_data]
    val_bpb    = [s["val_bpb"]      for s in steps_data]
    bal_loss   = [s["balance_loss"] for s in steps_data]

    config      = log.get("config", {})
    total_params = log.get("total_params", 0)
    final       = log.get("final", {})
    dataset     = log.get("dataset", "unknown")

    use_ema = len(steps) > 4

    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle(
        f"DeepSeek-V3 Training  ·  {total_params / 1e6:.1f}M params  ·  "
        f"{final.get('total_steps', '?')} steps in {final.get('elapsed_s', '?')}s  ·  {dataset}",
        fontsize=14, fontweight="bold", color=TEXT_HI, y=0.98,
    )

    # ── Panel 1: Train & Val Loss ──────────────────────────────────────
    ax = axes[0, 0]
    ax.plot(steps, train_loss, color=COLORS["train"], alpha=0.25, linewidth=0.8)
    ax.plot(steps, val_loss,   color=COLORS["val"],   alpha=0.25, linewidth=0.8)
    if use_ema:
        ax.plot(steps, ema(train_loss), color=COLORS["train"], linewidth=2.2, label="Train (EMA)")
        ax.plot(steps, ema(val_loss),   color=COLORS["val"],   linewidth=2.2, label="Val (EMA)")
    else:
        ax.plot(steps, train_loss, color=COLORS["train"], linewidth=2.2, label="Train")
        ax.plot(steps, val_loss,   color=COLORS["val"],   linewidth=2.2, label="Val")
    ax.set_xlabel("Step")
    ax.set_ylabel("Cross-Entropy Loss")
    ax.set_title("Training & Validation Loss")
    ax.legend(loc="upper right")
    ax.grid(True)

    # ── Panel 2: Validation BPB ────────────────────────────────────────
    ax = axes[0, 1]
    bpb_smooth = ema(val_bpb) if use_ema else val_bpb
    ax.plot(steps, val_bpb, color=COLORS["bpb"], alpha=0.3, linewidth=0.8)
    ax.plot(steps, bpb_smooth, color=COLORS["bpb"], linewidth=2.5, label="Val BPB (EMA)" if use_ema else "Val BPB")

    min_bpb = min(val_bpb)
    min_idx = val_bpb.index(min_bpb)
    ax.scatter(
        [steps[min_idx]], [min_bpb],
        color=COLORS["accent"], s=90, zorder=5,
        edgecolors="white", linewidth=1.5,
    )
    ax.annotate(
        f"Best: {min_bpb:.3f}",
        xy=(steps[min_idx], min_bpb),
        xytext=(15, 15), textcoords="offset points",
        color=COLORS["accent"], fontweight="bold", fontsize=10,
        arrowprops=dict(arrowstyle="->", color=COLORS["accent"], lw=1.5),
    )
    ax.set_xlabel("Step")
    ax.set_ylabel("Bits per Byte")
    ax.set_title("Validation BPB")
    ax.legend(loc="upper right")
    ax.grid(True)

    # ── Panel 3: Balance Loss ──────────────────────────────────────────
    ax = axes[1, 0]
    ax.plot(steps, bal_loss, color=COLORS["balance"], linewidth=2, label="Balance Loss")
    if use_ema:
        ax.plot(steps, ema(bal_loss, alpha=0.6), color=COLORS["balance"],
                linewidth=1.2, linestyle="--", alpha=0.6, label="EMA")
    ax.set_xlabel("Step")
    ax.set_ylabel("Balance Loss")
    ax.set_title("MoE Balance Loss")
    ax.legend(loc="upper right")
    ax.grid(True)
    ax.ticklabel_format(style="scientific", axis="y", scilimits=(-3, 3))

    # ── Panel 4: Config Summary ────────────────────────────────────────
    ax = axes[1, 1]
    ax.axis("off")

    n_layers      = config.get("n_layers", "?")
    n_dense       = config.get("n_dense_layers", "?")
    n_moe         = n_layers - n_dense if isinstance(n_layers, int) and isinstance(n_dense, int) else "?"

    rows = [
        ["Model",          f"DeepSeek-V3 ({total_params / 1e6:.1f}M params)"],
        ["Dataset",        dataset],
        ["d_model",        str(config.get("d_model", "?"))],
        ["Layers",         f"{n_layers} ({n_dense} dense + {n_moe} MoE)"],
        ["Heads / d_head", f"{config.get('n_heads', '?')} / {config.get('d_head', '?')}"],
        ["Experts",        f"{config.get('n_routed_experts', '?')} routed, "
                           f"{config.get('n_shared_experts', '?')} shared, "
                           f"top-{config.get('top_k', '?')}"],
        ["Seq Len",        str(config.get("max_seq_len", "?"))],
        ["Batch Size",     str(config.get("batch_size", "?"))],
        ["LR",             str(config.get("lr", "?"))],
        ["Final Val Loss", f"{final['val_loss']:.4f}" if "val_loss" in final else "?"],
        ["Final Val BPB",  f"{final['val_bpb']:.4f}"  if "val_bpb"  in final else "?"],
        ["Total Steps",    str(final.get("total_steps", "?"))],
        ["Wall Time",      f"{final.get('elapsed_s', '?')}s"],
    ]

    table = ax.table(
        cellText=rows,
        colLabels=["Parameter", "Value"],
        cellLoc="left",
        loc="center",
        colWidths=[0.35, 0.65],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.35)

    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor(BORDER)
        if row == 0:
            cell.set_facecolor("#21262d")
            cell.set_text_props(color=TEXT_HI, fontweight="bold")
        else:
            cell.set_facecolor(PANEL_BG)
            cell.set_text_props(color=TEXT)

    ax.set_title("Training Configuration", pad=20)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(output_path)
    print(f"Saved {output_path}")
    plt.close()


def main():
    log_path = sys.argv[1] if len(sys.argv) > 1 else "training_log.json"
    output   = sys.argv[2] if len(sys.argv) > 2 else "training_plots.png"

    setup_style()

    with open(log_path) as f:
        log = json.load(f)

    plot_training(log, output)


if __name__ == "__main__":
    main()
