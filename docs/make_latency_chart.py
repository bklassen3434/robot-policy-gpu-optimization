"""Generate docs/latency.png — fused residual+LayerNorm kernel vs PyTorch.

Grouped bar chart (magnitude comparison): op-level latency in microseconds for the
PyTorch reference (elementwise add + layer_norm) vs the custom fused CUDA kernel, at
the tensor shapes that occur in ACT. CUDA-event timed on an RTX 4090; numbers from
`kernels/bench.py`. Direct value labels + per-group speedup so identity/magnitude are
never color-alone.

    python docs/make_latency_chart.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

# --- data (from kernels/bench.py, RTX 4090, iters=2000) ---
LABELS = [
    "(302, 512)\nencoder · infer",
    "(100, 512)\ndecoder · infer",
    "(2416, 512)\nencoder · train",
    "(800, 512)\ndecoder · train",
]
PYTORCH_US = [8.46, 8.11, 11.85, 8.16]
FUSED_US = [6.41, 5.81, 8.24, 5.71]
SPEEDUP = [p / f for p, f in zip(PYTORCH_US, FUSED_US)]

# --- style: recessive axes/grid, thin marks, ink text ---
INK = "#0F172A"
MUTED = "#64748B"
GRID = "#E2E8F0"
BASELINE = "#94A3B8"  # muted slate — the reference (CVD-safe vs blue by luminance+hue)
KERNEL = "#2563EB"  # confident blue — the hero (custom kernel)

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
    "text.color": INK,
    "axes.edgecolor": MUTED,
    "axes.labelcolor": INK,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
})

fig, ax = plt.subplots(figsize=(8.2, 4.6), dpi=200)

x = range(len(LABELS))
w = 0.36
bars_ref = ax.bar([i - w / 2 for i in x], PYTORCH_US, w, label="PyTorch (add + layer_norm)",
                  color=BASELINE, zorder=3)
bars_ker = ax.bar([i + w / 2 for i in x], FUSED_US, w, label="Fused CUDA kernel",
                  color=KERNEL, zorder=3)

# direct value labels on each bar
for bars in (bars_ref, bars_ker):
    for b in bars:
        ax.annotate(f"{b.get_height():.1f}", (b.get_x() + b.get_width() / 2, b.get_height()),
                    xytext=(0, 3), textcoords="offset points", ha="center", va="bottom",
                    fontsize=8.5, color=MUTED)

# per-group speedup badge above the pair
ymax = max(PYTORCH_US)
for i, s in enumerate(SPEEDUP):
    top = max(PYTORCH_US[i], FUSED_US[i])
    ax.annotate(f"{s:.2f}×", (i, top), xytext=(0, 20), textcoords="offset points",
                ha="center", va="bottom", fontsize=11, fontweight="bold", color=KERNEL)

ax.set_ylim(0, ymax * 1.28)
ax.set_ylabel("latency per call  (µs, lower is better)", fontsize=10)
ax.set_xticks(list(x))
ax.set_xticklabels(LABELS, fontsize=9)
ax.set_title("Fused residual + LayerNorm kernel vs PyTorch  ·  RTX 4090",
             fontsize=13, fontweight="bold", pad=30, loc="left")
ax.annotate("CUDA-event timed, 2000 iters · output-parity with PyTorch (rtol 1e-4) · "
            "1.3–1.4× on the op (end-to-end ~1–2%: model is GEMM/conv-bound)",
            (0, 1), xytext=(0, 12), textcoords="offset points", xycoords="axes fraction",
            fontsize=8.5, color=MUTED, va="bottom")

# recessive grid, no top/right spines
ax.yaxis.grid(True, color=GRID, linewidth=1, zorder=0)
ax.set_axisbelow(True)
for side in ("top", "right"):
    ax.spines[side].set_visible(False)
for side in ("left", "bottom"):
    ax.spines[side].set_color(GRID)
ax.tick_params(length=0)
ax.legend(frameon=False, fontsize=9.5, loc="upper right", ncol=1)

fig.tight_layout()
out = Path(__file__).parent / "latency.png"
fig.savefig(out, bbox_inches="tight", facecolor="white")
print(f"wrote {out}")
