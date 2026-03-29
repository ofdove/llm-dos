#!/usr/bin/env python3
"""Generate early detection figure comparing full-trace vs window-matched models."""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "font.size": 9,
    "axes.labelsize": 10,
    "xtick.labelsize": 8.5,
    "ytick.labelsize": 8.5,
    "legend.fontsize": 8,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,
})

windows =    [1,     2,     3,     5,     10,    15]
full_f1 =    [0.8791, 0.9875, 0.9899, 0.9899, 0.9950, 0.9950]
full_prec =  [0.7843, 0.9899, 0.9949, 0.9949, 0.9950, 0.9950]
full_rec =   [1.0000, 0.9850, 0.9850, 0.9850, 0.9950, 0.9950]
matched_f1 = [0.9925, 0.9924, 0.9899, 0.9899, 0.9950, 0.9950]

fig, ax = plt.subplots(figsize=(5, 3.0))

ax.plot(windows, full_f1, "o-", color="#2166ac", linewidth=1.5, markersize=5,
        label="Full-trace model (F1)")
ax.plot(windows, matched_f1, "s-", color="#b2182b", linewidth=1.5, markersize=5,
        label="Window-matched model (F1)")
ax.plot(windows, full_prec, "^--", color="#2166ac", linewidth=0.9, markersize=4,
        alpha=0.5, label="Full-trace (Precision)")
ax.plot(windows, full_rec, "v--", color="#2166ac", linewidth=0.9, markersize=4,
        alpha=0.5, label="Full-trace (Recall)")

ax.set_xlabel("Detection window $T$ (seconds)")
ax.set_ylabel("Score")
ax.set_ylim(0.70, 1.015)
ax.set_xticks(windows)
ax.legend(fontsize=7.5, loc="lower right")
ax.grid(True, alpha=0.3)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

fig.tight_layout()
fig.savefig("qian/26globecom/ieee-enhanced-main/figures/early_detection.pdf")
fig.savefig("qian/26globecom/ieee-enhanced-main/figures/early_detection.png")
print("Saved early_detection.pdf and early_detection.png")
plt.close()
