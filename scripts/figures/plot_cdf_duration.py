import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "font.size": 9,
    "axes.labelsize": 10,
    "xtick.labelsize": 8.5,
    "ytick.labelsize": 8.5,
    "legend.fontsize": 8.5,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,
    "text.usetex": False,
})

NORMAL_COLOR = "#2166ac"
DOS_COLOR = "#b2182b"

df = pd.read_csv("/home/is/hang-qi/llm-dos/datasets/llm_dos_cic_style.csv")
normal = df.loc[df["label"] == "normal", "total_time_seconds"]
dos = df.loc[df["label"] == "dos", "total_time_seconds"]

fig, axes = plt.subplots(1, 2, figsize=(7, 2.4), gridspec_kw={"width_ratios": [1.1, 1]})

# --- (a) Log-scale histogram ---
ax = axes[0]
bins = np.logspace(np.log10(0.5), np.log10(200), 40)
ax.hist(normal, bins=bins, alpha=0.75, label="Normal", color=NORMAL_COLOR, edgecolor="white", linewidth=0.4)
ax.hist(dos, bins=bins, alpha=0.75, label="DoS", color=DOS_COLOR, edgecolor="white", linewidth=0.4)
ax.set_xscale("log")
ax.set_xlabel("Request duration (s)")
ax.set_ylabel("Number of requests")
ax.xaxis.set_major_formatter(ticker.ScalarFormatter())
ax.xaxis.set_minor_formatter(ticker.NullFormatter())
ax.set_xticks([1, 2, 5, 10, 20, 50, 100])
ax.get_xaxis().get_major_formatter().set_scientific(False)
ax.legend(frameon=False, loc="upper right")
ax.set_title("(a)", fontsize=10, loc="left", fontweight="bold", pad=4)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

# --- (b) Box plot with jittered dots and median annotations ---
ax = axes[1]

bp = ax.boxplot(
    [normal, dos],
    labels=["Normal", "DoS"],
    widths=0.45,
    patch_artist=True,
    showfliers=False,
    medianprops=dict(color="black", linewidth=1.5),
    whiskerprops=dict(linewidth=0.8),
    capprops=dict(linewidth=0.8),
)
bp["boxes"][0].set_facecolor(NORMAL_COLOR)
bp["boxes"][0].set_alpha(0.5)
bp["boxes"][1].set_facecolor(DOS_COLOR)
bp["boxes"][1].set_alpha(0.5)

rng = np.random.default_rng(42)
for i, (data, color) in enumerate([(normal, NORMAL_COLOR), (dos, DOS_COLOR)], start=1):
    subsample = data.sample(n=min(150, len(data)), random_state=42)
    jitter = rng.uniform(-0.12, 0.12, size=len(subsample))
    ax.scatter(
        i + jitter, subsample, alpha=0.35, s=8, color=color,
        edgecolors="none", zorder=3,
    )

for i, data in enumerate([normal, dos], start=1):
    med = data.median()
    ax.annotate(
        f"{med:.1f} s",
        xy=(i, med),
        xytext=(i + 0.38, med + (2.0 if i == 1 else 4.0)),
        fontsize=8,
        fontweight="bold",
        arrowprops=dict(arrowstyle="-", lw=0.7, color="gray"),
        ha="left", va="center",
    )

ax.set_ylabel("Request duration (s)")
ax.set_title("(b)", fontsize=10, loc="left", fontweight="bold", pad=4)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

plt.tight_layout(w_pad=2.5)
fig.savefig("/home/is/hang-qi/llm-dos/qian/26globecom/ieee-enhanced-main/figures/cdf_duration.pdf")
fig.savefig("/home/is/hang-qi/llm-dos/qian/26globecom/ieee-enhanced-main/figures/cdf_duration.png")
print("Saved cdf_duration.pdf and cdf_duration.png")
plt.close()
