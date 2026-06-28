"""Generate Figure: Uniform-vs-tiered certification comparison.

Reads:
  results/contribution1/uniform_comparison.csv
  results/contribution1/scorecard.csv

Writes:
  results/figures/uniform_vs_tiered.pdf
  results/figures/uniform_vs_tiered.png

Run:
  python src/plot_uniform_vs_tiered.py
"""

import csv
import pathlib
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

# ── style ──────────────────────────────────────────────────────────────────
sns.set_theme(style="whitegrid", font_scale=1.05)
plt.rcParams.update({
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "grid.linestyle":    "--",
    "grid.alpha":        0.45,
    "axes.axisbelow":    True,
})

# ── palette ────────────────────────────────────────────────────────────────
pal = sns.color_palette("colorblind")
CERT_C = pal[2]          # green
TRIV_C = pal[1]          # orange
UNCE_C = "#d8d8d8"       # neutral grey

TIERED_C = pal[2]        # green  — proposed
U02_C    = pal[0]        # blue   — too conservative
U10_C    = pal[3]        # red    — too permissive

ANNOT_EFF  = pal[0]      # blue arrow  — efficiency losses
ANNOT_SAFE = pal[3]      # red arrow   — safety failures

# ── load data ──────────────────────────────────────────────────────────────
C1    = pathlib.Path("results/contribution1")
rows  = list(csv.DictReader(open(C1 / "uniform_comparison.csv")))
score = {r["class"]: r for r in csv.DictReader(open(C1 / "scorecard.csv"))}

tier_order = {"Critical": 0, "Important": 1, "Benign": 2}
rows_sorted = sorted(rows, key=lambda r: (tier_order[score[r["class"]]["tier"]], r["class"]))
lookup = {r["class"]: r for r in rows_sorted}

configs = [
    ("tiered", "tiered_status", "Tiered α\n(proposed)"),
    ("u0.02",  "u0.02_status",  "Uniform\nα = 0.02"),
    ("u0.05",  "u0.05_status",  "Uniform\nα = 0.05"),
    ("u0.10",  "u0.10_status",  "Uniform\nα = 0.10"),
]

counts = {}
for key, col, _ in configs:
    c = {"certified": 0, "trivial": 0, "uncertifiable": 0}
    for r in rows_sorted:
        c[r[col]] += 1
    counts[key] = c

cert_vals = [counts[k]["certified"]    for k, _, _ in configs]
triv_vals = [counts[k]["trivial"]       for k, _, _ in configs]
unce_vals = [counts[k]["uncertifiable"] for k, _, _ in configs]

# ── layout ─────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(13, 6.8))
gs  = fig.add_gridspec(1, 2, width_ratios=[1.75, 1], wspace=0.38,
                        left=0.07, right=0.97, top=0.82, bottom=0.13)
ax  = fig.add_subplot(gs[0])
ax2 = fig.add_subplot(gs[1])

# ── LEFT: stacked bar chart ────────────────────────────────────────────────
x = np.arange(len(configs))
w = 0.50

ax.bar(x, unce_vals, w, color=UNCE_C, label="Uncertifiable",  zorder=2, linewidth=0)
ax.bar(x, triv_vals, w, bottom=unce_vals,
       color=TRIV_C, label="Trivial  (λ = 0)", zorder=2, linewidth=0)
ax.bar(x, cert_vals, w,
       bottom=[u + t for u, t in zip(unce_vals, triv_vals)],
       color=CERT_C, label="Certified", zorder=2, linewidth=0)

# in-bar value labels — skip segments too thin to label
MIN_SHOW = 1
for i, (u, t, c) in enumerate(zip(unce_vals, triv_vals, cert_vals)):
    if u >= MIN_SHOW:
        ax.text(i, u / 2, str(u),
                ha="center", va="center", fontsize=12, color="#444", fontweight="bold")
    if t >= MIN_SHOW:
        ax.text(i, u + t / 2, str(t),
                ha="center", va="center", fontsize=12, color="white", fontweight="bold")
    if c >= MIN_SHOW:
        ax.text(i, u + t + c / 2, str(c),
                ha="center", va="center", fontsize=12, color="white", fontweight="bold")

# ── failure-mode annotations — placed above plot, no arrow collision ───────
# top of tallest bar is 24; annotations sit at y=26 (above ylim=25)
# arrows point to bar tops from a text box in the headroom above
TOP = 24  # max bar height
ANNOT_Y = 29.5  # text y — well above bars (ylim top = 32)

ax.annotate(
    "9 efficiency losses\n(α = 0.02 too strict)",
    xy=(1, TOP + 0.3), xytext=(0.55, ANNOT_Y),
    fontsize=9, color=ANNOT_EFF, ha="center", fontweight="bold", va="bottom",
    arrowprops=dict(arrowstyle="-|>", color=ANNOT_EFF, lw=1.5,
                    connectionstyle="arc3,rad=-0.18"),
    bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=ANNOT_EFF, lw=1.2, alpha=0.95),
)

ax.annotate(
    "6 safety failures\n(α = 0.10 too loose)",
    xy=(3, TOP + 0.3), xytext=(3.45, ANNOT_Y),
    fontsize=9, color=ANNOT_SAFE, ha="center", fontweight="bold", va="bottom",
    arrowprops=dict(arrowstyle="-|>", color=ANNOT_SAFE, lw=1.5,
                    connectionstyle="arc3,rad=0.18"),
    bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=ANNOT_SAFE, lw=1.2, alpha=0.95),
)

ax.set_xticks(x)
ax.set_xticklabels([lbl for _, _, lbl in configs], fontsize=11)
ax.set_ylabel("Number of classes  (n = 24 supported)", fontsize=11, labelpad=8)
ax.set_ylim(0, 34)
ax.set_yticks(range(0, 25, 4))
ax.yaxis.set_tick_params(labelsize=10)
ax.set_title(
    "Certification outcome by α strategy\n"
    "Binomial LTT  ·  δ = 0.10 / 24  ·  24 supported classes",
    fontsize=12, pad=14, loc="left",
)

# legend bottom-left so it doesn't compete with the annotations at top
ax.legend(
    handles=[
        mpatches.Patch(facecolor=CERT_C, label="Certified",        linewidth=0),
        mpatches.Patch(facecolor=TRIV_C, label="Trivial  (λ = 0)", linewidth=0),
        mpatches.Patch(facecolor=UNCE_C, label="Uncertifiable",    linewidth=0, ec="#aaa"),
    ],
    loc="lower right", fontsize=10, framealpha=0.92,
    handlelength=1.2, handleheight=1.0, borderpad=0.7,
)

# ── RIGHT: certified count per tier ────────────────────────────────────────
tiers    = ["Critical", "Important", "Benign"]
tier_cls = {
    t: [r["class"] for r in rows_sorted if score[r["class"]]["tier"] == t]
    for t in tiers
}

configs_right = [
    ("tiered_status", "Tiered (proposed)", TIERED_C),
    ("u0.02_status",  "Uniform  α = 0.02", U02_C),
    ("u0.10_status",  "Uniform  α = 0.10", U10_C),
]

bar_w = 0.22
x2    = np.arange(len(tiers))

for j, (col, label, color) in enumerate(configs_right):
    cert_counts = [
        sum(1 for c in tier_cls[t] if lookup[c][col] == "certified")
        for t in tiers
    ]
    bars = ax2.bar(
        x2 + (j - 1) * bar_w, cert_counts, bar_w,
        color=color, label=label, alpha=0.90, zorder=2, linewidth=0,
    )
    for b, v in zip(bars, cert_counts):
        if v > 0:
            ax2.text(
                b.get_x() + b.get_width() / 2,
                v + 0.15,
                str(v),
                ha="center", va="bottom", fontsize=11,
                color=color, fontweight="bold",
            )

# n= labels below x-axis
tier_totals = [len(tier_cls[t]) for t in tiers]
for i, tot in enumerate(tier_totals):
    ax2.text(i, -0.7, f"n = {tot}", ha="center", va="top", fontsize=9, color="#666")

ax2.set_xticks(x2)
ax2.set_xticklabels(tiers, fontsize=11)
ax2.set_ylabel("Certified classes per tier", fontsize=11, labelpad=8)
ax2.set_ylim(-1.1, 12.5)
ax2.set_yticks(range(0, 12, 2))
ax2.yaxis.set_tick_params(labelsize=10)
ax2.set_title(
    "Certified count by tier\nCertified classes only",
    fontsize=12, pad=14, loc="left",
)
ax2.legend(fontsize=9, framealpha=0.92, loc="upper right",
           handlelength=1.2, handleheight=1.0, borderpad=0.7)

# ── save ───────────────────────────────────────────────────────────────────
out_dir = pathlib.Path("results/figures")
out_dir.mkdir(parents=True, exist_ok=True)

for ext in ("pdf", "png"):
    p = out_dir / f"uniform_vs_tiered.{ext}"
    fig.savefig(p, bbox_inches="tight", dpi=150)
    print(f"Saved: {p}")
