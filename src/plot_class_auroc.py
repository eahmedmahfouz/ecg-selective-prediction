"""Per-class ROC curves — backbone discrimination for the 24 supported classes.

Supporting figure: explains why some certified classes need permissive thresholds
(weak backbone discrimination) vs. uncertifiable classes with high AUROC but
insufficient val-set positives (data-scale constraint, not discrimination failure).

Layout: 3 panels — Critical | Important | Benign.
  Solid line  = certified in at least one α config.
  Dashed line = uncertifiable or trivial at all configs.

Reads:
  checkpoints/acuity_test_logits.npy
  checkpoints/acuity_test_labels.npy
  checkpoints/acuity_labels/class_names.json
  results/contribution1/sensitivity_sweep.csv

Writes:
  results/figures/class_auroc.pdf
  results/figures/class_auroc.png

Run:
  python src/plot_class_auroc.py
"""

import csv
import json
import pathlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve

# ── style ──────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "figure.facecolor":   "white",
    "axes.facecolor":     "white",
    "axes.spines.top":    False,
    "axes.spines.right":  False,
    "axes.grid":          True,
    "axes.axisbelow":     True,
    "grid.color":         "#e8e8e8",
    "grid.linewidth":     0.8,
    "font.family":        "sans-serif",
    "font.size":          10,
})

TIER_C = {
    "Critical":  "#C44E52",
    "Important": "#DD8452",
    "Benign":    "#55A868",
}

# ── load ───────────────────────────────────────────────────────────────────────
CKP         = pathlib.Path("checkpoints")
test_logits = np.load(CKP / "acuity_test_logits.npy")
test_labels = np.load(CKP / "acuity_test_labels.npy")
class_names = json.load(open(CKP / "acuity_labels" / "class_names.json"))
probs       = 1 / (1 + np.exp(-test_logits))

C1   = pathlib.Path("results/contribution1")
rows = list(csv.DictReader(open(C1 / "sensitivity_sweep.csv")))

# certified = "certified" at tiered/primary config only (canonical definition, matches all tables)
tiered_row = {r["class"]: r for r in rows if r["config"] == "tiered"}
supported   = {}
for cls, row in tiered_row.items():
    supported[cls] = {
        "tier":   row["tier"],
        "cert":   row["status"] == "certified",
    }

# ── compute ROC curves ─────────────────────────────────────────────────────────
roc_data = {}
for cls, info in supported.items():
    idx = class_names.index(cls)
    y   = test_labels[:, idx]
    s   = probs[:, idx]
    if y.sum() < 2 or y.sum() >= len(y):
        continue
    fpr, tpr, _ = roc_curve(y, s)
    auc          = roc_auc_score(y, s)
    roc_data[cls] = {"fpr": fpr, "tpr": tpr, "auc": auc, **info}

# ── split by tier, sorted by AUROC descending ─────────────────────────────────
def tier_classes(tier):
    return sorted(
        [c for c, d in roc_data.items() if d["tier"] == tier],
        key=lambda c: -roc_data[c]["auc"],
    )

tiers       = ["Critical", "Important", "Benign"]
tier_cls    = {t: tier_classes(t) for t in tiers}
n_per_tier  = {t: len(tier_cls[t]) for t in tiers}

# per-class colour within each panel — use tab10 / tab20 for many classes
def make_palette(n):
    if n <= 10:
        return plt.cm.tab10(np.linspace(0, 0.9, n))
    return plt.cm.tab20(np.linspace(0, 0.95, n))

palettes = {t: make_palette(n_per_tier[t]) for t in tiers}
cls_color = {}
for t in tiers:
    for i, cls in enumerate(tier_cls[t]):
        cls_color[cls] = palettes[t][i]

# ── figure ─────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(15, 5.5),
                         gridspec_kw={"wspace": 0.28})
fig.subplots_adjust(left=0.06, right=0.98, top=0.88, bottom=0.13)

for ax, tier in zip(axes, tiers):
    classes = tier_cls[tier]
    pal     = palettes[tier]

    # diagonal reference
    ax.plot([0, 1], [0, 1], color="#bbbbbb", lw=1.0, ls="--", zorder=1)

    h_list, l_list = [], []
    for i, cls in enumerate(classes):
        d      = roc_data[cls]
        color  = cls_color[cls]
        ls     = "-" if d["cert"] else (0, (4, 2))
        lw     = 1.6 if d["cert"] else 1.2
        alpha  = 1.0 if d["cert"] else 0.65
        label  = f"{cls}  ({d['auc']:.3f})"
        line,  = ax.plot(d["fpr"], d["tpr"],
                         color=color, lw=lw, ls=ls, alpha=alpha, zorder=3)
        h_list.append(line)
        l_list.append(label)

    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.05)
    ax.set_xlabel("False positive rate", fontsize=10, labelpad=5)
    ax.set_ylabel("True positive rate", fontsize=10, labelpad=5)

    n_cert = sum(1 for c in classes if roc_data[c]["cert"])
    n_unc  = len(classes) - n_cert
    subtitle = f"{n_cert} certified"
    if n_unc:
        subtitle += f"  +  {n_unc} uncertifiable"
    budget = {"Critical": 0.02, "Important": 0.05, "Benign": 0.10}[tier]
    ax.set_title(
        f"{tier}  (α ≤ {budget})\n{subtitle}",
        fontsize=10.5, fontweight="bold", pad=8,
        color=TIER_C[tier],
    )

    # legend inside panel — certified first, then uncertifiable, sorted by AUROC desc
    order = sorted(range(len(classes)),
                   key=lambda j: (not roc_data[classes[j]]["cert"],
                                  -roc_data[classes[j]]["auc"]))
    ax.legend(
        [h_list[j] for j in order],
        [l_list[j] for j in order],
        fontsize=7.5 if len(classes) > 8 else 8.2,
        framealpha=0.92, edgecolor="#ddd",
        loc="lower right",
        handlelength=1.6,
        labelspacing=0.35,
    )

# ── shared legend for line style ───────────────────────────────────────────────
solid  = mlines.Line2D([], [], color="#555", lw=1.8, ls="-",
                        label="Certified  (tiered/primary α)")
dashed = mlines.Line2D([], [], color="#555", lw=1.4, ls=(0, (4, 2)), alpha=0.65,
                        label="Uncertifiable / trivial")
fig.legend(handles=[solid, dashed],
           loc="upper center", ncol=2,
           fontsize=9, framealpha=0.93, edgecolor="#ddd",
           bbox_to_anchor=(0.5, 1.03),
           handlelength=2.0, columnspacing=3.0)

# ── save ───────────────────────────────────────────────────────────────────────
out_dir = pathlib.Path("results/figures")
out_dir.mkdir(parents=True, exist_ok=True)
for ext in ("pdf", "png"):
    p = out_dir / f"class_auroc.{ext}"
    fig.savefig(p, bbox_inches="tight", dpi=150)
    print(f"Saved: {p}")
