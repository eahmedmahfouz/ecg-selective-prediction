"""Risk-coverage operating points — single panel, tier-colored.

All 34 certified (flag-rate, FNR) points across 3 α sweep configurations,
colored by tier. Marker shape encodes the sweep config (▼ stricter, ● tiered,
▲ looser). Background bands mark FNR zones by tier budget.

Reads:
  results/contribution1/sensitivity_sweep.csv

Writes:
  results/figures/risk_coverage_curves.pdf
  results/figures/risk_coverage_curves.png

Run:
  python src/plot_risk_coverage_curves.py
"""

import csv
import pathlib
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from adjustText import adjust_text

# ── style ──────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "figure.facecolor":   "white",
    "axes.facecolor":     "white",
    "axes.spines.top":    False,
    "axes.spines.right":  False,
    "axes.spines.left":   True,
    "axes.spines.bottom": True,
    "axes.grid":          True,
    "grid.color":         "white",
    "grid.linewidth":     1.1,
    "grid.alpha":         1.0,
    "axes.axisbelow":     False,
    "font.family":        "sans-serif",
    "font.size":          11,
})

# ── tier colours — explicit hex, no seaborn ────────────────────────────────────
TIER_C = {
    "Critical":  "#C44E52",   # muted red
    "Important": "#DD8452",   # muted orange
    "Benign":    "#55A868",   # muted green
}

# background band tints — hand-picked light versions of the tier colours
TIER_BAND = {
    "Critical":  "#F7D5D6",   # light rose
    "Important": "#FAEBD7",   # light peach
    "Benign":    "#D5EFD9",   # light sage
}
ABOVE_BAND = "#EBEBEB"        # neutral grey above all budgets

XLIM         = (-0.02, 1.03)
YLIM         = (-0.005, 0.122)
TIER_ALPHA   = {"Critical": 0.02, "Important": 0.05, "Benign": 0.10}
CONFIG_ORDER = {"stricter": 0, "tiered": 1, "looser": 2}
MARKERS      = {"stricter": "v", "tiered": "o", "looser": "^"}
MS           = {"stricter": 6.5, "tiered": 9.5, "looser": 6.5}

# ── load ───────────────────────────────────────────────────────────────────────
C1   = pathlib.Path("results/contribution1")
rows = list(csv.DictReader(open(C1 / "sensitivity_sweep.csv")))

by_class: dict = defaultdict(list)
for r in rows:
    if r["status"] == "certified":
        by_class[r["class"]].append(r)
for cls in by_class:
    by_class[cls].sort(key=lambda r: CONFIG_ORDER[r["config"]])

cls_tier = {cls: pts[0]["tier"] for cls, pts in by_class.items()}

# ── verification ───────────────────────────────────────────────────────────────
_plotted: list = []

def record(cls, cfg, x, y):
    _plotted.append((cls, cfg, x, y))

def verify():
    csv_ok = {(r["class"], r["config"]): r
              for r in rows if r["status"] == "certified"}
    errs = []
    for cls, cfg, x, y in _plotted:
        r = csv_ok.get((cls, cfg))
        if r is None:
            errs.append(f"NOT IN CSV: {cls} {cfg}"); continue
        cx, cy = float(r["flag_pct"]) / 100, float(r["test_fnr"])
        if abs(x - cx) > 1e-9: errs.append(f"x-mismatch {cls} {cfg}")
        if abs(y - cy) > 1e-9: errs.append(f"y-mismatch {cls} {cfg}")
    for cls, cfg in csv_ok:
        if not any(pc == cls and pr == cfg for pc, pr, _, _ in _plotted):
            errs.append(f"MISSING: {cls} {cfg}")
    if errs:
        for e in errs: print(f"  ERR: {e}")
        raise AssertionError(f"{len(errs)} errors")
    print(f"  Verification PASSED: {len(_plotted)} points.")

# ── figure ─────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(12, 6.2))
fig.subplots_adjust(left=0.09, right=0.86, top=0.91, bottom=0.13)

# ── background bands ───────────────────────────────────────────────────────────
bands = [
    (TIER_BAND["Critical"],  YLIM[0], 0.02),
    (TIER_BAND["Important"], 0.02,    0.05),
    (TIER_BAND["Benign"],    0.05,    0.10),
    (ABOVE_BAND,             0.10,    YLIM[1]),
]
for color, ylo, yhi in bands:
    ax.axhspan(ylo, yhi, color=color, lw=0, zorder=0)

# ── α ceiling lines + styled right-margin labels ───────────────────────────────
line_ls = {
    "Critical":  (0, (6, 3)),
    "Important": (0, (3, 2)),
    "Benign":    (0, (1, 2)),
}
for tier in ("Critical", "Important", "Benign"):
    budget = TIER_ALPHA[tier]
    color  = TIER_C[tier]
    band   = TIER_BAND[tier]
    ax.axhline(budget, color=color, lw=1.5, ls=line_ls[tier],
               alpha=0.85, zorder=3)
    # label in a tinted box matching the band below the line
    ax.text(1.030, budget, f"α = {budget}",
            transform=ax.get_yaxis_transform(),
            ha="left", va="center", fontsize=8,
            color=color, fontweight="bold", clip_on=False,
            bbox=dict(boxstyle="round,pad=0.25",
                      fc=band, ec=color, lw=0.8, alpha=0.95))

# ── scatter all certified points ───────────────────────────────────────────────
texts     = []
all_pts_x = [float(r["flag_pct"]) / 100 for r in rows if r["status"] == "certified"]
all_pts_y = [float(r["test_fnr"])        for r in rows if r["status"] == "certified"]

sort_key  = lambda cls: ({"Critical": 0, "Important": 1, "Benign": 2}[cls_tier[cls]], cls)

for cls in sorted(by_class, key=sort_key):
    tier   = cls_tier[cls]
    color  = TIER_C[tier]
    pts    = by_class[cls]
    single = len(pts) == 1

    for p in pts:
        x = float(p["flag_pct"]) / 100
        y = float(p["test_fnr"])
        ax.plot(x, y,
                marker=MARKERS[p["config"]],
                ms=MS[p["config"]],
                color=color,
                mec="white", mew=1.0,
                alpha=0.40 if single else 1.0,
                linestyle="None", zorder=5)
        record(cls, p["config"], x, y)

    # label at tiered config (or sole point for single-config†)
    ref    = next((p for p in pts if p["config"] == "tiered"), pts[0])
    lx, ly = float(ref["flag_pct"]) / 100, float(ref["test_fnr"])
    lbl    = f"{cls}†" if single else cls
    t = ax.text(lx, ly, f"  {lbl}",
                fontsize=8.5, color=color,
                fontweight="bold",
                fontstyle="italic" if single else "normal",
                va="center", ha="left", zorder=7,
                bbox=dict(boxstyle="round,pad=0.12",
                          fc="white", ec="none", alpha=0.80))
    if single:
        t.set_alpha(0.65)
    texts.append(t)

# ── label repulsion ────────────────────────────────────────────────────────────
adjust_text(texts, x=all_pts_x, y=all_pts_y, ax=ax,
            expand_points=(2.2, 2.2), expand_text=(1.6, 1.6),
            force_text=(0.55, 0.75), force_points=(0.35, 0.45))

# ── axes ───────────────────────────────────────────────────────────────────────
ax.set_xlim(XLIM)
ax.set_ylim(YLIM)
ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v*100:.0f}%"))
ax.xaxis.set_major_locator(plt.MultipleLocator(0.20))
ax.set_xlabel("Flag rate  (fraction of test set flagged positive)",
              fontsize=10.5, labelpad=7)
ax.set_ylabel("Test FNR", fontsize=10.5, labelpad=7)

# ── legend ─────────────────────────────────────────────────────────────────────
tier_h = [
    mlines.Line2D([], [], marker="o", color=TIER_C[t], ms=9,
                  linestyle="None", mec="white", mew=1.0,
                  label=f"{t}  (α ≤ {TIER_ALPHA[t]})")
    for t in ("Critical", "Important", "Benign")
]
config_h = [
    mlines.Line2D([], [], marker=m, color="#555", ms=s,
                  linestyle="None", mec="white", mew=1.0,
                  label=lbl)
    for m, s, lbl in (
        ("v", 6.5, "Stricter α"),
        ("o", 9.5, "Tiered / primary"),
        ("^", 6.5, "Looser α"),
    )
] + [
    mlines.Line2D([], [], marker="o", color="#bbb", ms=6.5, alpha=0.40,
                  linestyle="None", mec="white", mew=1.0,
                  label="Single-config only (†)"),
]

ax.legend(handles=tier_h + config_h,
          loc="upper right", bbox_to_anchor=(0.985, 0.985),
          fontsize=8.5, framealpha=0.95, edgecolor="#ccc",
          handlelength=1.5, labelspacing=0.45, borderpad=0.8,
          title="Tier / config", title_fontsize=9.0,
          fancybox=False)

fig.suptitle(
    "Risk–coverage operating points  ·  Binomial LTT, δ = 0.10/24  ·  "
    "34 certified points, 16 classes, 3 α configurations",
    fontsize=10, y=0.975)

# ── verify & save ──────────────────────────────────────────────────────────────
print("Running verification …")
verify()

out_dir = pathlib.Path("results/figures")
out_dir.mkdir(parents=True, exist_ok=True)
for ext in ("pdf", "png"):
    p = out_dir / f"risk_coverage_curves.{ext}"
    fig.savefig(p, bbox_inches="tight", dpi=150)
    print(f"Saved: {p}")
