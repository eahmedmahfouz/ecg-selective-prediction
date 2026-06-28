"""Generate LaTeX tables for Contribution 1 results."""
import csv
from pathlib import Path

RESULTS = Path("results")
C1      = RESULTS / "contribution1"
TABLES  = RESULTS / "tables"
TABLES.mkdir(exist_ok=True)

tier_order = {"Critical": 0, "Important": 1, "Benign": 2}


def tex_cls(s):
    return s.replace("_", r"\_")


def fmt_lam(v):
    return "---" if float(v) == 0.0 else "%.4f" % float(v)


def fmt_fnr(v, status):
    return "---" if status in ("uncertifiable", "trivial") else "%.4f" % float(v)


def fmt_flag(v, status):
    return "---" if status in ("uncertifiable", "trivial") else "%.1f\\%%" % float(v)


STATUS_LABEL = {
    "certified":     "certified",
    "trivial":       r"trivial$^{\dag}$",
    "uncertifiable": r"uncert.$^{\ddag}$",
}


# ═══════════════════════════════════════════════════════════════════════════
# TABLE 1 — Tier definitions
# ═══════════════════════════════════════════════════════════════════════════

# Supported / unsupported counts come from the CSVs on disk
sup_rows = list(csv.DictReader(open(C1 / "supported_classes.csv")))
uns_rows = list(csv.DictReader(open(C1 / "unsupported_classes.csv")))

sup_by_tier  = {}
uns_by_tier  = {}
for r in sup_rows:
    sup_by_tier.setdefault(r["tier"], []).append((r["class"], r["n_val_pos"]))
for r in uns_rows:
    uns_by_tier.setdefault(r["tier"], []).append((r["class"], r["n_val_pos"]))

RATIONALE = {
    "Critical":  (
        "MI subtypes and AV block cannot be stratified by acuity at this label "
        "granularity---treated conservatively by necessity. PTB-XL has no VT/VF; "
        r"\texttt{\_AVB} is one merged 1st--3rd-degree bucket."
    ),
    "Important": (
        "Ischaemia patterns; CLBBB can mask ischaemia; WPW carries arrhythmia risk; "
        "supraventricular tachyarrhythmias and ectopy markers."
    ),
    "Benign": (
        r"Chronic/structural or physiological-variant findings; PACE is a "
        "device-status marker already under clinical management."
    ),
}
ALPHA = {"Critical": "0.02", "Important": "0.05", "Benign": "0.10"}


def cls_list(pairs):
    return ", ".join(r"\texttt{%s}\,(%s)" % (tex_cls(c), n) for c, n in pairs)


lines = [
    r"\begin{table}[t]",
    r"\centering",
    r"\caption{Acuity tier definitions. Supported: $n_{\mathrm{val}}^{+}\!\ge\!10$; "
    r"unsupported classes are excluded from calibration. "
    r"PTB-XL contains no VT or VF; \texttt{\_AVB} is a merged "
    r"first--third-degree AV-block bucket. `Critical' denotes the most severe "
    r"conditions \emph{available in this dataset}, not ICU-grade emergencies.}",
    r"\label{tab:tiers}",
    r"\small",
    r"\begin{tabularx}{\linewidth}{@{}l c p{3.0cm} p{2.6cm} X@{}}",
    r"\toprule",
    (r"\textbf{Tier} & $\boldsymbol{\alpha}$ & "
     r"\textbf{Supported} ($n_{\mathrm{val}}^{+}$) & "
     r"\textbf{Unsupported} ($n_{\mathrm{val}}^{+}$) & "
     r"\textbf{Rationale} \\"),
    r"\midrule",
]
for tier in ("Critical", "Important", "Benign"):
    sup_str = cls_list(sup_by_tier.get(tier, []))
    uns_str = cls_list(uns_by_tier.get(tier, []))
    lines.append(
        r"%s & %s & %s & %s & %s \\[4pt]"
        % (tier, ALPHA[tier], sup_str, uns_str, RATIONALE[tier])
    )
lines += [r"\bottomrule", r"\end{tabularx}", r"\end{table}"]
(TABLES / "table1_tier_definitions.tex").write_text("\n".join(lines))
print("Saved: table1_tier_definitions.tex")


# ═══════════════════════════════════════════════════════════════════════════
# TABLE 2 — Primary calibration results
# ═══════════════════════════════════════════════════════════════════════════

prim = list(csv.DictReader(open(C1 / "primary_calibration.csv")))
prim.sort(key=lambda r: (tier_order[r["tier"]], r["class"]))

lines = [
    r"\begin{table*}[t]",
    r"\centering",
    r"\caption{Primary calibration results under tiered $\alpha$ (binomial LTT, "
    r"$\delta=0.10/24\approx0.0042$, \texttt{bonferroni\_grid=False}). "
    r"Flag\% = fraction of test set with score $\ge\lambda_c$ (predicted positive), "
    r"computed identically for every row. "
    r"$^{\dag}$Trivial: certifies only $\lambda_c=0$ (no useful abstention). "
    r"$^{\ddag}$Uncertifiable: $n_{\mathrm{val}}^{+}$ insufficient to reject $H_0$ "
    r"at the required $\alpha$ and $\delta$.}",
    r"\label{tab:primary-calibration}",
    r"\small",
    r"\begin{tabular}{@{}llcrcccl@{}}",
    r"\toprule",
    (r"\textbf{Class} & \textbf{Tier} & $\boldsymbol{\alpha}$ & "
     r"$n_{\mathrm{val}}^{+}$ & $\lambda_c$ & \textbf{Test FNR} & "
     r"\textbf{Flag\%} & \textbf{Status} \\"),
    r"\midrule",
]
prev_tier = None
for r in prim:
    if prev_tier and r["tier"] != prev_tier:
        lines.append(r"\addlinespace[3pt]")
    prev_tier = r["tier"]
    st = r["status"]
    lines.append(
        r"\texttt{%s} & %s & %.2f & %s & %s & %s & %s & %s \\"
        % (
            tex_cls(r["class"]), r["tier"], float(r["alpha"]),
            r["n_val_pos"],
            fmt_lam(r["lambda_c"]),
            fmt_fnr(r["test_fnr"], st),
            fmt_flag(r["flag_pct"], st),
            STATUS_LABEL[st],
        )
    )
lines += [r"\bottomrule", r"\end{tabular}", r"\end{table*}"]
(TABLES / "table2_primary_calibration.tex").write_text("\n".join(lines))
print("Saved: table2_primary_calibration.tex")


# ═══════════════════════════════════════════════════════════════════════════
# TABLE 3 — Failure-mode breakdown (two panels)
# ═══════════════════════════════════════════════════════════════════════════

unif  = {r["class"]: r for r in csv.DictReader(open(C1 / "uniform_comparison.csv"))}
score = {r["class"]: r for r in csv.DictReader(open(C1 / "scorecard.csv"))}

safety = sorted(
    [(c, s) for c, s in score.items() if s["safety_failure"] == "True"],
    key=lambda x: (tier_order.get(x[1]["tier"], 9), x[0]),
)
eff = sorted(
    [(c, s) for c, s in score.items() if s["efficiency_loss"] == "True"],
    key=lambda x: x[0],
)

lines = [
    r"\begin{table*}[t]",
    r"\centering",
    r"\caption{Uniform-$\alpha$ failure modes illustrating the necessity of "
    r"acuity-weighting. \textbf{Panel~A} (safety failures, 6~classes): a uniform "
    r"run certifies a class at $\alpha>\alpha_{\mathrm{tier}}$, publishing a "
    r"guarantee weaker than the clinical tier requires. "
    r"\textbf{Panel~B} (efficiency losses, 9~classes): classes certified under "
    r"tiered $\alpha{=}0.10$ become uncertifiable under uniform $\alpha{=}0.02$, "
    r"discarding a working guarantee through over-conservatism. "
    r"The two effects are disjoint and hit different tiers; no single uniform "
    r"$\alpha$ avoids both simultaneously.}",
    r"\label{tab:failure-modes}",
    r"\small",
    r"",
    r"% Panel A",
    r"\begin{tabular}{@{}llccccc@{}}",
    r"\toprule",
    r"\multicolumn{7}{@{}l}{\textbf{Panel~A\;---\;Safety failures} "
    r"(certified at $\alpha>\alpha_{\mathrm{tier}}$ under a uniform run)} \\",
    r"\addlinespace[2pt]",
    (r"\textbf{Class} & \textbf{Tier} & $\alpha_{\mathrm{tier}}$ & "
     r"\multicolumn{2}{c}{\textbf{Tiered}} & "
     r"\multicolumn{2}{c}{\textbf{Worst uniform}} \\"),
    r"\cmidrule(lr){4-5}\cmidrule(lr){6-7}",
    r" & & & $\lambda_c$ & Status & $\lambda_c$\;($\alpha$) & Status \\",
    r"\midrule",
]
for cls, sc in safety:
    u = unif[cls]
    t_alpha = float(sc["tiered_alpha"])
    u10_st  = u["u0.10_status"]
    u05_st  = u["u0.05_status"]
    if u10_st == "certified":
        wlam, wstat, walpha = u["u0.10_lambda_c"], u10_st, "0.10"
    elif u05_st == "certified":
        wlam, wstat, walpha = u["u0.05_lambda_c"], u05_st, "0.05"
    else:
        wlam, wstat, walpha = u["u0.05_lambda_c"], u05_st, "0.05"
    lines.append(
        r"\texttt{%s} & %s & %.2f & %s & %s & %s\;(%.2f) & %s \\"
        % (
            tex_cls(cls), sc["tier"], t_alpha,
            fmt_lam(u["tiered_lambda_c"]),
            STATUS_LABEL.get(u["tiered_status"], u["tiered_status"]),
            ("%.4f" % float(wlam)) if float(wlam) > 0 else "---",
            float(walpha),
            STATUS_LABEL.get(wstat, wstat),
        )
    )
lines += [r"\bottomrule", r"\end{tabular}", r"", r"\bigskip", r""]

# Panel B
lines += [
    r"% Panel B",
    r"\begin{tabular}{@{}llrccl@{}}",
    r"\toprule",
    r"\multicolumn{6}{@{}l}{\textbf{Panel~B\;---\;Efficiency losses} "
    r"(tiered $\alpha{=}0.10$ certifies; uniform $\alpha{=}0.02$ does not)} \\",
    r"\addlinespace[2pt]",
    (r"\textbf{Class} & \textbf{Tier} & $n_{\mathrm{val}}^{+}$ & "
     r"\multicolumn{2}{c}{\textbf{Tiered} ($\alpha{=}0.10$)} & "
     r"\textbf{Uniform} ($\alpha{=}0.02$) \\"),
    r"\cmidrule(lr){4-5}",
    r" & & & $\lambda_c$ & Test FNR & Status \\",
    r"\midrule",
]
for cls, sc in eff:
    u = unif[cls]
    lines.append(
        r"\texttt{%s} & %s & %s & %s & %s & %s \\"
        % (
            tex_cls(cls), sc["tier"], sc["n_val_pos"],
            fmt_lam(u["tiered_lambda_c"]),
            "%.4f" % float(u["tiered_test_fnr"]),
            STATUS_LABEL.get(u["u0.02_status"], u["u0.02_status"]),
        )
    )
lines += [r"\bottomrule", r"\end{tabular}", r"\end{table*}"]
(TABLES / "table3_failure_modes.tex").write_text("\n".join(lines))
print("Saved: table3_failure_modes.tex")


# ═══════════════════════════════════════════════════════════════════════════
# TABLE 4 — Coverage simulation
# ═══════════════════════════════════════════════════════════════════════════

sim = list(csv.DictReader(open(RESULTS / "coverage_simulation.csv")))

METHOD_LABEL = {
    ("hoeffding", "True"):  "Hoeffding + grid-Bonferroni",
    ("hoeffding", "False"): "Hoeffding + monotone boundary",
    ("binomial",  "False"): r"\textbf{Binomial + monotone boundary} (adopted)",
}

lines = [
    r"\begin{table}[t]",
    r"\centering",
    r"\caption{Coverage-simulation validation ($n_{\mathrm{trials}}{=}1000$, "
    r"$n_{\mathrm{cal}}{=}2000$, $\alpha{=}0.10$, $\delta{=}0.10$). "
    r"Positive scores $\sim\mathrm{Beta}(7,3)$, negative $\sim\mathrm{Beta}(3,7)$; "
    r"true FNR evaluated analytically via the Beta CDF. "
    r"Requirement: violation rate $\le\delta=0.10$.}",
    r"\label{tab:coverage-sim}",
    r"\small",
    r"\begin{tabular}{@{}lccc c@{}}",
    r"\toprule",
    (r"\textbf{Method} & \textbf{Viol.\,rate} & "
     r"$\overline{\lambda}_c$ & $\overline{\mathrm{FNR}}$ & \\"),
    r"\midrule",
]
for r in sim:
    key   = (r["method"], r["bonferroni_grid"])
    label = METHOD_LABEL.get(key, r["method"])
    vr    = float(r["violation_rate"])
    badge = r"\textbf{PASS}" if vr <= 0.10 else r"\textbf{FAIL}"
    lines.append(
        r"%s & %.3f & %.4f & %.4f & %s \\"
        % (label, vr, float(r["mean_lambda_c"]), float(r["mean_true_fnr"]), badge)
    )
lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
(TABLES / "table4_coverage_simulation.tex").write_text("\n".join(lines))
print("Saved: table4_coverage_simulation.tex")

print()
for p in sorted(TABLES.iterdir()):
    print("  %s  (%d bytes)" % (p.name, p.stat().st_size))
