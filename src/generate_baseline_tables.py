"""Generate paper-ready tables for the baseline comparison.

Reads from already-verified files on disk — no recomputation.

Outputs:
  results/tables/table_baseline_headline.csv
  results/tables/table_baseline_headline.md
  results/tables/table_baseline_mechanism.csv
  results/tables/table_baseline_mechanism.md

Run: python src/generate_baseline_tables.py
"""

import csv
import pathlib
from collections import defaultdict

RESULTS = pathlib.Path("results")
C1      = RESULTS / "contribution1"
TABLES  = RESULTS / "tables"
TABLES.mkdir(exist_ok=True)

# ── load sources ───────────────────────────────────────────────────────────────
comp_rows = list(csv.DictReader(open(RESULTS / "baseline_comparison.csv")))
ltt_src   = {r["class"]: r for r in csv.DictReader(open(C1 / "primary_calibration.csv"))}
ts_src    = {r["class"]: r for r in csv.DictReader(open(RESULTS / "baseline_temp_scaling.csv"))}
mc_src    = {r["class"]: r for r in csv.DictReader(open(RESULTS / "baseline_mc_dropout.csv"))}

N_SUPPORTED = 24
MECHANISM_CLASSES = ["AMI", "IMI", "IVCD", "LVH", "NST_", "SARRH", "SBRAD", "STTC"]

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TABLE 1 — headline violation counts
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

method_display = {
    "ltt_binomial":       "LTT binomial",
    "temp_scaling_naive": "Temperature scaling (naive)",
    "mc_dropout_naive":   "MC Dropout (naive)",
}
guarantee_type = {
    "ltt_binomial":       "Statistical (binomial LTT)",
    "temp_scaling_naive": "Empirical (no correction)",
    "mc_dropout_naive":   "Empirical (no correction)",
}
expected_violations = {
    "ltt_binomial":       0,
    "temp_scaling_naive": 14,
    "mc_dropout_naive":   14,
}

violations_by_method = defaultdict(int)
for r in comp_rows:
    if r["violates_alpha"] == "True":
        violations_by_method[r["method"]] += 1

print("TABLE 1: Headline violation counts")
print("-" * 50)
errors = []
for method_key in ["ltt_binomial", "temp_scaling_naive", "mc_dropout_naive"]:
    computed  = violations_by_method[method_key]
    expected  = expected_violations[method_key]
    match     = computed == expected
    status    = "OK" if match else f"MISMATCH (expected {expected})"
    print(f"  {method_display[method_key]:30s}  {computed}/{N_SUPPORTED}  [{status}]")
    if not match:
        errors.append(f"{method_key}: computed={computed}, expected={expected}")

if errors:
    print("\nSTOPPING: violation counts do not match expected values.")
    for e in errors:
        print(f"  {e}")
    raise SystemExit(1)

print("  Counts verified. Building table.\n")

headline_rows = []
for method_key in ["ltt_binomial", "temp_scaling_naive", "mc_dropout_naive"]:
    v    = violations_by_method[method_key]
    rate = f"{v / N_SUPPORTED * 100:.0f}%"
    headline_rows.append({
        "method":           method_display[method_key],
        "guarantee_type":   guarantee_type[method_key],
        "violations":       f"{v}/{N_SUPPORTED}",
        "rate":             rate,
    })

# CSV
fields = ["method", "guarantee_type", "violations", "rate"]
p = TABLES / "table_baseline_headline.csv"
with open(p, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=fields)
    w.writeheader()
    w.writerows(headline_rows)
print(f"Saved: {p}")

# Markdown
md_lines = [
    "| Method | Guarantee type | Violations | Rate |",
    "|---|---|---|---|",
]
for r in headline_rows:
    md_lines.append(f"| {r['method']} | {r['guarantee_type']} | {r['violations']} | {r['rate']} |")
p = TABLES / "table_baseline_headline.md"
p.write_text("\n".join(md_lines) + "\n")
print(f"Saved: {p}")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TABLE 2 — mechanism table (8 certified-but-violates classes)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# Pivot comparison CSV for easy lookup
comp = defaultdict(dict)
for r in comp_rows:
    comp[r["class"]][r["method"]] = r

print("\nTABLE 2: Mechanism table (certified-but-violates classes)")
print("-" * 70)

# Verify all 8 classes are LTT-certified and violate under both naive methods
for cls in MECHANISM_CLASSES:
    ltt_status = ltt_src[cls]["status"]
    ts_viol    = comp[cls]["temp_scaling_naive"]["violates_alpha"]
    mc_viol    = comp[cls]["mc_dropout_naive"]["violates_alpha"]
    ok = ltt_status == "certified" and ts_viol == "True" and mc_viol == "True"
    print(f"  {cls:12s}  ltt_status={ltt_status:12s}  ts_violates={ts_viol}  mc_violates={mc_viol}"
          f"  [{'OK' if ok else 'MISMATCH'}]")
    if not ok:
        errors.append(f"{cls}: not in expected group")

if errors:
    print("\nSTOPPING: class membership does not match expected.")
    for e in errors:
        print(f"  {e}")
    raise SystemExit(1)

mechanism_rows = []
for cls in MECHANISM_CLASSES:
    ltt = ltt_src[cls]
    ts  = ts_src[cls]
    mc  = mc_src[cls]

    ltt_lam   = float(ltt["lambda_c"])
    ts_lam    = float(ts["lambda_c"])
    mc_lam    = float(mc["lambda_c"])

    # Mechanical verification: naive λ > LTT λ_c (already confirmed in sanity pass)
    assert ts_lam > ltt_lam, f"ts_lambda NOT > ltt_lambda_c for {cls}"
    assert mc_lam > ltt_lam, f"mc_lambda NOT > ltt_lambda_c for {cls}"

    mechanism_rows.append({
        "class":        cls,
        "tier":         ltt["tier"],
        "alpha":        ltt["alpha"],
        "ltt_lambda_c": ltt["lambda_c"],
        "ltt_test_fnr": ltt["test_fnr"],
        "ts_lambda":    ts["lambda_c"],
        "ts_test_fnr":  ts["test_fnr"],
        "mc_lambda":    mc["lambda_c"],
        "mc_test_fnr":  mc["test_fnr"],
    })

print("  Direction check (naive λ > LTT λ): PASSED for all 8 rows.\n")

# CSV
fields = ["class", "tier", "alpha",
          "ltt_lambda_c", "ltt_test_fnr",
          "ts_lambda", "ts_test_fnr",
          "mc_lambda", "mc_test_fnr"]
p = TABLES / "table_baseline_mechanism.csv"
with open(p, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=fields)
    w.writeheader()
    w.writerows(mechanism_rows)
print(f"Saved: {p}")

# Markdown — 4-decimal display for λ and FNR
def fmt4(v):
    return f"{float(v):.4f}"

md_lines = [
    "| Class | Tier | α | LTT λ | LTT FNR | TS λ | TS FNR | MC λ | MC FNR |",
    "|---|---|---|---|---|---|---|---|---|",
]
for r in mechanism_rows:
    md_lines.append(
        f"| {r['class']} | {r['tier']} | {r['alpha']} "
        f"| {fmt4(r['ltt_lambda_c'])} | {fmt4(r['ltt_test_fnr'])} "
        f"| {fmt4(r['ts_lambda'])} | {fmt4(r['ts_test_fnr'])} "
        f"| {fmt4(r['mc_lambda'])} | {fmt4(r['mc_test_fnr'])} |"
    )
p = TABLES / "table_baseline_mechanism.md"
p.write_text("\n".join(md_lines) + "\n")
print(f"Saved: {p}")
