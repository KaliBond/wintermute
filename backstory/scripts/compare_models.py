"""Compare two or more raw CAMS score files cell by cell.

Usage: python compare_models.py <reference_raw.csv> <other_raw.csv> [more ...] --out <report.md>
Cells are keyed on (Society, Node). Year is checked but not used as a key, so a model
that wrote a different anchor year is still compared (and flagged).
"""
import sys, argparse, os
import pandas as pd

def spearman(a, b):
    """Spearman without scipy: Pearson on average ranks."""
    return a.rank().corr(b.rank())

DIMS = ["Coherence","Capacity","Stress","Abstraction"]
ap = argparse.ArgumentParser(); ap.add_argument("files", nargs="+"); ap.add_argument("--out", required=True)
a = ap.parse_args()
load = lambda f: pd.read_csv(f, na_values=["NA"], keep_default_na=False, skipinitialspace=True)
ref = load(a.files[0]); lines = [f"# Model comparison\n\nReference: `{os.path.basename(a.files[0])}`\n"]
for f in a.files[1:]:
    o = load(f)
    m = ref.merge(o, on=["Society","Node"], how="outer", suffixes=("_ref","_oth"), indicator=True)
    lines.append(f"## vs `{os.path.basename(f)}`\n")
    lines.append(f"- Cells matched: {(m._merge=='both').sum()} of {len(ref)}; only in reference {(m._merge=='left_only').sum()}; only in other {(m._merge=='right_only').sum()}")
    yr = m[(m._merge=='both') & (m.Year_ref != m.Year_oth)]
    if len(yr): lines.append(f"- Anchor-year mismatches: {sorted(set(yr.Society))}")
    m = m[m._merge == "both"]
    lines.append("\n| Dimension | n both scored | MAE | exact | within 1 | Spearman | mean diff (other - ref) | NA agree | NA only ref | NA only other |")
    lines.append("| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |")
    for dmn in DIMS:
        r, x = m[dmn+"_ref"], m[dmn+"_oth"]; both = r.notna() & x.notna()
        diff = (x - r)[both]
        sp = spearman(r[both], x[both]) if both.sum() > 2 else float("nan")
        lines.append(f"| {dmn} | {both.sum()} | {diff.abs().mean():.2f} | {(diff==0).mean():.0%} | {(diff.abs()<=1).mean():.0%} | {sp:.2f} | {diff.mean():+.2f} | {(r.isna()&x.isna()).sum()} | {(r.isna()&x.notna()).sum()} | {(r.notna()&x.isna()).sum()} |")
    for side in ["ref","oth"]:
        m["NV_"+side] = (m["Coherence_"+side] + m["Capacity_"+side] - m["Stress_"+side] + 0.5*m["Abstraction_"+side]).round(1)
    cs = m.groupby("Society")[["NV_ref","NV_oth"]].mean()
    lines.append(f"\n- Case-level ranking agreement (Spearman on mean Node Value, {cs.dropna().shape[0]} cases): {spearman(cs.dropna().NV_ref, cs.dropna().NV_oth):.2f}")
    m["absNV"] = (m.NV_oth - m.NV_ref).abs()
    top = m.sort_values("absNV", ascending=False).head(10)
    lines.append("\n### Ten largest Node Value disagreements\n\n| Society | Node | NV ref | NV other | C K S A ref | C K S A other |\n| --- | --- | --- | --- | --- | --- |")
    for _, t in top.iterrows():
        cr = " ".join(str(t[d+"_ref"]) for d in DIMS); co = " ".join(str(t[d+"_oth"]) for d in DIMS)
        lines.append(f"| {t.Society} | {t.Node} | {t.NV_ref} | {t.NV_oth} | {cr} | {co} |")
    lines.append("")
open(a.out, "w").write("\n".join(lines) + "\n"); print("\n".join(lines))
