#!/usr/bin/env python3
"""
sybond_report_exporter.py

Exports a 10-section Sybond Report Template in Markdown.

Use it in two modes:

1. Blank template:
   python sybond_report_exporter.py --blank --society France --sybond-name Marianne --output France_template.md

2. Data-filled template from a CAMS ensemble mean CSV:
   python sybond_report_exporter.py \
       --mean-csv France_CAMS_ensemble_mean.csv \
       --envelope-csv France_CAMS_envelope.csv \
       --society France \
       --sybond-name Marianne \
       --output France_Sybond_Report.md

Expected mean CSV columns, with flexible aliases:
Society, Year, Node, Coherence, Capacity, Stress, Abstraction, Node Value, Bond Strength

The script will compute the current CAMS v3.2-R working operators:
- Canonical Node Value: V_i = C_i + K_i - S_i + 0.5 A_i
- ESCH Activation: sigma_i = (A_i * C_i) * (K_i - S_i)
- Mean system viability: V_bar
- Node-value dispersion: sigma_V
- Fast/slow loop diagnostics
- Library Attractor proxy: eta_loop = (B_Lore * B_Archive) / S_Hands where available
"""

from __future__ import annotations

import argparse
import math
import re
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

try:
    import pandas as pd
except ImportError as exc:
    raise SystemExit(
        "This script requires pandas. Install with: pip install pandas"
    ) from exc


# ---------------------------------------------------------------------------
# CAMS v3.2-R working kernel
# ---------------------------------------------------------------------------

NODES: List[str] = [
    "Helm", "Shield", "Lore", "Stewards", "Craft", "Hands", "Archive", "Flow"
]

FAST_LOOP = ["Helm", "Shield", "Flow", "Hands", "Craft"]
SLOW_LOOP = ["Archive", "Lore", "Stewards"]

NODE_FUNCTIONS: Dict[str, str] = {
    "Helm": "executive arbitration, steering, regime command, decision tempo",
    "Shield": "boundary defence, security, coercive capacity, threat processing",
    "Lore": "meaning, legitimacy, symbolic integration, worldview and social myth",
    "Stewards": "resource ownership, allocation, property, capital and elite custody",
    "Craft": "technical competence, professions, engineering, skill and execution",
    "Hands": "population, labour, embodied production, demographic and civic base",
    "Archive": "institutional memory, records, bureaucracy, law, continuity and learning",
    "Flow": "commerce, circulation, exchange, infrastructure and distribution",
}

FAILURE_MODES: Dict[str, str] = {
    "Helm": "Executive Decoupling: paralysis, capture, legitimacy rupture or over-centralisation",
    "Shield": "Praetorian Condition: shield inversion, hyperactivation, atrophy or threat capture",
    "Lore": "Normative collapse: symbolic exhaustion, myth capture, legitimacy breakdown",
    "Stewards": "Rent-seeking decay: extraction, misallocation, oligarchic insulation",
    "Craft": "Skill erosion: loss of technical competence, professional degradation, innovation failure",
    "Hands": "Motivational or demographic failure: exhaustion, immiseration, withdrawal or revolt",
    "Archive": "Amnesia or rigidity: institutional forgetting, brittle procedure, loss of learning",
    "Flow": "Mythic-material decoupling: circulation failure, price/reality break, logistical stress",
}

CAMS_FORMULATION = """
## CAMS v3.2-R Working Formulation Used by This Exporter

This exporter uses the April 2026 CAMS v3.2-R working kernel.

**Ontology**
- Eight-node partition: Helm, Shield, Lore, Stewards, Craft, Hands, Archive, Flow.
- Fast loop: Helm, Shield, Flow, Hands, Craft.
- Slow loop: Archive, Lore, Stewards.
- State vector per node: Ψ_i(t) = (C_i, K_i, A_i, S_i, B_i), where:
  - C = Coherence
  - K = Capacity
  - A = Abstraction
  - S = Stress
  - B = Bond Strength / coupling proxy, if supplied

**Core scalar operators**
- Canonical Node Value: `V_i = C_i + K_i - S_i + 0.5 A_i`
- ESCH Activation Index: `sigma_i = (A_i * C_i) * (K_i - S_i)`
- System viability: `V_bar = mean(V_i)`
- Node dispersion: `sigma_V = population standard deviation of V_i`
- Library Attractor proxy, where bond data exists: `eta_loop = (B_Lore * B_Archive) / S_Hands`

**Operational phase-space thresholds**
- Crisis floor: `V_bar < 12`
- High node dispersion / shear: `sigma_V > 3.5`
- Disregard / buffering trigger: `V_bar > 15 and sigma_V < 2.0`

**Interpretive stance**
CAMS is used here as an observational instrument. It does not prove destiny. It reads system morphology: viability, stress, coordination, memory, symbolic coherence, and adaptive headroom.
"""


# ---------------------------------------------------------------------------
# Column handling
# ---------------------------------------------------------------------------

ALIASES: Dict[str, List[str]] = {
    "society": ["society", "nation", "country", "civilisation", "civilization", "polity"],
    "year": ["year", "date", "time", "t"],
    "node": ["node", "locus", "domain", "function"],
    "C": ["coherence", "c", "c_i", "ci"],
    "K": ["capacity", "k", "k_i", "ki"],
    "S": ["stress", "s", "s_i", "si"],
    "A": ["abstraction", "a", "a_i", "ai"],
    "V": ["node value", "node_value", "v", "v_i", "vi"],
    "B": ["bond strength", "bond_strength", "bond", "b", "b_i", "bi", "bs"],
    "V_min": ["v_min", "vmin", "node value min", "node_value_min"],
    "V_max": ["v_max", "vmax", "node value max", "node_value_max"],
    "V_range": ["v_range", "vrange", "node value range", "node_value_range"],
}


def norm_col(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(name).strip().lower())


def find_col(df: pd.DataFrame, logical_name: str, required: bool = False) -> Optional[str]:
    normalised = {norm_col(c): c for c in df.columns}
    for alias in ALIASES.get(logical_name, []):
        key = norm_col(alias)
        if key in normalised:
            return normalised[key]
    if required:
        raise ValueError(
            f"Missing required column for {logical_name!r}. "
            f"Available columns: {list(df.columns)}"
        )
    return None


def canonicalise_node(node: str) -> str:
    raw = str(node).strip()
    aliases = {
        "executive": "Helm",
        "state": "Helm",
        "government": "Helm",
        "army": "Shield",
        "military": "Shield",
        "security": "Shield",
        "priesthood": "Lore",
        "ideology": "Lore",
        "culture": "Lore",
        "property owners": "Stewards",
        "owners": "Stewards",
        "capital": "Stewards",
        "trades": "Craft",
        "professions": "Craft",
        "technical": "Craft",
        "proletariat": "Hands",
        "labour": "Hands",
        "labor": "Hands",
        "people": "Hands",
        "state memory": "Archive",
        "memory": "Archive",
        "bureaucracy": "Archive",
        "storekeepers": "Flow",
        "merchants": "Flow",
        "commerce": "Flow",
    }
    low = raw.lower()
    return aliases.get(low, raw.title() if low not in [n.lower() for n in NODES] else next(n for n in NODES if n.lower() == low))


def load_mean_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)

    col_soc = find_col(df, "society", required=False)
    col_year = find_col(df, "year", required=True)
    col_node = find_col(df, "node", required=True)
    col_c = find_col(df, "C", required=True)
    col_k = find_col(df, "K", required=True)
    col_s = find_col(df, "S", required=True)
    col_a = find_col(df, "A", required=True)
    col_v = find_col(df, "V", required=False)
    col_b = find_col(df, "B", required=False)

    out = pd.DataFrame()
    out["Society"] = df[col_soc].astype(str) if col_soc else ""
    out["Year"] = pd.to_numeric(df[col_year], errors="coerce").astype("Int64")
    out["Node"] = df[col_node].map(canonicalise_node)
    out["C"] = pd.to_numeric(df[col_c], errors="coerce")
    out["K"] = pd.to_numeric(df[col_k], errors="coerce")
    out["S"] = pd.to_numeric(df[col_s], errors="coerce")
    out["A"] = pd.to_numeric(df[col_a], errors="coerce")
    out["V_calc"] = out["C"] + out["K"] - out["S"] + 0.5 * out["A"]
    out["V"] = pd.to_numeric(df[col_v], errors="coerce") if col_v else out["V_calc"]
    out["sigma"] = (out["A"] * out["C"]) * (out["K"] - out["S"])
    if col_b:
        out["B"] = pd.to_numeric(df[col_b], errors="coerce")
    else:
        out["B"] = float("nan")

    out = out.dropna(subset=["Year", "Node", "C", "K", "S", "A"])
    out["Year"] = out["Year"].astype(int)
    node_order = {n: i for i, n in enumerate(NODES)}
    out["node_order"] = out["Node"].map(lambda n: node_order.get(n, 99))
    return out.sort_values(["Year", "node_order", "Node"]).drop(columns=["node_order"])


def load_envelope_csv(path: Optional[Path]) -> Optional[pd.DataFrame]:
    if not path:
        return None
    if not path.exists():
        raise FileNotFoundError(path)

    df = pd.read_csv(path)
    col_soc = find_col(df, "society", required=False)
    col_year = find_col(df, "year", required=True)
    col_node = find_col(df, "node", required=True)
    col_vmin = find_col(df, "V_min", required=False)
    col_vmax = find_col(df, "V_max", required=False)
    col_vrange = find_col(df, "V_range", required=False)

    out = pd.DataFrame()
    out["Society"] = df[col_soc].astype(str) if col_soc else ""
    out["Year"] = pd.to_numeric(df[col_year], errors="coerce").astype("Int64")
    out["Node"] = df[col_node].map(canonicalise_node)
    if col_vmin:
        out["V_min"] = pd.to_numeric(df[col_vmin], errors="coerce")
    if col_vmax:
        out["V_max"] = pd.to_numeric(df[col_vmax], errors="coerce")
    if col_vrange:
        out["V_range"] = pd.to_numeric(df[col_vrange], errors="coerce")
    return out.dropna(subset=["Year", "Node"]).assign(Year=lambda x: x["Year"].astype(int))


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------

def fmt(x: object, digits: int = 2) -> str:
    try:
        if x is None or pd.isna(x):
            return "n/a"
        f = float(x)
        if abs(f - round(f)) < 1e-9:
            return str(int(round(f)))
        return f"{f:.{digits}f}"
    except Exception:
        return str(x)


def md_table(rows: List[Iterable[object]], headers: List[str]) -> str:
    text = "| " + " | ".join(headers) + " |\n"
    text += "| " + " | ".join(["---"] * len(headers)) + " |\n"
    for row in rows:
        text += "| " + " | ".join(str(x) for x in row) + " |\n"
    return text.rstrip()


def latest_snapshot(df: pd.DataFrame) -> pd.DataFrame:
    latest_year = int(df["Year"].max())
    snap = df[df["Year"] == latest_year].copy()
    node_order = {n: i for i, n in enumerate(NODES)}
    snap["node_order"] = snap["Node"].map(lambda n: node_order.get(n, 99))
    return snap.sort_values(["node_order", "Node"]).drop(columns=["node_order"])


def system_yearly(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for year, g in df.groupby("Year"):
        rows.append({
            "Year": int(year),
            "V_bar": g["V"].mean(),
            "sigma_V": g["V"].std(ddof=0),
            "Stress_bar": g["S"].mean(),
            "C_bar": g["C"].mean(),
            "K_bar": g["K"].mean(),
            "A_bar": g["A"].mean(),
            "B_bar": g["B"].mean() if "B" in g else float("nan"),
        })
    return pd.DataFrame(rows).sort_values("Year")


def classify_attractor(v_bar: float, sigma_v: float) -> str:
    if pd.isna(v_bar) or pd.isna(sigma_v):
        return "Unclassified"
    if v_bar < 0:
        return "Thermodynamic Freeze candidate"
    if v_bar < 12 and sigma_v > 3.5:
        return "Fracture / regime-transition risk"
    if v_bar < 12:
        return "Low-headroom crisis floor"
    if v_bar > 15 and sigma_v < 2.0:
        return "Buffering / disregard trigger satisfied"
    if sigma_v > 3.5:
        return "High-dispersion oscillation"
    return "Managed oscillation / adaptive tension"


def loop_summary(snap: pd.DataFrame) -> Dict[str, float]:
    fast = snap[snap["Node"].isin(FAST_LOOP)]
    slow = snap[snap["Node"].isin(SLOW_LOOP)]
    return {
        "fast_V": fast["V"].mean(),
        "slow_V": slow["V"].mean(),
        "fast_S": fast["S"].mean(),
        "slow_S": slow["S"].mean(),
        "fast_B": fast["B"].mean(),
        "slow_B": slow["B"].mean(),
    }


def library_attractor_proxy(snap: pd.DataFrame) -> Optional[float]:
    try:
        b_lore = float(snap.loc[snap["Node"] == "Lore", "B"].iloc[0])
        b_archive = float(snap.loc[snap["Node"] == "Archive", "B"].iloc[0])
        s_hands = float(snap.loc[snap["Node"] == "Hands", "S"].iloc[0])
        if s_hands <= 0:
            return None
        return (b_lore * b_archive) / s_hands
    except Exception:
        return None


def dominant_nodes(snap: pd.DataFrame, n: int = 3) -> Tuple[List[str], List[str], List[str]]:
    strongest = snap.sort_values("V", ascending=False).head(n)["Node"].tolist()
    stressed = snap.sort_values("S", ascending=False).head(n)["Node"].tolist()
    activated = snap.sort_values("sigma", ascending=False).head(n)["Node"].tolist()
    return strongest, stressed, activated


def phase_rows(df: pd.DataFrame, max_rows: int = 12) -> List[List[str]]:
    yearly = system_yearly(df).copy()
    yearly["Period"] = (yearly["Year"] // 10) * 10
    decade = yearly.groupby("Period").agg(
        V_bar=("V_bar", "mean"),
        sigma_V=("sigma_V", "mean"),
        Stress_bar=("Stress_bar", "mean"),
        B_bar=("B_bar", "mean"),
    ).reset_index()

    # Keep a compact phase table: first, last, lowest V, highest stress, highest dispersion, plus spread.
    interesting_periods = set()
    if not decade.empty:
        interesting_periods.add(int(decade.iloc[0]["Period"]))
        interesting_periods.add(int(decade.iloc[-1]["Period"]))
        for col in ["V_bar", "Stress_bar", "sigma_V"]:
            if col == "V_bar":
                idx = decade[col].idxmin()
            else:
                idx = decade[col].idxmax()
            interesting_periods.add(int(decade.loc[idx, "Period"]))

    # Add evenly spaced decades if there is room.
    for p in decade["Period"].tolist():
        if len(interesting_periods) >= max_rows:
            break
        interesting_periods.add(int(p))

    selected = decade[decade["Period"].isin(sorted(interesting_periods))].sort_values("Period").head(max_rows)
    rows = []
    for _, r in selected.iterrows():
        label = f"{int(r['Period'])}s"
        rows.append([
            label,
            fmt(r["V_bar"]),
            fmt(r["sigma_V"]),
            fmt(r["Stress_bar"]),
            fmt(r["B_bar"]),
            classify_attractor(float(r["V_bar"]), float(r["sigma_V"])),
        ])
    return rows


def envelope_latest_rows(envelope: Optional[pd.DataFrame], latest_year: int) -> str:
    if envelope is None:
        return "_No uncertainty envelope supplied._"
    env = envelope[envelope["Year"] == latest_year].copy()
    if env.empty:
        return "_No envelope rows found for the latest year._"
    cols = [c for c in ["Node", "V_min", "V_max", "V_range"] if c in env.columns]
    if not cols:
        return "_Envelope supplied, but no V_min/V_max/V_range columns were found._"
    if "V_range" in env.columns:
        env = env.sort_values("V_range", ascending=False)
    node_order = {n: i for i, n in enumerate(NODES)}
    env["node_order"] = env["Node"].map(lambda n: node_order.get(n, 99))
    env = env.sort_values(["node_order", "Node"])
    headers = [c.replace("_", " ") for c in cols]
    rows = [[fmt(v) if c != "Node" else v for c, v in zip(cols, row)] for row in env[cols].values]
    return md_table(rows, headers)


# ---------------------------------------------------------------------------
# Markdown generation
# ---------------------------------------------------------------------------

def blank_report(society: str, sybond_name: str) -> str:
    return f"""# Sybond Report Template: {society}

**Sybond name:** {sybond_name}  
**Framework:** CAMS v3.2-R / ESCH-compatible 10-section template  
**Status:** Blank template ready for scoring, evidence, and narrative completion.

{CAMS_FORMULATION}

---

## 1. Sybond Name, Geographic Type, and Origin

**Sybond name:** {sybond_name}  
**Geographic type:** [continental core / maritime archipelago / frontier state / riverine civilisation / island microstate / imperial network / other]  
**Historical origin:** [short origin narrative]

Write this section as the sybond’s birth certificate. Describe the ecological base, settlement pattern, founding pressures, memory traditions, and the original coordination problem that forced this society into shape.

---

## 2. Core Purpose and Function

Define what the sybond does in the world.

Use this question: **what coordination problem does this society solve better than its neighbours?**

Include:
- Core adaptive function
- Dominant survival logic
- Civilisational or institutional signature
- Main energetic / thermodynamic pattern

---

## 3. Composition: Nodes and Functional Elements

Use the eight CAMS nodes.

{md_table([[n, "Fast" if n in FAST_LOOP else "Slow", NODE_FUNCTIONS[n], "[C/K/S/A/B reading]"] for n in NODES], ["Node", "Loop", "Function", "Dataset reading"])}

---

## 4. Mechanisms of Coherence and Unity

Explain how the sybond holds itself together.

Look for:
- Archive continuity
- Lore legitimacy
- Helm authority
- Shield boundary enforcement
- Flow circulation
- Hands consent or endurance
- Craft competence
- Steward allocation

---

## 5. Adaptive Capacity and Resilience

Interpret the system’s ability to absorb shock, repair itself, and re-synchronise.

Use:
- V_bar for viability
- sigma_V for node dispersion
- Stress level and stress-rate dispersion where available
- Bond Strength / coupling where available
- Library Attractor proxy where available

---

## 6. Recommended Visualisations

Recommended standard pack:
1. System Health / V_bar over time
2. Bond Strength over time
3. Node heatmap across C/K/S/A
4. Latest-year radar chart
5. Stress topology map
6. Phase-space plot: V_bar vs sigma_V
7. ESCH activation map: sigma_i by node

---

## 7. Environmental and External Stressors

Describe external and ecological pressures.

Include:
- Geography
- Energy and resource base
- Neighbours and rivals
- Trade system exposure
- Imperial or great-power pressure
- Climate, demography, migration, technology
- Narrative warfare / information pressure where relevant

---

## 8. Interaction with Surroundings and Other Societies

Explain how the sybond couples to other sybonds.

Look for:
- Tributary, imperial, alliance, trade, civilisational, or dependency patterns
- Entropy import/export
- Cultural export
- Security dependency
- Strategic autonomy or subordination

---

## 9. Evolutionary Patterns and Phases

Create a phase table using the dataset.

Suggested columns:
- Period
- V_bar
- sigma_V
- dominant stressor
- repair mechanism
- attractor state

---

## 10. Broader Insights and Implications

Close with the systemic diagnosis.

Answer:
- What is this sybond’s genius?
- What is its recurring failure mode?
- What keeps it alive?
- What would break it?
- What does the latest snapshot imply?

**One-sentence kernel:** [Write the simplest possible truth about the sybond.]
"""


def filled_report(
    df: pd.DataFrame,
    envelope: Optional[pd.DataFrame],
    society: str,
    sybond_name: str,
) -> str:
    snap = latest_snapshot(df)
    latest_year = int(snap["Year"].max())
    yearly = system_yearly(df)
    latest_metrics = yearly[yearly["Year"] == latest_year].iloc[0]
    v_bar = float(latest_metrics["V_bar"])
    sigma_v = float(latest_metrics["sigma_V"])
    attractor = classify_attractor(v_bar, sigma_v)
    loops = loop_summary(snap)
    eta_loop = library_attractor_proxy(snap)

    strongest, stressed, activated = dominant_nodes(snap)

    long_avg = df.groupby("Node").agg(
        V=("V", "mean"),
        C=("C", "mean"),
        K=("K", "mean"),
        S=("S", "mean"),
        A=("A", "mean"),
        B=("B", "mean"),
        sigma=("sigma", "mean"),
    ).reset_index()
    node_order = {n: i for i, n in enumerate(NODES)}
    long_avg["node_order"] = long_avg["Node"].map(lambda n: node_order.get(n, 99))
    long_avg = long_avg.sort_values(["node_order", "Node"]).drop(columns=["node_order"])

    latest_rows = []
    for _, r in snap.iterrows():
        latest_rows.append([
            r["Node"],
            fmt(r["C"]),
            fmt(r["K"]),
            fmt(r["S"]),
            fmt(r["A"]),
            fmt(r["V"]),
            fmt(r["sigma"]),
            fmt(r["B"]),
            FAILURE_MODES.get(r["Node"], ""),
        ])

    long_rows = []
    for _, r in long_avg.iterrows():
        long_rows.append([
            r["Node"],
            "Fast" if r["Node"] in FAST_LOOP else "Slow",
            fmt(r["V"]),
            fmt(r["C"]),
            fmt(r["K"]),
            fmt(r["S"]),
            fmt(r["A"]),
            fmt(r["B"]),
        ])

    crisis_years = yearly[(yearly["V_bar"] < 12) | (yearly["sigma_V"] > 3.5)].copy()
    if not crisis_years.empty:
        crisis_short = crisis_years.sort_values(["V_bar", "sigma_V"]).head(12)
        crisis_text = md_table(
            [[int(r["Year"]), fmt(r["V_bar"]), fmt(r["sigma_V"]), fmt(r["Stress_bar"]), classify_attractor(float(r["V_bar"]), float(r["sigma_V"]))]
             for _, r in crisis_short.iterrows()],
            ["Year", "V_bar", "sigma_V", "Stress_bar", "Reading"]
        )
    else:
        crisis_text = "_No years crossed the V_bar < 12 or sigma_V > 3.5 crisis/dispersion thresholds._"

    phase_text = md_table(
        phase_rows(df),
        ["Period", "V_bar", "sigma_V", "Stress_bar", "B_bar", "Attractor reading"]
    )

    envelope_text = envelope_latest_rows(envelope, latest_year)

    eta_text = fmt(eta_loop) if eta_loop is not None else "n/a"

    start_year = int(df["Year"].min())
    end_year = int(df["Year"].max())

    return f"""# Sybond Report: {society}, {start_year}-{end_year}

**Sybond name:** {sybond_name}  
**Framework:** CAMS v3.2-R / ESCH-compatible 10-section report  
**Latest snapshot:** {latest_year}  
**Generated by:** sybond_report_exporter.py

{CAMS_FORMULATION}

---

## Snapshot Dashboard: {latest_year}

{md_table([
    ["V_bar", fmt(v_bar), "Mean canonical node value"],
    ["sigma_V", fmt(sigma_v), "Population dispersion of node values"],
    ["Mean Stress", fmt(latest_metrics["Stress_bar"]), "Average node stress"],
    ["Mean Bond Strength", fmt(latest_metrics["B_bar"]), "Average supplied bond/coupling proxy"],
    ["Fast-loop V", fmt(loops["fast_V"]), "Helm, Shield, Flow, Hands, Craft"],
    ["Slow-loop V", fmt(loops["slow_V"]), "Archive, Lore, Stewards"],
    ["Library Attractor proxy", eta_text, "(B_Lore * B_Archive) / S_Hands"],
    ["Attractor reading", attractor, "Threshold-based diagnostic"],
], ["Metric", "Value", "Meaning"])}

**Strongest nodes:** {", ".join(strongest)}  
**Most stressed nodes:** {", ".join(stressed)}  
**Highest ESCH activation:** {", ".join(activated)}

---

## 1. Sybond Name, Geographic Type, and Origin

**Sybond name:** {sybond_name}  
**Geographic type:** [complete manually: continental core / maritime system / riverine civilisation / frontier state / island system / imperial network / other]  
**Origin narrative:** [complete manually]

Dataset clue: this sybond’s measured period runs from **{start_year} to {end_year}**. The report should connect the pre-dataset origin story to the measured CAMS morphology rather than pretending the CSV contains the whole origin history.

---

## 2. Core Purpose and Function

The core function should be inferred from the strongest and most persistent nodes.

Long-run node profile:

{md_table(long_rows, ["Node", "Loop", "Mean V", "Mean C", "Mean K", "Mean S", "Mean A", "Mean B"])}

Working interpretation prompt:

> This sybond appears to solve its survival problem through the repeated alignment of **[dominant nodes]**. Its adaptive purpose is **[write purpose]**, while its recurring stress signature appears in **[stressed nodes]**.

---

## 3. Composition: Nodes and Functional Elements

Latest node-level reading:

{md_table(latest_rows, ["Node", "C", "K", "S", "A", "V", "sigma", "B", "Failure mode to watch"])}

Interpretive notes:
- **V** reads structural viability/headroom.
- **sigma** reads ESCH cognitive activation: where abstraction and coherence are energised by capacity exceeding stress.
- A node can be highly activated but still structurally strained, so read V and sigma together.

---

## 4. Mechanisms of Coherence and Unity

Read coherence through the relationship among Archive, Lore, Helm, and Hands.

Current coherence clues:
- Fast-loop V: **{fmt(loops["fast_V"])}**
- Slow-loop V: **{fmt(loops["slow_V"])}**
- Fast-loop stress: **{fmt(loops["fast_S"])}**
- Slow-loop stress: **{fmt(loops["slow_S"])}**
- Library Attractor proxy: **{eta_text}**

Draft interpretation:

> The sybond’s unity is likely produced by **[Archive/Lore/Helm/Flow/etc.]**. Where coherence fails, the likely fracture line is **[Hands/Helm/Stewards/Flow/etc.]**.

---

## 5. Adaptive Capacity and Resilience

Threshold evidence:

{crisis_text}

Resilience reading:
- If **V_bar < 12**, the system is near or below the crisis floor.
- If **sigma_V > 3.5**, node divergence is high enough to imply serious shear.
- If **V_bar > 15 and sigma_V < 2.0**, the disregard/buffering trigger is satisfied.
- Current reading for {latest_year}: **{attractor}**.

---

## 6. Recommended Visualisations

Standard visual pack for this sybond:

1. **System Health / V_bar over time** — shows viability cycles and repair phases.
2. **sigma_V over time** — shows node divergence and shear.
3. **Bond Strength over time** — shows coupling, if bond data exists.
4. **Node heatmap** — C/K/S/A by node and year.
5. **Latest-year radar** — C/K/S/A profile for the current snapshot.
6. **Stress topology map** — stress differentials across the eight-node network.
7. **ESCH activation plot** — sigma_i by node, showing the active cognitive morphology.

---

## 7. Environmental and External Stressors

Complete manually using historical and ecological evidence.

Suggested headings:
- Geography and resource base
- Energy regime
- Neighbour and rival pressure
- Trade exposure
- Military/security pressure
- Demography and migration
- Technological transition
- Narrative/information pressure

Dataset prompt:

> External stressors should explain the largest rises in Stress_bar, the weakest V_bar periods, and the most stressed nodes: **{", ".join(stressed)}**.

---

## 8. Interaction with Surroundings and Other Societies

Complete manually.

Read the sybond as coupled to other sybonds through:
- Trade and Flow
- War and Shield
- Cultural export and Lore
- Institutional transfer and Archive
- Capital and Stewards
- Technology and Craft
- Alliance/subordination/autonomy via Helm

Interpretive prompt:

> Does this sybond export order, import entropy, absorb shocks, act as a buffer, or attempt strategic autonomy?

---

## 9. Evolutionary Patterns and Phases

Compact phase table:

{phase_text}

Suggested narrative:

> The sybond’s evolutionary pattern should be written as a sequence of attractor states: formation, consolidation, expansion, crisis, repair, oscillation, fracture, or re-synchronisation.

---

## 10. Broader Insights and Implications

Current systemic diagnosis:

> In {latest_year}, **{sybond_name}** reads as **{attractor}**, with strongest support from **{", ".join(strongest)}** and greatest stress around **{", ".join(stressed)}**. Its immediate question is whether its coherence mechanisms can keep pace with its stress morphology.

One-sentence kernel:

> **[Write the simplest possible sentence: “{sybond_name} survives by ___, but is threatened when ___.”]**

---

## Appendix: Latest-Year Uncertainty Envelope

{envelope_text}
"""


def write_report(text: str, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(text, encoding="utf-8")


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Export a CAMS v3.2-R / ESCH-compatible 10-section Sybond Report Template."
    )
    p.add_argument("--mean-csv", type=Path, help="CAMS ensemble mean CSV.")
    p.add_argument("--envelope-csv", type=Path, help="Optional CAMS envelope/uncertainty CSV.")
    p.add_argument("--society", default="Unnamed Society", help="Society/civilisation/nation name.")
    p.add_argument("--sybond-name", default="Unnamed Sybond", help="Mythic/system name for the sybond.")
    p.add_argument("--output", type=Path, default=Path("sybond_report.md"), help="Output Markdown file.")
    p.add_argument("--blank", action="store_true", help="Export a blank template instead of reading CSV data.")
    return p.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)

    if args.blank:
        report = blank_report(args.society, args.sybond_name)
        write_report(report, args.output)
        print(f"Wrote blank Sybond template to: {args.output}")
        return 0

    if not args.mean_csv:
        raise SystemExit("Provide --mean-csv, or use --blank for an empty template.")

    if not args.mean_csv.exists():
        raise SystemExit(f"Mean CSV not found: {args.mean_csv}")

    df = load_mean_csv(args.mean_csv)

    # If society was not supplied explicitly, infer it from the CSV if possible.
    society = args.society
    if society == "Unnamed Society" and "Society" in df.columns:
        names = [x for x in df["Society"].dropna().unique().tolist() if str(x).strip()]
        if names:
            society = str(names[0])

    envelope = load_envelope_csv(args.envelope_csv) if args.envelope_csv else None
    report = filled_report(df, envelope, society, args.sybond_name)
    write_report(report, args.output)
    print(f"Wrote Sybond report to: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
