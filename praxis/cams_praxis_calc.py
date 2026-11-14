#!/usr/bin/env python3
"""
cams_praxis_calc.py — reference calculator for CAMS Praxis.

Purpose
-------
One small file the Praxis assistant must RUN for every number it reports.
It never does CAMS arithmetic "in its head".

Canonical classifier: JUNO v1.2-Final
    source: juno/JUNO_v1.2-Final_Formalism.md, section 3 (6 July 2026).
    This is the classifier named in the lodged 2027-2028 prospective register.

Legacy comparator: v1.0.1
    source: CAMS PRAXIS GPT Build Pack v1.0 (14 November 2026), Part 2 section 2.
    Reported beside the canonical label, always under its own name. It is a
    separate instrument. Its labels are never translated into v1.2-Final
    labels or the reverse, and nothing here rewrites stored historical
    outputs or lodged predictions.

Exact cascades, thresholds, rounding and provenance for both are set out in
praxis/CLASSIFIER_SPECIFICATIONS.md. How often they disagree is set out in
praxis/concordance/CLASSIFIER_CONCORDANCE.md.

Unit of analysis
----------------
    --unit nation   (default) metrics, canonical regime, legacy comparator
    --unit sector   metrics and functional relationships only. No regime
                    label is produced: the thresholds were fitted on
                    nation-level societies. Use this for the soil-sector
                    pilot and any other non-national unit.

Activation when Capacity equals Stress
--------------------------------------
    --activation-mode formal        (default) the formula as written:
                                    sign(0) = 0, so s = 0 when K == S.
    --activation-mode v2.4-compat   reproduces cams_framework_v2_4.py:
                                    K == S is read as K - S = +0.1, and no
                                    floor is applied elsewhere.
The two never give different regime labels (see CLASSIFIER_SPECIFICATIONS.md
section 5 for the argument and the corpus check). Which reading the formal
definition should carry is a custodian decision.

Usage
-----
    python cams_praxis_calc.py scores.csv
    python cams_praxis_calc.py scores.csv --spread 2000 --noise-sd 0.5 --seed 1
    python cams_praxis_calc.py soil_scores.csv --unit sector
    python cams_praxis_calc.py --self-test

Input CSV: long format, one row per node, columns
    Society, Year, Node, Coherence, Capacity, Stress, Abstraction
(extra columns are ignored). Exactly the eight canonical nodes per unit-year.
Scores follow CAMS_SCORING_PROTOCOL_V2_1.md: 1 to 10.

Only numpy is required.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import OrderedDict

import numpy as np

CALC_VERSION = "praxis-calc 1.1.0"
NODES = ["Helm", "Shield", "Lore", "Stewards", "Craft", "Hands", "Archive", "Flow"]
CANONICAL = "v1.2-final"
LEGACY = "v1.0.1"
CLASSIFIERS = (CANONICAL, LEGACY)
ACTIVATION_MODES = ("formal", "v2.4-compat")
UNITS = ("nation", "sector")
ROUND_DP = {"v1.0.1": 4, "v1.2-final": 6}
SCORE_MIN, SCORE_MAX = 1.0, 10.0  # CAMS_SCORING_PROTOCOL_V2_1.md: integers 1 to 10


# ---------------------------------------------------------------------------
# Operators (identical in the build pack and in JUNO v1.2-Final)
# ---------------------------------------------------------------------------

def node_viability(C, K, S, A):
    """V_i = C_i + K_i - S_i + 0.5*A_i"""
    C, K, S, A = (np.asarray(x, dtype=float) for x in (C, K, S, A))
    return C + K - S + 0.5 * A


def cognitive_activation(C, K, S, A, mode="formal"):
    """Cognitive activation s_i.

    mode="formal" (default): the definition as written in
    juno/JUNO_v1.2-Final_Formalism.md section 1,

        s_i = (A_i*C_i/100) * sign(K_i - S_i) * max(|K_i - S_i|, 0.1)

    with the standard signum, sign(0) = 0. A node with K == S has s = 0.

    mode="v2.4-compat": cams_framework_v2_4.py esch_activation with its
    default singularity_floor,

        s_i = (A_i*C_i/100) * d_i,   d_i = 0.1 if K_i == S_i else K_i - S_i

    i.e. K == S is read as a small POSITIVE gap, and no floor is applied to
    0 < |K - S| < 0.1.
    """
    C, K, S, A = (np.asarray(x, dtype=float) for x in (C, K, S, A))
    d = K - S
    if mode == "formal":
        return (A * C / 100.0) * np.sign(d) * np.maximum(np.abs(d), 0.1)
    if mode == "v2.4-compat":
        return (A * C / 100.0) * np.where(d == 0.0, 0.1, d)
    raise ValueError(f"unknown activation mode {mode!r}; use one of {ACTIVATION_MODES}")


def coupling_quality(C, A):
    """q_i = (0.6*C_i + 0.4*A_i) / 10"""
    C, A = np.asarray(C, dtype=float), np.asarray(A, dtype=float)
    return (0.6 * C + 0.4 * A) / 10.0


def bond_matrix(C, S, A):
    """B_ij = sqrt(q_i*q_j) * 2^(-(S_i+S_j)/10), zero diagonal."""
    S = np.asarray(S, dtype=float)
    q = coupling_quality(C, A)
    B = np.sqrt(np.outer(q, q)) * np.power(2.0, -(S[:, None] + S[None, :]) / 10.0)
    np.fill_diagonal(B, 0.0)
    return B


def lambda2(B):
    """Second-smallest eigenvalue of the RAW Laplacian L = D - W, W = B."""
    L = np.diag(B.sum(axis=1)) - B
    return float(np.sort(np.linalg.eigvalsh(L))[1])


def phase_space(C, K, S, A, activation_mode="formal"):
    """F_G = (V_mean, s_V, V_min, B_mean, lambda2, s_min) plus per-node values.

    s_V is the population standard deviation of V (ddof = 0), matching
    cams_framework_v2_4.py. B_mean is the mean over the 28 unique pairs.
    """
    V = node_viability(C, K, S, A)
    s = cognitive_activation(C, K, S, A, activation_mode)
    q = coupling_quality(C, A)
    B = bond_matrix(C, S, A)
    n = len(V)
    iu = np.triu_indices(n, k=1)
    return {
        "V_mean": float(V.mean()),
        "s_V": float(V.std()),
        "V_min": float(V.min()),
        "B_mean": float(B[iu].mean()),
        "lambda2": lambda2(B),
        "s_min": float(s.min()),
        "V": V, "s": s, "q": q, "B": B,
    }


# ---------------------------------------------------------------------------
# Regime classifiers
# ---------------------------------------------------------------------------

def _r(fg, dp):
    return {k: round(fg[k], dp) for k in ("V_mean", "V_min", "B_mean", "s_min")}


def classify_v1_0_1(fg):
    """LEGACY COMPARATOR. Cascade exactly as written in CAMS PRAXIS GPT Build
    Pack v1.0, Part 2 section 2. Round to 4 dp, strict order, first match wins."""
    m = _r(fg, 4)
    V, Vmin, B, smin = m["V_mean"], m["V_min"], m["B_mean"], m["s_min"]
    if V < 0 and Vmin < -3 and B < 0.15:
        return "Freeze/Collapse"
    if 3 <= V <= 6 and Vmin < 0 and smin < -0.7:
        return "Phantom Type II"
    if V < 6 and Vmin < 0 and B < 0.20:
        return "Systemic Crisis"
    if Vmin < 4.5 or smin < -0.75:
        return "Local Node Failure"
    if V > 10 and B > 0.25:
        return "Stable Adaptive"
    return "Strained"


def classify_v1_2_final(fg):
    """CANONICAL. Cascade exactly as written in juno/JUNO_v1.2-Final_Formalism.md section 3.
    Round to 6 dp. Unclassified is preserved as a data-quality tripwire."""
    m = _r(fg, 6)
    V, Vmin, B, smin = m["V_mean"], m["V_min"], m["B_mean"], m["s_min"]
    if V < 0 and Vmin < -3 and B < 0.15:
        return "Freeze/Collapse"
    if V < 6 and Vmin < 0 and B < 0.20:
        return "Systemic Crisis"
    if smin < -0.65:
        return "Phantom Type II"
    if Vmin < 4.0:
        return "Local Node Failure"
    if V > 10 and Vmin > 5.0 and B > 0.28 and smin > -0.25:
        return "Stable Adaptive"
    if V >= 4:
        return "Strained"
    return "Unclassified"


def classify(fg, classifier=CANONICAL):
    if classifier == "v1.0.1":
        return classify_v1_0_1(fg)
    if classifier == "v1.2-final":
        return classify_v1_2_final(fg)
    raise ValueError(f"unknown classifier {classifier!r}; use one of {CLASSIFIERS}")


# Thresholds each classifier compares against, for the distance report.
_THRESHOLDS = {
    "v1.0.1": {"V_mean": [0, 3, 6, 10], "V_min": [-3, 0, 4.5],
               "B_mean": [0.15, 0.20, 0.25], "s_min": [-0.75, -0.7]},
    "v1.2-final": {"V_mean": [0, 4, 6, 10], "V_min": [-3, 0, 4.0, 5.0],
                   "B_mean": [0.15, 0.20, 0.28], "s_min": [-0.65, -0.25]},
}


def threshold_distances(fg, classifier):
    """For each classifier input, the nearest threshold and the signed gap
    (metric minus threshold). A small gap means the label is fragile."""
    dp = ROUND_DP[classifier]
    out = {}
    for k, ths in _THRESHOLDS[classifier].items():
        v = round(fg[k], dp)
        t = min(ths, key=lambda x: abs(v - x))
        out[k] = {"value": v, "nearest_threshold": t, "gap": round(v - t, dp)}
    return out


# ---------------------------------------------------------------------------
# Sensitivity: how stable is the label if the scores move?
# ---------------------------------------------------------------------------

def regime_spread(C, K, S, A, classifier=CANONICAL, n=2000, noise_sd=0.5, seed=0,
                  activation_mode="formal"):
    """Perturb every raw score with independent Normal(0, noise_sd) noise,
    clip to the 1-10 scale, re-classify, and return the share of draws that
    land in each regime.

    This is a SENSITIVITY ANALYSIS, not a calibrated confidence interval.
    noise_sd is the caller's assumption about scoring noise in raw score
    units; it is not derived from the between-run drift figures, and scoring
    errors in real runs are unlikely to be independent across nodes. Always
    report noise_sd, n and seed alongside the result.
    """
    rng = np.random.default_rng(seed)
    base = np.array([C, K, S, A], dtype=float)
    counts = OrderedDict()
    for _ in range(int(n)):
        p = np.clip(base + rng.normal(0.0, noise_sd, base.shape), SCORE_MIN, SCORE_MAX)
        r = classify(phase_space(*p, activation_mode=activation_mode), classifier)
        counts[r] = counts.get(r, 0) + 1
    return {k: round(v / n, 4) for k, v in sorted(counts.items(), key=lambda kv: -kv[1])}


# ---------------------------------------------------------------------------
# Input handling
# ---------------------------------------------------------------------------

def read_scores(path):
    """Return {(society, year): {node: (C, K, S, A)}} and a list of warnings.
    Duplicate (society, year, node) rows are averaged and reported."""
    acc, warns = OrderedDict(), []
    with open(path, newline="", encoding="utf-8-sig") as f:
        rd = csv.DictReader(f)
        need = ["Society", "Year", "Node", "Coherence", "Capacity", "Stress", "Abstraction"]
        missing = [c for c in need if c not in (rd.fieldnames or [])]
        if missing:
            raise ValueError(f"{path}: missing column(s) {missing}; need {need}")
        for row in rd:
            key = (row["Society"].strip(), row["Year"].strip())
            node = row["Node"].strip()
            vals = tuple(float(row[c]) for c in need[3:])
            acc.setdefault(key, OrderedDict()).setdefault(node, []).append(vals)
    out = OrderedDict()
    for key, nodes in acc.items():
        unknown = [n for n in nodes if n not in NODES]
        absent = [n for n in NODES if n not in nodes]
        if unknown or absent:
            raise ValueError(f"{key}: need exactly the eight canonical nodes; "
                             f"missing {absent}, unrecognised {unknown}")
        d = {}
        for n in NODES:
            rows = nodes[n]
            if len(rows) > 1:
                warns.append(f"{key} {n}: {len(rows)} duplicate rows averaged")
            v = tuple(float(np.mean([r[i] for r in rows])) for i in range(4))
            if any(math.isnan(x) for x in v):
                raise ValueError(f"{key} {n}: missing score (NaN); refuse to guess")
            if any(x < SCORE_MIN or x > SCORE_MAX for x in v):
                warns.append(f"{key} {n}: score outside the 1-10 protocol scale {v}; "
                             "thresholds were not calibrated for this")
            d[n] = v
        out[key] = d
    return out, warns


def functional_relationships(fg, dp=6, top=5):
    """The pairwise bond structure, for units where no regime label applies.
    Returns the full 8x8 bond matrix, the strongest and weakest pairs, and
    each node's mean bond to the other seven."""
    B = fg["B"]
    pairs = [(NODES[i], NODES[j], float(B[i, j]))
             for i in range(len(NODES)) for j in range(i + 1, len(NODES))]
    pairs.sort(key=lambda x: -x[2])
    fmt = lambda p: {"pair": [p[0], p[1]], "B": round(p[2], dp)}
    return {
        "bond_matrix": {NODES[i]: {NODES[j]: round(float(B[i, j]), dp) for j in range(len(NODES))}
                        for i in range(len(NODES))},
        "strongest_pairs": [fmt(p) for p in pairs[:top]],
        "weakest_pairs": [fmt(p) for p in pairs[-top:][::-1]],
        "node_mean_bond": {n: round(float(B[i].sum() / (len(NODES) - 1)), dp)
                           for i, n in enumerate(NODES)},
    }


def analyse(nodes, unit="nation", activation_mode="formal", spread=0, noise_sd=0.5, seed=0):
    """nodes: {node: (C, K, S, A)} for the eight canonical nodes."""
    if unit not in UNITS:
        raise ValueError(f"unknown unit {unit!r}; use one of {UNITS}")
    M = np.array([nodes[n] for n in NODES], dtype=float)
    C, K, S, A = M[:, 0], M[:, 1], M[:, 2], M[:, 3]
    fg = phase_space(C, K, S, A, activation_mode)
    dp = ROUND_DP[CANONICAL]
    res = {
        "calculator": CALC_VERSION,
        "unit": unit,
        "activation_mode": activation_mode,
        "in_protocol_range": bool(((M >= SCORE_MIN) & (M <= SCORE_MAX)).all()),
        "F_G": {k: round(fg[k], dp) for k in ("V_mean", "s_V", "V_min", "B_mean", "lambda2", "s_min")},
        "nodes": {n: {"C": M[i, 0], "K": M[i, 1], "S": M[i, 2], "A": M[i, 3],
                      "V": round(float(fg["V"][i]), dp), "s": round(float(fg["s"][i]), dp),
                      "q": round(float(fg["q"][i]), dp)}
                  for i, n in enumerate(NODES)},
        "functional_relationships": functional_relationships(fg, dp),
    }
    if unit == "sector":
        res["regime"] = None
        res["regime_withheld_reason"] = (
            "No regime label is assigned to a non-national unit. Both classifiers' "
            "thresholds were fitted on nation-level societies.")
        return res
    res["regime"] = {"classifier": CANONICAL, "status": "canonical",
                     "rounding_dp": ROUND_DP[CANONICAL], "label": classify(fg, CANONICAL)}
    res["legacy_comparator"] = {
        "classifier": LEGACY, "status": "legacy comparator",
        "rounding_dp": ROUND_DP[LEGACY], "label": classify(fg, LEGACY),
        "note": "A separate instrument. Not a translation of the canonical label."}
    res["threshold_distances"] = threshold_distances(fg, CANONICAL)
    if spread:
        res["regime_spread"] = {
            "classifier": CANONICAL, "n": int(spread), "noise_sd": noise_sd, "seed": seed,
            "note": "sensitivity analysis under an assumed noise level, not a confidence interval",
            "shares": regime_spread(C, K, S, A, CANONICAL, spread, noise_sd, seed, activation_mode)}
    return res


# ---------------------------------------------------------------------------
# Self-test: analytic cases that need no data file
# ---------------------------------------------------------------------------

def self_test():
    # 1. All eight nodes identical: C=K=S=A=5.
    #    V = 5+5-5+2.5 = 7.5 ; s = 0 ; q = 0.5 ; B_ij = 0.5 * 2^-1 = 0.25
    #    Complete graph with equal weight w has lambda2 = n*w = 8*0.25 = 2.0
    u = np.full(8, 5.0)
    fg = phase_space(u, u, u, u)
    assert abs(fg["V_mean"] - 7.5) < 1e-12 and abs(fg["s_V"]) < 1e-12
    assert abs(fg["B_mean"] - 0.25) < 1e-12
    assert abs(fg["lambda2"] - 2.0) < 1e-9
    assert fg["s_min"] == 0.0
    # 2. Sign preservation: K < S must give negative activation.
    s = cognitive_activation([8], [2], [9], [10])  # (10*8/100) * -1 * 7 = -5.6
    assert abs(s[0] + 5.6) < 1e-12
    # 3. Floor applies to magnitude only: |K-S| = 0.05 -> 0.1, sign kept.
    s = cognitive_activation([10], [5.0], [5.05], [10])
    assert abs(s[0] + 0.1) < 1e-12
    # 4. Classifier orderings differ where the documents differ.
    case = {"V_mean": 4.0, "V_min": -1.0, "B_mean": 0.18, "s_min": -0.9}
    assert classify_v1_0_1(case) == "Phantom Type II"      # Phantom tested before Systemic
    assert classify_v1_2_final(case) == "Systemic Crisis"  # Systemic tested before Phantom
    # 5. Rounding guard: 3.9999999999999996 must behave as 4.0.
    edge = {"V_mean": 8.0, "V_min": 3.9999999999999996, "B_mean": 0.3, "s_min": 0.0}
    assert classify_v1_2_final(edge) == "Strained"
    # 6. Activation modes differ only at K == S, and never below zero there.
    f = cognitive_activation([6], [5], [5], [8], "formal")[0]
    v = cognitive_activation([6], [5], [5], [8], "v2.4-compat")[0]
    assert f == 0.0 and abs(v - 0.048) < 1e-12
    # 7. A sector unit gets metrics and bonds but no regime label.
    r = analyse({n: (5, 5, 5, 5) for n in NODES}, unit="sector")
    assert r["regime"] is None and "legacy_comparator" not in r and "regime_spread" not in r
    assert len(r["functional_relationships"]["strongest_pairs"]) == 5
    # 8. A nation unit names both classifiers and keeps them apart.
    r = analyse({n: (5, 5, 5, 5) for n in NODES})
    assert r["regime"]["classifier"] == "v1.2-final" and r["legacy_comparator"]["classifier"] == "v1.0.1"
    print("self-test passed (8 checks)")


def main(argv=None):
    ap = argparse.ArgumentParser(description="CAMS Praxis reference calculator "
                                 "(canonical classifier: JUNO v1.2-Final; legacy comparator: v1.0.1)")
    ap.add_argument("csv", nargs="?", help="long-format scores CSV")
    ap.add_argument("--unit", choices=UNITS, default="nation",
                    help="nation: metrics and regime. sector: metrics and bonds only, no regime label")
    ap.add_argument("--activation-mode", choices=ACTIVATION_MODES, default="formal",
                    help="how to read K == S (default: formal definition)")
    ap.add_argument("--spread", type=int, default=0, metavar="N",
                    help="also run N perturbed draws and report canonical regime shares (nation only)")
    ap.add_argument("--noise-sd", type=float, default=0.5,
                    help="assumed scoring noise, raw score units (with --spread)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--self-test", action="store_true")
    a = ap.parse_args(argv)
    if a.self_test:
        self_test()
        return 0
    if not a.csv:
        ap.error("give a scores CSV, or --self-test")
    data, warns = read_scores(a.csv)
    out = {"input_file": a.csv, "warnings": warns, "results": []}
    for (soc, yr), nodes in data.items():
        r = analyse(nodes, a.unit, a.activation_mode, a.spread, a.noise_sd, a.seed)
        out["results"].append({"society": soc, "year": yr, **r})
    json.dump(out, sys.stdout, indent=2, ensure_ascii=False)
    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
