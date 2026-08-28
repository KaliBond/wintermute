#!/usr/bin/env python3
"""Verify JUNO v1.2 Bond_Strength_Calc is aligned to each row's Node.

Recomputes per-node mean bonds from raw Coherence / Capacity / Stress /
Abstraction using the rank-1 form:

    q_i = (0.6*C_i + 0.4*A_i) / 10
    w_i = sqrt(q_i) * 2^(-S_i/10)
    B_ij = clip(w_i * w_j, 0, 1)   for i != j;  B_ii = 0
    per-node mean bond_i = mean of the 7 off-diagonal edges

SBD is the mean of the 28 unique pairwise (upper-triangle) bonds, which is
identical to the mean of the 8 per-node means.

Usage (from repo root or juno/):
    python3 juno/verify_bond_alignment.py
"""
from __future__ import annotations

import csv
import math
import sys
from collections import defaultdict
from pathlib import Path

ATOL = 1e-4
N_NODES = 8
N_EDGES = N_NODES - 1  # 7
N_PAIRS = N_NODES * N_EDGES // 2  # 28

HERE = Path(__file__).resolve().parent
CSV_PATH = HERE / "JUNO_Unified_Dataset.csv"


def q_i(C: float, A: float) -> float:
    return (0.6 * C + 0.4 * A) / 10.0


def w_i(C: float, A: float, S: float) -> float:
    q = q_i(C, A)
    if q <= 0.0:
        return 0.0
    return math.sqrt(q) * (2.0 ** (-S / 10.0))


def bond_matrix(C, A, S):
    n = len(C)
    w = [w_i(C[i], A[i], S[i]) for i in range(n)]
    B = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            bij = w[i] * w[j]
            if bij < 0.0:
                bij = 0.0
            elif bij > 1.0:
                bij = 1.0
            B[i][j] = bij
    return B


def per_node_means(B):
    n = len(B)
    return [sum(B[i][j] for j in range(n) if j != i) / (n - 1) for i in range(n)]


def pairwise_mean(B):
    n = len(B)
    acc = 0.0
    k = 0
    for i in range(n):
        for j in range(i + 1, n):
            acc += B[i][j]
            k += 1
    return acc / k if k else float("nan")


def lambda2(B):
    """Second-smallest eigenvalue of the graph Laplacian L = D - W."""
    n = len(B)
    # Build dense Laplacian
    L = [[0.0] * n for _ in range(n)]
    for i in range(n):
        deg = 0.0
        for j in range(n):
            if i == j:
                continue
            L[i][j] = -B[i][j]
            deg += B[i][j]
        L[i][i] = deg
    # Symmetric eigendecomposition via numpy if present, else power-free jacobi-ish
    try:
        import numpy as np
        evals = np.sort(np.linalg.eigvalsh(np.array(L, dtype=float)))
        return float(evals[1])
    except ImportError:
        return float("nan")


def parse_float(s):
    if s is None:
        return None
    t = str(s).strip()
    if t == "":
        return None
    try:
        return float(t)
    except ValueError:
        return None


def close(a, b, atol=ATOL):
    return abs(a - b) <= atol


def main() -> int:
    if not CSV_PATH.exists():
        print(f"ERROR: missing {CSV_PATH}", file=sys.stderr)
        return 2

    with CSV_PATH.open(newline="") as f:
        reader = csv.DictReader(f)
        cols = reader.fieldnames or []
        rows = list(reader)

    required = [
        "Society", "Year", "Node",
        "Coherence", "Capacity", "Stress", "Abstraction",
        "Bond_Strength_Calc", "SBD_Calc", "Lambda2_Calc",
    ]
    missing = [c for c in required if c not in cols]
    if missing:
        print(f"ERROR: missing columns {missing}", file=sys.stderr)
        return 2

    has_legacy = "Bond_Strength_Calc_legacy" in cols

    groups = defaultdict(list)
    for r in rows:
        groups[(r["Society"], r["Year"])].append(r)

    n_years = 0
    n_complete = 0
    n_incomplete = 0
    exact_after = 0
    sorted_after = 0
    exact_before = 0
    sorted_before = 0
    cells_changed = 0
    cells_compared = 0
    sbd_match = 0
    lam_match = 0
    lam_checked = 0
    nv_match = 0
    nv_checked = 0
    max_bond_err = 0.0
    max_sbd_err = 0.0
    max_lam_err = 0.0
    fail_examples = []

    for (soc, yr), g in groups.items():
        n_years += 1
        complete = len(g) == N_NODES
        C, K, S, A = [], [], [], []
        if complete:
            try:
                for r in g:
                    c = float(r["Coherence"])
                    k = float(r["Capacity"])
                    s = float(r["Stress"])
                    a = float(r["Abstraction"])
                    C.append(c); K.append(k); S.append(s); A.append(a)
            except (TypeError, ValueError):
                complete = False

        if not complete:
            n_incomplete += 1
            continue
        n_complete += 1

        B = bond_matrix(C, A, S)
        means = per_node_means(B)
        sbd_re = pairwise_mean(B)
        stored = [parse_float(r["Bond_Strength_Calc"]) for r in g]
        sbd_st = parse_float(g[0]["SBD_Calc"])
        lam_st = parse_float(g[0]["Lambda2_Calc"])

        if all(v is not None for v in stored) and all(
            close(stored[i], means[i]) for i in range(N_NODES)
        ):
            exact_after += 1
        else:
            if len(fail_examples) < 5:
                fail_examples.append((soc, yr, stored, means))
        if all(v is not None for v in stored) and all(
            close(a, b) for a, b in zip(sorted(stored), sorted(means))
        ):
            sorted_after += 1

        for i in range(N_NODES):
            if stored[i] is not None:
                max_bond_err = max(max_bond_err, abs(stored[i] - means[i]))

        if sbd_st is not None:
            err = abs(sbd_st - sbd_re)
            max_sbd_err = max(max_sbd_err, err)
            if err <= ATOL:
                sbd_match += 1

        lam_re = lambda2(B)
        if lam_st is not None and not math.isnan(lam_re):
            lam_checked += 1
            err = abs(lam_st - lam_re)
            max_lam_err = max(max_lam_err, err)
            if err <= ATOL:
                lam_match += 1

        if "Node_Value_Calc" in cols:
            for i, r in enumerate(g):
                nv = parse_float(r["Node_Value_Calc"])
                if nv is None:
                    continue
                nv_re = C[i] + K[i] - S[i] + 0.5 * A[i]
                nv_checked += 1
                if close(nv, nv_re):
                    nv_match += 1

        if has_legacy:
            legacy = [parse_float(r["Bond_Strength_Calc_legacy"]) for r in g]
            if all(v is not None for v in legacy):
                if all(close(legacy[i], means[i]) for i in range(N_NODES)):
                    exact_before += 1
                if all(close(a, b) for a, b in zip(sorted(legacy), sorted(means))):
                    sorted_before += 1
            for i in range(N_NODES):
                if stored[i] is None or legacy[i] is None:
                    continue
                cells_compared += 1
                if abs(stored[i] - legacy[i]) > ATOL:
                    cells_changed += 1

    print("JUNO v1.2 Bond_Strength_Calc alignment verification")
    print(f"  file: {CSV_PATH}")
    print(f"  rows: {len(rows)}")
    print(f"  society-years: {n_years}  complete (8 nodes + CKSA): {n_complete}  incomplete: {n_incomplete}")
    print()
    print("Per-node mean bond vs Bond_Strength_Calc (atol=1e-4)")
    if has_legacy:
        print(f"  years exact-aligned BEFORE (legacy): {exact_before}/{n_complete}")
        print(f"  years sorted-value match BEFORE:     {sorted_before}/{n_complete}")
    print(f"  years exact-aligned AFTER:           {exact_after}/{n_complete}")
    print(f"  years sorted-value match AFTER:      {sorted_after}/{n_complete}")
    print(f"  max |Bond_Strength_Calc - recomputed|: {max_bond_err:.6e}")
    if has_legacy:
        print(f"  Bond_Strength_Calc cells changed vs legacy (>{ATOL:g}): {cells_changed}/{cells_compared}")
    print()
    print("SBD_Calc vs mean of 28 unique pairwise bonds")
    print(f"  match within {ATOL:g}: {sbd_match}/{n_complete}  max |diff|: {max_sbd_err:.6e}")
    print()
    print("Lambda2_Calc vs Laplacian λ2 (unchanged column)")
    print(f"  match within {ATOL:g}: {lam_match}/{lam_checked}  max |diff|: {max_lam_err:.6e}")
    if nv_checked:
        print()
        print("Node_Value_Calc vs V = C + K - S + 0.5*A")
        print(f"  match within {ATOL:g}: {nv_match}/{nv_checked}")

    ok = (
        n_complete > 0
        and exact_after == n_complete
        and sbd_match == n_complete
        and (lam_checked == 0 or lam_match == lam_checked)
    )
    if fail_examples and not ok:
        print("\nFirst alignment failures:")
        for soc, yr, stored, means in fail_examples:
            print(f"  {soc} {yr}")
            for s, m in zip(stored, means):
                print(f"    stored={s}  recomputed={m}")

    if not ok:
        print("\nASSERT FAILED: corrected Bond_Strength_Calc does not match recomputation.")
        return 1

    print("\nASSERTS PASSED: Bond_Strength_Calc matches recomputed per-node means;")
    print("SBD_Calc matches mean of 28 unique pairwise bonds;")
    print("Lambda2_Calc matches recomputed λ2 (column not rewritten).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
