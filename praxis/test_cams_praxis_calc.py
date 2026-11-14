"""Tests for the Praxis reference calculator. Run: pytest praxis/"""
import json
import os

import numpy as np

import cams_praxis_calc as c

HERE = os.path.dirname(__file__)
VEC = json.load(open(os.path.join(HERE, "test_vectors.json"), encoding="utf-8"))


def test_self_test():
    c.self_test()


def test_vectors_reproduce():
    for v in VEC["vectors"]:
        M = np.array([v["nodes"][n] for n in c.NODES], dtype=float)
        fg = c.phase_space(M[:, 0], M[:, 1], M[:, 2], M[:, 3])
        for k, want in v["expect"].items():
            assert abs(fg[k] - want) < 1e-5, (v["id"], k, fg[k], want)
        for name, want in v["regime"].items():
            assert c.classify(fg, name) == want, (v["id"], name)


def test_bond_mean_uses_28_pairs():
    M = np.array([VEC["vectors"][1]["nodes"][n] for n in c.NODES], dtype=float)
    B = c.bond_matrix(M[:, 0], M[:, 2], M[:, 3])
    pairs = [B[i, j] for i in range(8) for j in range(i + 1, 8)]
    assert len(pairs) == 28
    assert abs(np.mean(pairs) - c.phase_space(*M.T)["B_mean"]) < 1e-12


def test_spread_is_seeded_and_sums_to_one():
    M = np.array([VEC["vectors"][1]["nodes"][n] for n in c.NODES], dtype=float)
    a = c.regime_spread(*M.T, c.CANONICAL, n=300, noise_sd=0.5, seed=7)
    b = c.regime_spread(*M.T, c.CANONICAL, n=300, noise_sd=0.5, seed=7)
    assert a == b and abs(sum(a.values()) - 1) < 1e-3


def test_missing_node_is_refused(tmp_path):
    p = tmp_path / "bad.csv"
    p.write_text("Society,Year,Node,Coherence,Capacity,Stress,Abstraction\nX,2000,Helm,5,5,5,5\n")
    try:
        c.read_scores(str(p))
    except ValueError as e:
        assert "eight canonical nodes" in str(e)
    else:
        raise AssertionError("incomplete node set was accepted")


def _M(i):
    return np.array([VEC["vectors"][i]["nodes"][n] for n in c.NODES], dtype=float)


def test_canonical_is_v1_2_final_and_legacy_is_named():
    r = c.analyse({n: tuple(v) for n, v in VEC["vectors"][2]["nodes"].items()})  # Germany 2024
    assert r["regime"]["classifier"] == "v1.2-final" and r["regime"]["status"] == "canonical"
    assert r["legacy_comparator"]["classifier"] == "v1.0.1"
    # the two instruments disagree on this anchor case and both labels are kept, untranslated
    assert r["regime"]["label"] == "Phantom Type II"
    assert r["legacy_comparator"]["label"] == "Local Node Failure"
    assert r["threshold_distances"]["s_min"]["nearest_threshold"] == -0.65


def test_sector_unit_gets_no_regime():
    r = c.analyse({n: tuple(v) for n, v in VEC["vectors"][2]["nodes"].items()}, unit="sector", spread=500)
    assert r["regime"] is None and r["regime_withheld_reason"]
    for k in ("legacy_comparator", "regime_spread", "threshold_distances"):
        assert k not in r
    fr = r["functional_relationships"]
    assert len(fr["bond_matrix"]) == 8 and len(fr["strongest_pairs"]) == 5 and len(fr["weakest_pairs"]) == 5
    assert fr["strongest_pairs"][0]["B"] >= fr["weakest_pairs"][0]["B"]


def test_activation_modes_differ_only_at_equality_and_never_change_a_label():
    rng = np.random.default_rng(3)
    for _ in range(3000):
        M = rng.integers(1, 11, (8, 4)).astype(float)
        f = c.phase_space(*M.T, activation_mode="formal")
        g = c.phase_space(*M.T, activation_mode="v2.4-compat")
        eq = M[:, 1] == M[:, 2]
        assert np.allclose(f["s"][~eq], g["s"][~eq])
        assert (f["s"][eq] == 0).all() and (g["s"][eq] > 0).all()
        for name in c.CLASSIFIERS:
            assert c.classify(f, name) == c.classify(g, name)


def test_compat_mode_matches_existing_engine():
    import sys
    sys.path.insert(0, os.path.join(HERE, ".."))
    try:
        import cams_framework_v2_4 as old
    except Exception:  # engine has heavier dependencies; skip if unavailable
        return
    rng = np.random.default_rng(4)
    for _ in range(500):
        M = rng.integers(1, 11, (8, 4)).astype(float)
        C, K, S, A = M.T
        assert np.allclose(c.cognitive_activation(C, K, S, A, "v2.4-compat"), old.esch_activation(C, K, S, A))
        assert np.allclose(c.bond_matrix(C, S, A), old.bond_matrix(C, A, S))


def test_unknown_names_are_refused():
    for bad in (lambda: c.classify({"V_mean": 1, "V_min": 1, "B_mean": 1, "s_min": 1}, "v9"),
                lambda: c.cognitive_activation([5], [5], [5], [5], "guess"),
                lambda: c.analyse({n: (5, 5, 5, 5) for n in c.NODES}, unit="planet")):
        try:
            bad()
        except ValueError:
            continue
        raise AssertionError("an unknown name was accepted")
