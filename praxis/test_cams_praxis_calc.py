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
    a = c.regime_spread(*M.T, "v1.0.1", n=300, noise_sd=0.5, seed=7)
    b = c.regime_spread(*M.T, "v1.0.1", n=300, noise_sd=0.5, seed=7)
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
