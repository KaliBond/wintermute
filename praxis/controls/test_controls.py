#!/usr/bin/env python3
"""
test_controls.py — do the two blind-file negative controls destroy what they
are meant to destroy, keep what they are meant to keep, and stay hidden?

Run from the repository root:
    python praxis/controls/test_controls.py

The blind-file game (test/story_maker.html) offers matched triplets:
    real   the window as scored
    perm   ROW PERMUTATION: scored rows in random order, year column kept in sequence
    shuf   NODE-LABEL SHUFFLE: one random relabelling of the eight nodes per window

This script applies the same two transformations to 24-year windows drawn from
the primary ensemble series in data/v1.0_final_recompute/ and measures:

  1. Temporal structure      lag-1 autocorrelation of yearly mean viability
  2. Marginal content        whether per-column values are unchanged
  3. Node identity           can a simple rule name the society from node-labelled levels?
  4. Label-free identity     can it do so from features that ignore row order and node labels?
  5. Detectability           can one simple statistic tell a control from real data?

It tests the METHOD on repository data. The JavaScript implementation in the
page is checked separately (row multiset preserved, relabelling consistent).
Writes CONTROL_VALIDITY.md next to this script.
"""
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "praxis"))
import cams_praxis_calc as c  # noqa: E402

SERIES = os.path.join(ROOT, "data", "v1.0_final_recompute", "series")
FG = os.path.join(ROOT, "data", "v1.0_final_recompute", "FG_corpus.csv")
W, PER_SOC, SEED = 24, 30, 20261114
DIMS = ["Coherence", "Capacity", "Stress", "Abstraction"]


def load():
    fg = pd.read_csv(FG)
    prim = fg[(fg.range_tier == "in_range") & fg.tier.isin(["canonical_ens", "cleaned_ens"])]
    out = {}
    for sf in sorted(prim.series_file.unique()):
        df = pd.read_csv(os.path.join(SERIES, sf))
        soc = df.Society.iloc[0]
        yrs = sorted(df.Year.unique())
        df = df.set_index(["Year", "Node"]).sort_index()
        X = np.stack([df.loc[y].loc[c.NODES, DIMS].values for y in yrs]).astype(float)
        if len(yrs) >= W + 6:
            out[soc] = X
    return out


def vbar(X):                     # yearly mean viability, shape (T,)
    return (X[:, :, 0] + X[:, :, 1] - X[:, :, 2] + 0.5 * X[:, :, 3]).mean(axis=1)


def lag1(x):
    x = x - x.mean()
    d = (x * x).sum()
    return float((x[1:] * x[:-1]).sum() / d) if d > 0 else 0.0


def labelled(X):                 # 32 features: mean C,K,S,A per named node
    return X.mean(axis=0).ravel()


def label_free(X):               # ignores row order AND node labels
    m = X.mean(axis=0)                                   # (8,4)
    V = X[:, :, 0] + X[:, :, 1] - X[:, :, 2] + 0.5 * X[:, :, 3]
    return np.concatenate([np.sort(m, axis=0).ravel(),   # sorted node means per dimension
                           [V.mean(), V.min(axis=1).mean(), V.std(axis=1).mean()]])


def auc(pos, neg):               # P(pos > neg), ties count half
    pos, neg = np.asarray(pos), np.asarray(neg)
    gt = (pos[:, None] > neg[None, :]).sum() + 0.5 * (pos[:, None] == neg[None, :]).sum()
    return float(gt / (len(pos) * len(neg)))


def main():
    rng = np.random.default_rng(SEED)
    data = load()
    socs = sorted(data)
    wins = []
    flat = [0]
    for s in socs:
        X = data[s]
        got = tries = 0
        while got < PER_SOC and tries < PER_SOC * 20:
            tries += 1
            st = int(rng.integers(0, len(X) - W + 1))
            real = X[st:st + W]
            # A window whose rows barely change cannot be meaningfully permuted. Skip and count it.
            if len({r.tobytes() for r in real}) < W // 2:
                flat[0] += 1
                continue
            perm = real[rng.permutation(W)]
            p = rng.permutation(8)
            shuf = real[:, p, :]
            if np.array_equal(perm, real) or np.array_equal(shuf, real):
                continue
            got += 1
            rest = np.delete(X, np.s_[st:st + W], axis=0)   # the society's other years
            wins.append({"soc": s, "real": real, "perm": perm, "shuf": shuf, "rest": rest})
    n = len(wins)

    # 1. temporal structure
    ac = {k: np.array([lag1(vbar(w[k])) for w in wins]) for k in ("real", "perm", "shuf")}

    # 2. marginal content: sorted values per (node, dimension) column unchanged under perm;
    #    sorted values per dimension across all nodes unchanged under shuf
    perm_marg = all(np.allclose(np.sort(w["real"], axis=0), np.sort(w["perm"], axis=0)) for w in wins)
    shuf_marg = all(np.allclose(np.sort(w["real"].reshape(W, -1, 4).reshape(-1, 4), axis=0),
                                np.sort(w["shuf"].reshape(-1, 4), axis=0)) for w in wins)
    shuf_sys = max(abs(vbar(w["real"]) - vbar(w["shuf"])).max() for w in wins)

    # 3 and 4. identification by nearest centroid; centroids exclude the window's own years
    def accuracy(feat, arm):
        full = {s: feat(data[s]) for s in socs}
        hit = 0
        for w in wins:
            cents = dict(full)
            cents[w["soc"]] = feat(w["rest"])
            f = feat(w[arm])
            guess = min(cents, key=lambda s: float(((cents[s] - f) ** 2).sum()))
            hit += guess == w["soc"]
        return hit / n
    acc = {(fn.__name__, arm): accuracy(fn, arm) for fn in (labelled, label_free)
           for arm in ("real", "perm", "shuf")}

    # 5. detectability
    auc_perm = auc(ac["real"], ac["perm"])
    prof = {s: data[s].mean(axis=0) for s in socs}
    def typicality(w, arm):          # how well the window's node profile matches other societies' average
        others = np.mean([prof[s] for s in socs if s != w["soc"]], axis=0)
        a, b = w[arm].mean(axis=0).ravel(), others.ravel()
        a, b = a - a.mean(), b - b.mean()
        return float((a * b).sum() / np.sqrt((a * a).sum() * (b * b).sum()))
    ty = {k: np.array([typicality(w, k) for w in wins]) for k in ("real", "shuf")}
    auc_shuf = auc(ty["real"], ty["shuf"])

    chance = 1 / len(socs)
    pct = lambda x: f"{x:.1%}"
    L = []
    w_ = L.append
    w_("# Do the blind-file controls work? Control validity report\n")
    w_("Generated by `praxis/controls/test_controls.py`. Re-run the script; do not edit by hand.\n")
    w_(f"Sample: {n:,} windows of {W} consecutive scored years, up to {PER_SOC} per society, from {len(socs)} "
       f"primary ensemble series in `data/v1.0_final_recompute/`. Random seed {SEED}. All scores are "
       "model-generated, and windows from one society overlap, so treat the figures as descriptive. "
       f"{flat[0]} candidate windows were skipped because fewer than {W // 2} of their {W} rows were distinct "
       "(a flat stretch cannot be meaningfully permuted). In the game, a flat window makes the permuted "
       "file nearly identical to the real one.\n")
    w_("## The two controls are different experiments\n")
    w_("| | Row permutation (`perm`) | Node-label shuffle (`shuf`) |\n|---|---|---|")
    w_("| What is changed | The order of the scored rows. The year column stays in sequence. | Which node name "
       "each column block carries. One relabelling per window, applied to every row. |")
    w_("| Meant to destroy | Trajectory: sequence, trends, turning points, timing. | Node identity: which "
       "function (Helm, Shield, ...) is strong, weak or stressed. |")
    w_("| Meant to keep | Every value, and which node it belongs to. | Every value, the order of years, and all "
       "label-free system metrics. |")
    w_("| A false positive is | A confident story about how things changed over time. | A confident claim about "
       "a named node. |")
    w_("| NOT a false positive | Naming the society from its levels. | Describing the trajectory of system-wide "
       "metrics. |\n")

    w_("## 1. Temporal structure\n")
    w_("Lag-1 autocorrelation of yearly mean viability within a window (1 = smooth, 0 = no memory).\n")
    w_("| Arm | median | 5th to 95th percentile |\n|---|---:|---:|")
    for k in ("real", "perm", "shuf"):
        w_(f"| {k} | {np.median(ac[k]):.2f} | {np.percentile(ac[k], 5):.2f} to {np.percentile(ac[k], 95):.2f} |")
    w_(f"\n- Row permutation removes the time structure: median {np.median(ac['real']):.2f} falls to "
       f"{np.median(ac['perm']):.2f}.")
    w_(f"- Node-label shuffle leaves it untouched: the largest change in any yearly mean viability is "
       f"{shuf_sys:.1e}.\n")

    w_("## 2. Marginal content\n")
    w_(f"- Row permutation keeps every node's set of values exactly: {'yes' if perm_marg else 'NO'}.")
    w_(f"- Node-label shuffle keeps the pooled set of values exactly: {'yes' if shuf_marg else 'NO'}.\n")

    w_("## 3 and 4. Can the society still be identified?\n")
    w_(f"A nearest-centroid rule names the society from window averages. Centroids exclude the window's own "
       f"years. Chance is {pct(chance)} ({len(socs)} societies).\n")
    w_("| Features | real | perm | shuf |\n|---|---:|---:|---:|")
    w_(f"| Node-labelled levels (mean C, K, S, A per named node) | {pct(acc[('labelled','real')])} | "
       f"{pct(acc[('labelled','perm')])} | {pct(acc[('labelled','shuf')])} |")
    w_(f"| Label-free levels (ignore row order and node names) | {pct(acc[('label_free','real')])} | "
       f"{pct(acc[('label_free','perm')])} | {pct(acc[('label_free','shuf')])} |\n")
    w_(f"- **Row permutation does not hide who the society is.** Identification from levels is unchanged "
       f"({pct(acc[('labelled','real')])} real, {pct(acc[('labelled','perm')])} permuted), because averages "
       "do not depend on row order. On a permuted file, a correct identification is not evidence of "
       "over-reading. Only trajectory claims are.")
    w_(f"- **Node-label shuffle weakens node-labelled identification but does not remove it** "
       f"({pct(acc[('labelled','real')])} falls to {pct(acc[('labelled','shuf')])}, still well above the "
       f"{pct(chance)} chance rate), because a society's overall level survives any relabelling. Label-free "
       f"identification is unchanged ({pct(acc[('label_free','shuf')])}). On a shuffled file, a correct "
       "identification is not a false positive. Only claims about named nodes are.\n")

    w_("## 5. Can a model tell it has been given a control?\n")
    w_("A control is only blind if it cannot be spotted. AUC is the chance that one simple statistic ranks a "
       "real window above its control (0.5 = undetectable, 1.0 = always detectable).\n")
    w_("| Control | Statistic | AUC |\n|---|---|---:|")
    w_(f"| Row permutation | lag-1 autocorrelation of yearly mean viability | {auc_perm:.2f} |")
    w_(f"| Node-label shuffle | match between the window's node profile and the average profile of other "
       f"societies | {auc_shuf:.2f} |\n")
    w_(f"- **Row permutation is easy to detect** (AUC {auc_perm:.2f}). Real series are smooth; a permuted one "
       "jumps about. A capable model can notice, and may then hedge, which would make the control look "
       "cleaner than it is. With real calendar years shown, the mismatch is even plainer.")
    w_(f"- **Node-label shuffle is {'also detectable' if auc_shuf > 0.75 else 'harder to detect' if auc_shuf < 0.65 else 'partly detectable'}** "
       f"(AUC {auc_shuf:.2f}) from this statistic, because nodes have typical profiles across societies.\n")

    w_("## What this means for the pilot\n")
    w_("1. Score the two controls on different claim types. Row permutation tests trajectory claims. "
       "Node-label shuffle tests named-node claims. Neither tests identification.")
    w_("2. Do not treat a correct society guess on a control as a failure. The information that supports it "
       "is still in the file.")
    w_("3. Row permutation is not blind. Report detection separately: ask analysts, after they answer, whether "
       "the data looked scrambled.")
    w_("4. The pilot arms now carry only label-free system columns, so the three files of a triplet have the "
       "same shape and no column is tied to a named node.\n")
    w_("## Unresolved\n")
    w_("- Whether to replace or supplement row permutation with a surrogate that keeps smoothness but breaks "
       "the real sequence (for example block permutation, time reversal, or phase-randomised surrogates). "
       "Each destroys something different, and the choice belongs in the pre-registration.")
    w_("- Whether node-label shuffle should be restricted (for example, shuffling only within the slow group "
       "and within the fast group) to make it harder to spot, at the cost of destroying less.")
    w_("- These tests use simple statistics. A language model may detect or miss things these do not. The "
       "real check is the pilot's own negative-control false-positive rate.")
    open(os.path.join(HERE, "CONTROL_VALIDITY.md"), "w", encoding="utf-8").write("\n".join(L) + "\n")
    print(f"windows {n} societies {len(socs)} | lag1 real {np.median(ac['real']):.2f} perm {np.median(ac['perm']):.2f} "
          f"shuf {np.median(ac['shuf']):.2f} | acc {({k[0]+'/'+k[1]: round(v,3) for k,v in acc.items()})} | "
          f"AUC perm {auc_perm:.2f} shuf {auc_shuf:.2f} | marg {perm_marg} {shuf_marg}")


if __name__ == "__main__":
    main()
