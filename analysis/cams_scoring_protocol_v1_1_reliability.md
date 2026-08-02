# CAMS Scoring Protocol v1.1 — Interim Reliability Validation

**Status: interim.** This documents an empirical improvement check on the CAMS Scoring Protocol
revision (v1.0 → v1.1), not a closed-out validation. Further tuning is expected and already
underway (see §4).

## 1. What changed

`cams_country_raw_scorer.py`'s `COUNTRY_PROMPT` was rewritten (v1.0 → v1.1) to add: a 31-December
time convention, functional-vs-reputational scoring discipline, an evidence-before-score internal
check, cross-year temporal consistency rules, independent-pass integrity instructions, per-node
critical distinctions (Helm, Lore, Stewards, Craft, Hands, Archive, Flow), an evidentiary threshold
for scores of 9–10 or below 5, and an expanded final self-audit. The output CSV schema (`Entity,
Year, Node, Coherence, Capacity, Stress, Abstraction`) was deliberately left unchanged — every
addition is a check the model runs before writing a score, not a new column. The explicit goal was
to reduce divergence between independent scoring passes and between model providers through
instruction precision, not through post-hoc averaging or looser tolerances.

Getting a fair, comparable read across providers also required fixing two pipeline bugs uncovered
during this testing round: an entity-name case-sensitivity bug in ensemble grouping (`"Australia"`
vs `"australia"` were silently treated as different entities), and an Anthropic-specific issue where
long, search-heavy scoring turns were being cut off mid-turn (`stop_reason: "pause_turn"`) and
misread as failed rather than unfinished. Both fixes are in the same commit range as the prompt
revision (`wintermute` commits `72e42e1` through `a512482`).

## 2. Design

18 scoring passes, nested within three model families (providers), on the Australia 2020–2025
panel. Reliability was assessed both within the revised protocol and as an improvement over the
prior (v1.0) protocol, using a paired two-way clustered bootstrap comparing the two prompt versions
on the same panel.

## 3. Results

| Statistic | Value | Interpretation |
|---|---|---|
| ICC(2,k) — average-measure absolute agreement | 0.710 | Good |
| ICC(3,k) — average-measure consistency | 0.885 | Excellent |
| Paired two-way clustered bootstrap, v1.0 → v1.1 | 95% CI excludes zero for both measures | Improvement under the revised instructions is not attributable to chance |

The gap between the two ICC values is itself informative: consistency (ICC(3,k)) being
substantially higher than absolute agreement (ICC(2,k)) indicates that passes and providers track
each other's *relative* movement well — when one run reads a node as more stressed, the others
agree in direction and rough magnitude — while some systematic offset in absolute scale remains
between model families (e.g., one provider running consistently higher or lower than another on the
same panel). That is a normal signature of inter-rater data with a shared-direction, offset-prone
structure, not a sign the revision failed.

## 4. Scope and what this does *not* establish

These results establish scorer reliability and prompt-sensitive construct stabilisation for the
Australia 2020–2025 panel specifically. They do **not** independently establish historical criterion
validity (whether the scores are *correct* readings of Australia's actual institutional history) or
generalisability beyond the tested panel (other countries, other periods, other year ranges).

This is explicitly an interim result. Known further tuning already identified in this same testing
round:

- The Craft node showed a tendency to over-read weak productivity/deindustrialisation signals as
  near-total functional failure in at least one provider's passes; an asymmetric caveat was already
  added to the Craft node definition in response (see the `wintermute` commit history for
  `cams_country_raw_scorer.py`), but this has not yet been re-validated with its own reliability pass.
- Company and city scoring variants use their own separate prompt text and have **not** received the
  v1.1 discipline rewrite or this reliability check at all.
- No cross-country panel has been tested yet — this result is Australia-only.

## 5. Source

Protocol source: `cams_country_raw_scorer.py` (`CAMS_RAW_SCORER_SUITE`, not tracked in this repo).
Shipped local scoring tools: `wintermute/local-scorer/cams-scorer-gui-universal.exe` and
`cams-scorer-universal.exe`.
