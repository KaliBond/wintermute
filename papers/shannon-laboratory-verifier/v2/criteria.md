# Verifier v2 — "Consider feedback and incorporate refinements into model and paper"

Goal turn: 2026-11-24. Feedback inputs: `How information theory informs CAMS.pdf`
(independent audit of Shannon Lab v2.1 + research note) and
`cams_shannon_audit_scripts.zip` (the audit's Node scripts).

v1 lesson (the audit's own criticism): v1 checks were greps — they verified that
strings were present, not that numbers were true. v2 therefore has two layers:
(A) **numeric checks** that recompute lab quantities headlessly from the lab's own
extracted core and compare against both the audit's published outputs and the
figures quoted in the revised note; (B) **source checks** for the required
structural changes. A check passes only if the number matches within tolerance.

## A. Numeric checks (Node 18+, run from verifier/v2/)

Tolerance: |lab − audit| <= 0.002 bits for seed-42 default-run quantities
(identical estimator code, so differences should be ~0; 0.002 guards float noise).
Seed-robustness figures are checked as sign/share claims, not exact values.

| # | Check | Expected (audit output) |
|---|-------|-------------------------|
| A1 | Default run reproduces: A envMI 0.170, up 0.108, down 0.017, TC 5.455; B envMI 0.248, up 0.156, down 0.011, TC 5.803 | run_default_output.txt |
| A2 | Up-TE muted baseline (Hands+Craft+Flow silenced): 0.082 both gates; surplus A 0.024, B 0.068 (seed-mean, positive 30/30) | controls1/excess |
| A3 | Env-conditioned TE: A up|E 0.042, down|E 0.055 (asymmetry INVERTS in A); B up|E 0.072, down|E 0.041 | controls2 |
| A4 | TC conditioned on environment: 3.733 (A), 3.824 (B); ceiling stated as 7 bits | controls1 |
| A5 | O-information: +3.897 (A), +4.402 (B) — redundancy-dominated | controls1 |
| A6 | Authority sweep 0→1 at gate 0.9: Δ(TC\|E) ≈ +0.161, Δ(envMI Helm) ≈ −0.015 | controls2 |
| A7 | Archive calendar record, periodic env, lead 8, slip 0: value to Helm A 0.218, B 0.085; out-of-phase control strips it | controls3 |
| A8 | Circular-shift null (199 shifts): max null up-TE <= 0.005 both gates; printed floor p >= 0.005 | controls1 E3 |
| A9 | Layer-match chance rate printed: P(match>=6 | random 3-partition) = 0.129 (all 966 partitions; 0.164 for sizes 2,3,3) | controls1 E4 |

## B. Source checks

| # | Check |
|---|-------|
| B1 | Lab dictionary: η_loop row = Library Attractor (BS_Lore × BS_Archive)/S_Hands; no η_loop ↔ TE mapping anywhere |
| B2 | Lab dictionary: Stress row maps to environmental entropy rate / rate of change, not observation noise; noise demoted |
| B3 | Lab dictionary: coherence split (within-node C vs cross-node agreement); V row gains semantic-information analogue (Kolchinsky–Wolpert) |
| B4 | Lab: TE chart shows muted baseline and surplus for every quoted TE; capacity tile labelled as illustrative BSC model |
| B5 | Lab: authority-echo panel exists (Helm envMI + TC\|E across authority slider) |
| B6 | Lab: Archive memory panel (record toggle, cycle toggle, out-of-phase control) |
| B7 | Note: "until now" claim removed; dDIG (`src/ddig_analysis.py`) and ESD (`src/cams_analyzer.py`) cited as pre-existing registers |
| B8 | Note: η_loop used only as Library Attractor; mislabel correction logged |
| B9 | Note: revised findings — common-drive attenuation (24%/48% surplus shares), A's raw asymmetry inverts under conditioning, authority echo (+0.161 bits TC\|E, −0.015 envMI), memory-worth = environmental predictability (0.218 bits clean cycle → 0 near slip rate 0.24) |
| B10 | Note: new references — Ashby 1956; Tishby, Pereira & Bialek 1999; Bialek, Nemenman & Tishby 2001; Kolchinsky & Wolpert 2018; Rosas et al 2019; Flack & Krakauer 2006; Shin et al 2020; Heilmann 2008; Wang & Yang 2021 |
| B11 | Note: project-ledger figures (FSLS, 92 onsets, PC1 71.7%, PAS ruling) marked with provenance as project records, not repo-verifiable |
| B12 | Note: gate-as-regime caveat (Heilmann; Wang & Yang) — gate measures channels, not regime type |
| B13 | Note: hypotheses revised — INFO-2 attenuated, memory/echo/coding-fork tests added, dDIG named as starting estimator with its two required fixes |
| B14 | No em-dashes in note prose (user style contract) |
