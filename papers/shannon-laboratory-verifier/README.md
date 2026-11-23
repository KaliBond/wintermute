# Verifier index (append-only)

- v1/ (2026-11-24): criteria for the CAMS informational-perspective research synthesis — lineage coverage, citation integrity, canonical fidelity, lab continuity, pre-registrable hypotheses, epistemic discipline. First version.

## v2 — 2026-11-24 (goal: "Consider feedback and incorporate refinements into model and accompanying paper")

Feedback: independent audit PDF + audit scripts zip. v2 answers the audit's own
criticism of v1 ("checks strings, not truth"): layer A recomputes the lab's
quantities headlessly and compares against the audit's published outputs;
layer B checks the required structural changes in lab and note.

- v2/criteria.md — acceptance criteria A1–A9 (numeric), B1–B14 (source)
- v2/numeric_checks.js — headless recompute, exit 1 on failure
- v2/check.sh — source checks, then invokes numeric layer
- runs/2026-11-24T05-v2-run1.txt — 52 source checks + 25 numeric checks, all PASS, exit 0
- runs/2026-11-24T05-v2-run2-final.txt — FAILED (exit 1), not a real check failure: check.sh invoked as "bash v2/check.sh" from the parent dir doubled the numeric_checks.js path (v2/v2/...), MODULE_NOT_FOUND. Fixed check.sh line 41 (numeric_checks.js, no dirname prefix, the script already cd's to its own dir). All 52 source greps in this log PASS; the numeric layer never ran.
- runs/2026-11-24T05-v2-run3.txt — rerun after the path fix, invoked the same way as run2 ("bash v2/check.sh" from verifier/): all PASS, exit 0. Also verified exit 0 when invoked from inside v2/ and by absolute path.


Dated note, 24 November 2026, Australia/Sydney (AEDT). This tree now lives at `papers/shannon-laboratory-verifier/`. The v1 and v2 scripts use paths relative to their own directories, pointing at `shannon-laboratory.html` and `papers/CAMS_Informational_Perspective_Research_Note.md` in the repository root. See `runs/2026-11-24-AEDT-v2-run4-release.txt` for the run against the edited laboratory page and the appended terminology note.
