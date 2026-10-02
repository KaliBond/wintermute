# Backstory — CAMS Through Deep Time

An open project testing whether CAMS can read any human society, from Neolithic villages
to Bronze Age palace states, with the same eight nodes and four axes, and without
borrowing a society's story about itself.

Published through [neuralnations.org](https://neuralnations.org). Author: Kari McKern (ComplexityWorkz).

## Thesis

Human organisation is a coordination system that lets a primate pool skills and extract
thermodynamic surplus. Its shape is set by environment and extraction strategy, not by the
virtue of its members.

- **Cognition** per node = Abstraction × Coherence
- **Energy** per node = Capacity − Stress
- Node Value and Bond Strength from JUNO v1.2 (cams-calc)

## Method

- Rubric: CAMS RAW SCORER v1.2-OPT, evidence-gated (see `../CAMS_SCORING_PROTOCOL_V2_1.md`)
- Five independent scorers per case (camnationsm5n); scorer spread reported
- Thin evidence scored NA, never guessed; NA rate reported per case
- Ancient cases scored in windows of 50–200 years, window stated
- Material proxies first; texts and chronicles only after a first pass

## Open-project commitments

1. Method published before scoring starts.
2. All raw scores, NA flags, spreads and proxy notes released as CSV in `scores/`.
3. Hypotheses timestamped in `HYPOTHESES.md` before the first case is scored.
4. Specialists invited to rescore or contest; disagreements logged.
5. Failed hypotheses published with the same weight as those that hold.

## Layout

| Path | Contents |
| --- | --- |
| `README.md` | This file |
| `HYPOTHESES.md` | Pre-registered hypotheses H1–H5 |
| `cases.csv` | The twelve proposed society-periods |
| `RUNLOG.md` | Every run, decision and deviation |
| `TESTING.md` | How to test against another model |
| `prompts/` | Blind scorer prompt (v1.2-OPT, deep-time adapter DT-1) |
| `runs/` | One folder per scoring pass: raw scores, evidence log, metrics |
| `scripts/` | calc, hypothesis check, model comparison |
| `results/` | Comparison reports |
| `scores/` | Ensemble-mean and envelope CSVs, once five passes exist |
| `portraits/` | "What it felt like" sketches per case (to come) |

## Sequence

1. Foundations essay (CAMS from first principles)
2. Pilot: Old Kingdom + First Intermediate Egypt
3. Main scoring run, remaining ten cases
4. Portraits
5. Synthesis: cognition × energy map, hypothesis results
