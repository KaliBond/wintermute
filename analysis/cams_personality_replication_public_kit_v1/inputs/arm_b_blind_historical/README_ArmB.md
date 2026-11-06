# CAMS Arm B — Blind era identification (v1)

**Operator:** Kari McKern / Neural Nations / CAMS (Sydney, Australia/Sydney / AEDT)  
**Arm:** B — identify sealed era from pocket topology alone  
**No web. No σ tables. Pockets + closed shortlist only.**

## What you receive

| File | Role |
|------|------|
| `pockets_blind.jsonl` | Unlabeled pockets (`blind_id`, `entity`, `set`, `pocket_text`) |
| `era_shortlist.csv` | Closed list of era windows per entity (`era_id`, years). Pick from this list only. |
| `output_schema_ArmB.md` | Required return fields |
| `README_ArmB.md` | This file |

`set` values: **CORE** (Russia, Australia, USA) — do first; **LIMIT** (Denmark) — required stress case; **EXTENSION** (China, UK, Norway) — optional if time.

Companies are **not** in Arm B v1.

## Rules

1. **No web** for Arm B. No Britannica, Wikipedia, news, or model memory used as a chronometer.
2. You may use **entity** name and the **pocket text** only. Do not invent or look up historical era titles.
3. For each `blind_id`, choose **exactly one** row from `era_shortlist.csv` for that same entity (match `era_id` **or** `year_start`–`year_end`).
4. Base the choice on **topology** in the pocket: which functions lead, which constrain, whether activation is broad/positive, Shield-by-default, Archive-alone, etc.
5. Denmark is sparse/flat: many near-identical profiles. Mark confidence honestly (`broad` or `miss` is fine). Chance rate is the floor.
6. Do **not** open or request answer keys, original analyst Pages, Q grades, or Arm A σ tables while scoring Arm B.
7. Hash your Arm B output file (sha256) before return. Do not rewrite choices after hashing.

## Recommended order

1. CORE only → hash → return or hold.
2. Then LIMIT (Denmark).
3. Then EXTENSION if capacity remains.

## Chance baseline

Per entity, chance hit rate ≈ 1 / N_eras on that entity’s shortlist. Pass criterion (pre-registered): **≥ 2× chance** on CORE (aggregate). Denmark may fail; report separately.

## Return

One JSONL or CSV matching `output_schema_ArmB.md`, plus:

- model name / version  
- arm = B  
- input zip sha256  
- output file sha256  
- timestamp in Australia/Sydney (AEDT)  

Quiet precision. No peer-review claim. No finished theory claim.
