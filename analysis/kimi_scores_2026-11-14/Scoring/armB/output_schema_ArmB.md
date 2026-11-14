# Arm B output schema

One row / object per `blind_id` from `pockets_blind.jsonl`.

## Fields

| Field | Type | Notes |
|-------|------|-------|
| `blind_id` | string | e.g. B001 |
| `entity` | string | Must match the pocket’s entity |
| `chosen_era_id` | string | Roman or other id from `era_shortlist.csv` for that entity **or** leave empty if using years only |
| `chosen_year_start` | int | From shortlist |
| `chosen_year_end` | int | From shortlist |
| `confidence` | enum | immediate | recognisable | broad | miss |
| `brief_reason` | string | One line: which nodes/constraints drove the pick. No historical titles. |

Exactly one of (`chosen_era_id`) or (`chosen_year_start`+`chosen_year_end`) must identify a shortlist row for that entity. Prefer filling **all three** when possible.

## Confidence guide

- **immediate** — topology uniquely points to one shortlist window  
- **recognisable** — clear best match among a few close candidates  
- **broad** — weak discrimination; best guess among several  
- **miss** — cannot choose better than chance; still pick one and mark miss  

## Example JSONL line

```json
{"blind_id":"B001","entity":"Russia","chosen_era_id":"XIV","chosen_year_start":1954,"chosen_year_end":1964,"confidence":"recognisable","brief_reason":"Craft+Lore break forward together while Shield loosens"}
```

## Ledger line (append once)

`model=…; arm=B; entities=CORE[+LIMIT][+EXTENSION]; input_sha=…; output_sha=…; time_aedt=…`
