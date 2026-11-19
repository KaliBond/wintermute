# citizen-observatory

Cloudflare Worker + D1 database behind the CAMS Citizen Scoring Observatory
page (`observatory.html`). Ported on 19 November 2026 from the local FastAPI
prototype `CAMS_Citizen_Observatory_Prototype.zip`.

It serves one of the 105 original CAMS v1.2 scoring prompts (`src/prompts.json`,
byte-for-byte identical to `cams_scoring_v1_2/prompts/`), accepts a
visitor-pasted model reply, checks its structure and stores it for research
review. It never calls a model and never serves `targets.csv`.

## Changes from the prototype

- **Fixed:** the prototype could not store any submission. Its INSERT had 13
  placeholders for 14 columns, so every valid reply raised a server error.
- **Fixed:** the SESSION line may now carry a colon or Markdown emphasis
  (`SESSION: X`, `**SESSION:** X`); the code itself must still match.
- An all-NA reply is accepted but flagged "Every score is NA" for review.
- A consent tick box is required, alongside the fresh-chat attestation.
- Each row stores `prompt_sha256`, `cohort = 'public_v1'`, `review_status`
  (`pending` / `accepted` / `rejected`) and `reviewer_note`, as the prototype
  README recommended.
- Submissions are accepted only from neuralnations.org (plus Cloudflare
  previews and localhost for testing).
- At most 10 submissions per minute per connection (Workers rate-limit
  binding; the IP is used in memory only, never stored).
- Counters say "received", not "accepted", until review.

## Test locally (no Cloudflare login needed)

```
npm install
npm test                                   # validator tests
npx wrangler d1 execute cams-observatory --local --file=schema.sql
npx wrangler dev --port 8787
```

Then serve the repo root (e.g. `python -m http.server 8766`) and open
`http://127.0.0.1:8766/observatory.html?endpoint=http://127.0.0.1:8787`.
The `?endpoint=` override only works on localhost and *.pages.dev.

## Deploy

```
npm install
npx wrangler login
npx wrangler d1 create cams-observatory     # copy the database_id it prints
#   paste that id into wrangler.toml (database_id)
npx wrangler d1 execute cams-observatory --remote --file=schema.sql
npx wrangler deploy                         # prints https://citizen-observatory.<subdomain>.workers.dev
```

Then set `window.OBSERVATORY_CONFIG.endpoint` in `observatory.html` to that URL
and commit. Until then the page shows "The Observatory is not open yet".

## Reviewing submissions

```
npx wrangler d1 execute cams-observatory --remote --command "SELECT submission_id, assignment_id, actual_model, model_version, utc_timestamp, warnings_json, review_status FROM submissions ORDER BY utc_timestamp DESC LIMIT 50"
npx wrangler d1 execute cams-observatory --remote --command "UPDATE submissions SET review_status='accepted', reviewer_note='...' WHERE submission_id='CAMS-...'"
npx wrangler d1 export cams-observatory --remote --output=observatory_export.sql
```

## Before opening to the public

- Approve the "What is stored" text and consent wording on `observatory.html`
  (including whether submitted replies may be published in an open dataset).
- Decide a review process and how public submissions will be analysed (they are
  a separate, volunteer-selected cohort; never pool them into the original
  105-session `scores.csv`).
- Freshness and model identity are self-reported and cannot be verified.
