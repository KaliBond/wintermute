-- D1 schema for the CAMS Citizen Scoring Observatory.
-- Apply with:  npx wrangler d1 execute cams-observatory --remote --file=schema.sql
CREATE TABLE IF NOT EXISTS submissions (
  submission_id      TEXT PRIMARY KEY,          -- receipt ID shown to the visitor
  assignment_id      TEXT NOT NULL,             -- e.g. gpt-2_batch4 (original v1.2 prompt)
  family             TEXT NOT NULL,             -- model family the prompt was written for
  actual_model       TEXT NOT NULL,             -- self-reported by the visitor
  model_version      TEXT NOT NULL,             -- self-reported by the visitor
  utc_timestamp      TEXT NOT NULL,             -- server time
  visitor_local_time TEXT,                      -- optional, unverified
  optional_place     TEXT,                      -- optional city/country, unverified
  attested_fresh     INTEGER NOT NULL,          -- self-attested fresh chat (always 1 when stored)
  consent            INTEGER NOT NULL,          -- consent to storage (always 1 when stored)
  raw_hash           TEXT NOT NULL UNIQUE,      -- SHA-256 of the raw reply; blocks exact duplicates
  prompt_sha256      TEXT NOT NULL,             -- SHA-256 of the prompt that was served
  cohort             TEXT NOT NULL,             -- 'public_v1'; never pool with the controlled cohort
  raw_response       TEXT NOT NULL,
  rows_json          TEXT NOT NULL,             -- the 128 parsed rows
  status             TEXT NOT NULL,             -- 'structurally_valid'
  warnings_json      TEXT NOT NULL,
  review_status      TEXT NOT NULL DEFAULT 'pending',  -- pending | accepted | rejected
  reviewer_note      TEXT
);
CREATE INDEX IF NOT EXISTS idx_submissions_assignment ON submissions (assignment_id);
CREATE INDEX IF NOT EXISTS idx_submissions_family ON submissions (family);
