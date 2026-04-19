/**
 * site-constants.js — CAMS Site Constants
 *
 * Single source of truth for corpus counts and version labels.
 * Update this file when datasets are added or version numbers change.
 * Pages that display counts load this file and may use data-const="KEY"
 * spans to auto-populate values on DOMContentLoaded.
 *
 * Sources
 *   INDEX_SERIES / INDEX_RECORDS  →  cleaned_datasets/cams_index.json
 *   SOCIETIES                     →  38 unique political entities (45 series;
 *                                    France, Germany, China, Australia, USA, Rome,
 *                                    Romania each have 2–3 time-period splits)
 *   VALIDATED_DATASETS            →  datasets.html audit, Stress-column pass
 *   FRAMEWORK_VERSION             →  README.md badge + research-diary.html tags
 *   OPERATORS_EXTENSION           →  cams-explorer.html, cams-interpreter.html
 */
const CAMS_SITE = {

  // ── Corpus (from cams_index.json) ──────────────────────────────────────────
  SOCIETIES:              38,     // Distinct political entities / civilisations
  INDEX_SERIES:           45,     // Total indexed series (some entities split by period)
  INDEX_RECORDS:       39351,     // Node-year records in the current public index
  TIME_SPAN:      "5 CE – 2026",  // Roman series (earliest) to 2026 (latest)
  AI_SCORERS:              3,     // GPT-4o, Grok, Gemini — ensemble averaged

  // ── Validated dataset status (manual audit, August 2025) ───────────────────
  VALIDATED_DATASETS:     28,     // Datasets passing full Stress-column validation
  UNDER_REVIEW:            8,     // Further datasets undergoing quality review

  // ── Model versions ─────────────────────────────────────────────────────────
  FRAMEWORK_VERSION:    "2.3",    // Stable canonical framework — README.md
  OPERATORS_EXTENSION: "v3.2-R", // Research-grade operator extension (ESCH σ,
                                  // κ capacity fraction, headroom, attractor).
                                  // Used in CAMS Explorer and Interpreter.
                                  // This is NOT a new framework version.

  // ── TODO ───────────────────────────────────────────────────────────────────
  // RECORD_COUNT_VALIDATED: The "30,856" figure on earlier pages was computed
  // from the Aug 2025 validated-CSV set (32 files). The current index has
  // 39,351 node-year records across all 45 series. Use INDEX_RECORDS above
  // for current-state copy; retain "30,856" only in historically-dated diary
  // entries where it was the contemporaneously correct count.
};

// Auto-populate data-const="KEY" elements on DOMContentLoaded
document.addEventListener('DOMContentLoaded', () => {
  document.querySelectorAll('[data-const]').forEach(el => {
    const key = el.dataset.const;
    if (CAMS_SITE[key] !== undefined) el.textContent = CAMS_SITE[key];
  });
});
