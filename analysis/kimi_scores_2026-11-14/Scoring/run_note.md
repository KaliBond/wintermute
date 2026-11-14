# Run note — CAMS Part 2 corporate panel (China + 7 corporates, 2021–2024)

Run date/time: 4 November 2026, approximately 22:40–23:40 AEDT (Australia/Sydney).
Scorer: Kimi Work desktop agent (Kari McKern's local agent runtime), operating parallel fresh-context subagents. One provider error trace during the run identified the backing model as `k2d8-preview`; the agent cannot independently verify the model identifier.
MODE: A (five isolated passes run as fresh subagent contexts; each pass wrote its own CSV).

## How isolation was achieved

- Evidence pre-fetch (eight entities) ran in eight separate fresh subagent contexts on 4 Nov 2026 (China national dossier assembled directly by the orchestrator after two subagent attempts failed on provider errors; see below).
- Each of the five scoring passes ran in a fresh subagent context, launched in parallel, given only: the protocol text (WHAT_YOU_PRODUCE.txt), the eight evidence files, and its own output path. Pass contexts were instructed not to read, list or open any other pass file, run_note.md or MANIFEST.sha256; each pass confirmed compliance in its run summary.
- Pass 2 was interrupted and restarted: its first attempt (agent-126) was stopped by a provider content filter before producing any output (no text written, no file created). The pass was relaunched fresh (agent-130) with a rephrased brief; the restart produced pass2.csv. No partial output from the failed attempt existed to contaminate anything.
- Pass 4 and pass 5 each ran a small number of targeted supplemental web searches of their own (7 and 2 respectively); passes 1–3 ran none.
- Isolation was enforced by fresh contexts plus instruction, not by hardware-level sandboxing. The orchestrator did not show any pass any other pass's output; passes ran concurrently and wrote distinct files.

## Searched vs memory; cutoff rule

- Cutoff rule applied: 30 June of Y+1, inclusive (as confirmed by Kari 4 Nov 2026 21:23 AEDT). Baseline end-2020 established only from material dated on or before 30 June 2021.
- All passes scored from the pre-fetched, date-disciplined evidence corpus (web-searched, dated sources), not from bare memory — EXCEPT: China.md contains items explicitly marked [memory] (date treated as certain, no URL captured; about a dozen, mostly baseline and some NBS full-year releases). These were declared to every pass and used as lower-grade evidence. This is a declared softening of "dated sources only" and is the main limit on date certification (below).

## Post-cutoff exposure and discards

- Orchestrator evidence pre-fetch discarded, per entity discard logs: Evergrande ~7 (incl. AFRC April 2026 PwC sanction, 2026 liquidators' claim, Hui plea, Aug 2025 delisting; 3 borderline items flagged), Vanke 4+ (incl. 36Kr/Caproasia 2025 items, undatable Baidu Baike), Longfor 14 (2025–26 rating actions and retrospectives), Huawei 12 (2025 AR, Caixin 2025-09, aggregators), SMIC 13 (2025 results, undatable aggregators), BYD 13 (Berkshire exit Sep 2025, 2025 recalls/results, undatable compilations), CATL ~6 (rolling/undatable pages, post-window personnel tables), China 5 categories (2026 retrospectives incl. a China Daily 2026-01-28 drills timeline, a Springer 2025 COVID paper, 2026-dated aggregator pages). Full lists are in each evidence file's "Discarded" section.
- Pass-level discards: pass 4 discarded ~25 items from its own searches (undatable compilations, irrelevant documents; no post-cutoff item used); pass 5 discarded 2 (irrelevant/undatable). Passes 1–3 ran no searches.
- Zero-exposure certification: NOT certified. (a) The shared corpus was compiled in Nov 2026 from searches that returned post-cutoff material, all logged as discarded, but discarding was a human/agent judgment at fetch time. (b) [memory]-sourced China items cannot be certified free of post-cutoff knowledge. (c) Model training memory may contain post-cutoff information; passes state they scored only from dated in-corpus items, but this cannot be verified from inside.

## Contamination check

The orchestrator and all five passes confirm: no Section 7 predictions, no Part 1 results, and no prior CAMS ensemble outputs were seen or used anywhere in this run. The panel was scored cold from the protocol text and the dated evidence corpus only.

## NA cells per pass (mechanically validated; all corporate NAs are all-four-or-none; China has no NA in any pass)

- pass1.csv: 64 NA cells (16 node-years)
- pass2.csv: 60 NA cells (15 node-years)
- pass3.csv: 60 NA cells (15 node-years)
- pass4.csv: 32 NA cells (8 node-years)
- pass5.csv: 60 NA cells (15 node-years)

General reason: evidence gate failure, concentrated in Workforce node-years (no dated in-window performance and/or strain indicator), plus a small number of Board/RD/Markets/Brand node-years. Workforce 2021–2022 for BYD is a systematic case (quantified headcount published 2023-07-31, post-cutoff for both years). Pass-level gate strictness varied: pass 4 was the most permissive (32 NA cells), pass 1 the most conservative (64). No commentary on scores is offered here, per protocol.

## Rules not fully followed, with reasons

1. **Shared evidence corpus (main deviation).** The protocol requires each pass to do its own evidence gathering. Operator direction ("fetch data", then "search and run script") led to a single pre-fetched corpus shared by all five passes, with optional per-pass supplemental search. Pass-level judgment remained independent (NA counts and gate outcomes differ materially across passes), but evidence-gathering independence was not achieved. Declared honestly rather than certified around.
2. **[memory] items in China.md** used as declared lower-grade evidence (see above).
3. **China subagent failures.** Two subagent attempts to fetch China's evidence died on provider errors ("400 Unsupported tool result"); the dossier was assembled directly by the orchestrator via its own searches. Consequence: China evidence is single-collector rather than independently collected.
4. **Pass 2 restart** (content filter) as described above.
5. **Supplemental searching** was optional and unevenly used (passes 4–5 only); this asymmetry is declared.
6. All other rules (chronological scoring, no back-revision, cutoff dates, 2+ point move evidence rule, no derived values, 256-row format, canonical ordering) were verified mechanically by validate_passes.py: all five passes conform (256 data rows, canonical header, values 1–10 or NA, all-four-or-none corporate NAs, China NA-free).
