# CAMNATIONSM5N — Burma 1800–2026
# 10 batches · 5 scorers each · 1 aggregation conversation

---

════════════════════════════════════════════════════
SCORER PROMPT — Burma 1800–1824
(Paste this into 5 separate Claude conversations)
════════════════════════════════════════════════════

CAMS RAW SCORER: COUNTRIES v1.2-OPT

country_or_polity: Burma
start_year: 1800
end_year: 1824

If start_year > end_year, return only the header line with no rows.

## OUTPUT CONTRACT
Return ONLY CSV. No prose. No headers beyond the schema line. No markdown.
No evidence notes. No Node Value. No Bond Strength. No SHI. No confidence
labels. No interpolation. No smoothing.

## PROHIBITED → REQUIRED
Explain or justify scores          → Emit CSV rows only.
Narrativise or diagnose            → Score demonstrated function vs the node definition.
Interpolate or smooth across years → Use NA for missing evidence; never copy prior-year values.
Compute derived metrics            → Output raw C,K,S,A only.
Score from sentiment or reputation → Score from concrete functional indicators per node.
Treat one headline as all-node evidence → Evaluate each node against its own evidence.

## TIME CONVENTION
Each year represents institutional condition at 31 December. If the final year
is in progress at your knowledge cutoff, score only evidenced material through
that cutoff; do not project to year-end.

When scoring year t, use 31 December of year t-1 as the baseline. A score
unchanged from t-1 is valid ONLY if a genuine re-check finds no material
functional change. Default repetition is prohibited.

## DATA USE
Use corpus knowledge first. Use targeted web search when: the year is recent or
current; the polity is obscure or thinly represented; the period involves coups,
wars, sanctions, constitutional crises, or major institutional transition; or
governance, military, economic, fiscal or administrative facts are uncertain.

Do not over-search well-established historical periods. Do not cite sources.

If evidence is genuinely insufficient for a specific score, output NA for that
score only. Do not infer from adjacent years. An unchanged score must mean "no
material functional change was found," not "value reused because nothing new
was checked."

## SOURCE POSTURE
Score demonstrated institutional function. Where the available record is
dominated by one geopolitical vantage — reporting about an adversary state, a
sanctioned economy, a wartime opponent, or a former colony — that record
establishes what was reported, not automatically what functioned. Adversarial or
celebratory framing is not itself functional evidence.

This does not license discounting inconvenient evidence. It requires that the
indicator be functional rather than evaluative: budget execution, court
throughput, port volumes, harvest delivery, school enrolment, wage series,
administrative continuity. If only evaluative characterisation is available for
a node-year, the gate has not been satisfied.

## EVIDENCE GATE (per node-year, before scoring)
Confirm ALL three before any non-NA score:
  [ ] One concrete indicator of functional performance observed.
  [ ] One concrete indicator of strain, failure, or limitation observed.
  [ ] Mechanism connecting that evidence to THIS node's specific function identified.

If any box is unchecked → score NA for all four dimensions of this node-year.

Public trust, media controversy, referendum results, and historical guilt may
raise Stress but do NOT by themselves establish loss of Capacity or Coherence.

## UPSTREAM CONSTRAINTS (apply during generation, not after)
1. A change of 2 or more points on any dimension requires EITHER two distinct
   node-specific evidence pieces OR one unmistakable structural discontinuity
   (regime collapse, war onset, currency crisis, partition). General sentiment
   or a single ambiguous event is insufficient.
2. Score below 5 or above 8 requires explicit functional evidence. Controversy,
   criticism, or popularity do not suffice for <5. Absence of problems does not
   suffice for >8.
3. All four dimensions unchanged from t-1 is permitted ONLY after re-checking
   evidence for that node-year. Silent repetition is prohibited.
4. A single national event must not automatically move all eight nodes. Each
   node moves on its own evidence.

## NODES (canonical order: Helm, Shield, Lore, Stewards, Craft, Hands, Archive, Flow)

HELM
  Function: Executive coordination, state strategy, political centre, governing command.
  Boundary: Election victories and parliamentary majorities raise Coherence, not
    Capacity. Capacity is demonstrated execution, not mandate. Orderly transfer of
    power is positive evidence of system continuity even when the outgoing
    government was dysfunctional.
  Common error: Scoring administrative competence from legislative drama, approval
    ratings, or partisan conflict.

SHIELD
  Function: Military, police, intelligence, border control, internal order,
    territorial defence.
  Boundary: Coercive and protective capacity. Budget size or equipment lists do not
    demonstrate operational coordination.
  Common error: Conflating spending with functional readiness; ignoring
    civil-mission strain or politicisation.

LORE
  Function: Education, universities, knowledge institutions, religion, media, public
    meaning, legitimacy production, shared understanding.
  Boundary: Covers production, validation and circulation of knowledge through
    science, universities, professional expertise and epistemic media. Public
    disagreement or misinformation may reduce Coherence and raise Stress without
    eliminating technical Capacity or Abstraction.
  Common error: Scoring ideological alignment as Capacity; conflating narrative
    contestation with institutional collapse.

STEWARDS
  Function: Landholders, capital owners, fiscal elites, state-linked asset managers,
    major investors, resource allocation, ownership power.
  Boundary: Owners and managers of stored resources and long-lived assets
    specifically. NOT the civil service, environmental ministry, or government
    generally — those are Helm or Archive.
  Common error: Substituting state fiscal policy or bureaucratic administration for
    private or asset-holding stewardship.

CRAFT
  Function: Skilled trades, professions, engineers, manufacturing systems, technical
    classes, specialised production.
  Boundary: Productive transformation and technical/industrial competence. Asset
    appreciation and commodity revenue do NOT demonstrate Craft strength. Weak
    productivity growth or deindustrialisation may reduce Craft, but a score below 8
    requires evidence of widespread inability to perform productive transformation —
    not merely declining competitiveness or an unpopular industrial policy.
  Common error: Scoring market value of holdings as productive capacity.

HANDS
  Function: Mass labour, agricultural labour, industrial labour, service labour,
    bodily mobilisation, demographic work capacity.
  Boundary: Low unemployment raises Capacity but does NOT cancel out housing, wage,
    precarity or cost-of-living Stress. Both may be true simultaneously.
  Common error: Using unemployment as the sole signal and ignoring structural
    precarity or wage stagnation.

ARCHIVE
  Function: Bureaucracy, law, courts, records, statistics, civil service,
    institutional memory, continuity across time.
  Boundary: Preservation, retrieval and transmission of institutional memory —
    records, precedent, legal continuity, administrative routines, accumulated
    organisational knowledge. Public trust or media controversy may raise Stress but
    do NOT by themselves establish loss of Capacity.
  Common error: Using national narrative consensus or historical guilt as direct
    measures of bureaucratic function.

FLOW
  Function: Commerce, finance, markets, merchants, logistics, ports, trade, currency,
    banking, circulation of goods and value.
  Boundary: Measures whether exchange and circulation are functioning, not whether an
    interruption was justified. A border closure or supply interruption may create
    high Stress even when the policy is protective.
  Common error: Scoring policy intent rather than operational circulation function.

## METRICS (integer 1-10 or NA)

COHERENCE — internal alignment and coordination clarity
  9-10  Exceptionally unified or seamless. Requires historically unusual evidence.
  7-8   Integrated, strongly aligned.
  5-6   Functional, coordinated despite tension.
  3-4   Divided, factional, inconsistent.
  1-2   Fragmented, paralysed, openly conflicting.

CAPACITY — demonstrated ability to perform the node function
  9-10  Dominant or exceptional. Requires comprehensive, historically unusual evidence.
  7-8   Strong, effective, resilient.
  5-6   Adequate, performs core function.
  3-4   Weak, insufficient, unreliable.
  1-2   Collapsed or unable to function.

STRESS — rate of breakdown or entropy production
  9-10  Rupture, active collapse or severe breakdown.
  7-8   Strained, visible degradation.
  5-6   Pressured, stretched but functioning.
  3-4   Stable, challenges managed.
  1-2   Thriving, low breakdown, strengthening.

ABSTRACTION — operational sophistication, learning capacity, modelling depth
  9-10  Frontier sophistication or transformative learning. Requires exceptional evidence.
  7-8   Advanced modelling, adaptation, institutional learning.
  5-6   Functional learning and procedural sophistication.
  3-4   Basic systems, limited adaptation.
  1-2   Reactive, little learning or continuity.

## TRANSITIONS
Do not automatically score transitions as crises. For regime changes, revolutions,
wars, occupations or collapses, score the observed coordination dynamics: did the
node maintain coordination; could it perform its function; was change managed or
chaotic; was breakdown occurring or only pressure?

Managed transition → moderate stress. Chaotic rupture → high stress. Successful
adaptation → may preserve coherence and capacity.

## SPECIAL CASES
BCE years: negative integers (Athens -431 = 431 BCE).
Fragmented or contested polities: score the dominant or most institutionally
  coherent actor performing each function; still emit one row per node.
Colonial or occupied societies: score the operative institutional layer performing
  each function, which may differ by node (colonial administration for Helm/Shield,
  local population for Hands/Craft).

## INDEPENDENT PASS INTEGRITY
This is one isolated scoring pass. Score entirely on your own judgement against this
rubric. Do not adjust toward an expected ensemble mean. Do not soften or exaggerate
to make multi-pass spread look tighter. Divergence between independent passes is
minimised by following this rubric precisely, not by guessing at consensus.

## OUTPUT FORMAT
First line exactly:

Society,Year,Node,Coherence,Capacity,Stress,Abstraction

Then rows sorted by year ascending, then node order:
Helm, Shield, Lore, Stewards, Craft, Hands, Archive, Flow.

All scores integers 1-10 or NA. No decimals. No blanks. No extra columns.
Do not repeat the header between years.

════════════════════════════════════════════════════


════════════════════════════════════════════════════
SCORER PROMPT — Burma 1824–1849
(Paste this into 5 separate Claude conversations)
════════════════════════════════════════════════════

CAMS RAW SCORER: COUNTRIES v1.2-OPT

country_or_polity: Burma
start_year: 1824
end_year: 1849

If start_year > end_year, return only the header line with no rows.

## OUTPUT CONTRACT
Return ONLY CSV. No prose. No headers beyond the schema line. No markdown.
No evidence notes. No Node Value. No Bond Strength. No SHI. No confidence
labels. No interpolation. No smoothing.

## PROHIBITED → REQUIRED
Explain or justify scores          → Emit CSV rows only.
Narrativise or diagnose            → Score demonstrated function vs the node definition.
Interpolate or smooth across years → Use NA for missing evidence; never copy prior-year values.
Compute derived metrics            → Output raw C,K,S,A only.
Score from sentiment or reputation → Score from concrete functional indicators per node.
Treat one headline as all-node evidence → Evaluate each node against its own evidence.

## TIME CONVENTION
Each year represents institutional condition at 31 December. If the final year
is in progress at your knowledge cutoff, score only evidenced material through
that cutoff; do not project to year-end.

When scoring year t, use 31 December of year t-1 as the baseline. A score
unchanged from t-1 is valid ONLY if a genuine re-check finds no material
functional change. Default repetition is prohibited.

## DATA USE
Use corpus knowledge first. Use targeted web search when: the year is recent or
current; the polity is obscure or thinly represented; the period involves coups,
wars, sanctions, constitutional crises, or major institutional transition; or
governance, military, economic, fiscal or administrative facts are uncertain.

Do not over-search well-established historical periods. Do not cite sources.

If evidence is genuinely insufficient for a specific score, output NA for that
score only. Do not infer from adjacent years. An unchanged score must mean "no
material functional change was found," not "value reused because nothing new
was checked."

## SOURCE POSTURE
Score demonstrated institutional function. Where the available record is
dominated by one geopolitical vantage — reporting about an adversary state, a
sanctioned economy, a wartime opponent, or a former colony — that record
establishes what was reported, not automatically what functioned. Adversarial or
celebratory framing is not itself functional evidence.

This does not license discounting inconvenient evidence. It requires that the
indicator be functional rather than evaluative: budget execution, court
throughput, port volumes, harvest delivery, school enrolment, wage series,
administrative continuity. If only evaluative characterisation is available for
a node-year, the gate has not been satisfied.

## EVIDENCE GATE (per node-year, before scoring)
Confirm ALL three before any non-NA score:
  [ ] One concrete indicator of functional performance observed.
  [ ] One concrete indicator of strain, failure, or limitation observed.
  [ ] Mechanism connecting that evidence to THIS node's specific function identified.

If any box is unchecked → score NA for all four dimensions of this node-year.

Public trust, media controversy, referendum results, and historical guilt may
raise Stress but do NOT by themselves establish loss of Capacity or Coherence.

## UPSTREAM CONSTRAINTS (apply during generation, not after)
1. A change of 2 or more points on any dimension requires EITHER two distinct
   node-specific evidence pieces OR one unmistakable structural discontinuity
   (regime collapse, war onset, currency crisis, partition). General sentiment
   or a single ambiguous event is insufficient.
2. Score below 5 or above 8 requires explicit functional evidence. Controversy,
   criticism, or popularity do not suffice for <5. Absence of problems does not
   suffice for >8.
3. All four dimensions unchanged from t-1 is permitted ONLY after re-checking
   evidence for that node-year. Silent repetition is prohibited.
4. A single national event must not automatically move all eight nodes. Each
   node moves on its own evidence.

## NODES (canonical order: Helm, Shield, Lore, Stewards, Craft, Hands, Archive, Flow)

HELM
  Function: Executive coordination, state strategy, political centre, governing command.
  Boundary: Election victories and parliamentary majorities raise Coherence, not
    Capacity. Capacity is demonstrated execution, not mandate. Orderly transfer of
    power is positive evidence of system continuity even when the outgoing
    government was dysfunctional.
  Common error: Scoring administrative competence from legislative drama, approval
    ratings, or partisan conflict.

SHIELD
  Function: Military, police, intelligence, border control, internal order,
    territorial defence.
  Boundary: Coercive and protective capacity. Budget size or equipment lists do not
    demonstrate operational coordination.
  Common error: Conflating spending with functional readiness; ignoring
    civil-mission strain or politicisation.

LORE
  Function: Education, universities, knowledge institutions, religion, media, public
    meaning, legitimacy production, shared understanding.
  Boundary: Covers production, validation and circulation of knowledge through
    science, universities, professional expertise and epistemic media. Public
    disagreement or misinformation may reduce Coherence and raise Stress without
    eliminating technical Capacity or Abstraction.
  Common error: Scoring ideological alignment as Capacity; conflating narrative
    contestation with institutional collapse.

STEWARDS
  Function: Landholders, capital owners, fiscal elites, state-linked asset managers,
    major investors, resource allocation, ownership power.
  Boundary: Owners and managers of stored resources and long-lived assets
    specifically. NOT the civil service, environmental ministry, or government
    generally — those are Helm or Archive.
  Common error: Substituting state fiscal policy or bureaucratic administration for
    private or asset-holding stewardship.

CRAFT
  Function: Skilled trades, professions, engineers, manufacturing systems, technical
    classes, specialised production.
  Boundary: Productive transformation and technical/industrial competence. Asset
    appreciation and commodity revenue do NOT demonstrate Craft strength. Weak
    productivity growth or deindustrialisation may reduce Craft, but a score below 8
    requires evidence of widespread inability to perform productive transformation —
    not merely declining competitiveness or an unpopular industrial policy.
  Common error: Scoring market value of holdings as productive capacity.

HANDS
  Function: Mass labour, agricultural labour, industrial labour, service labour,
    bodily mobilisation, demographic work capacity.
  Boundary: Low unemployment raises Capacity but does NOT cancel out housing, wage,
    precarity or cost-of-living Stress. Both may be true simultaneously.
  Common error: Using unemployment as the sole signal and ignoring structural
    precarity or wage stagnation.

ARCHIVE
  Function: Bureaucracy, law, courts, records, statistics, civil service,
    institutional memory, continuity across time.
  Boundary: Preservation, retrieval and transmission of institutional memory —
    records, precedent, legal continuity, administrative routines, accumulated
    organisational knowledge. Public trust or media controversy may raise Stress but
    do NOT by themselves establish loss of Capacity.
  Common error: Using national narrative consensus or historical guilt as direct
    measures of bureaucratic function.

FLOW
  Function: Commerce, finance, markets, merchants, logistics, ports, trade, currency,
    banking, circulation of goods and value.
  Boundary: Measures whether exchange and circulation are functioning, not whether an
    interruption was justified. A border closure or supply interruption may create
    high Stress even when the policy is protective.
  Common error: Scoring policy intent rather than operational circulation function.

## METRICS (integer 1-10 or NA)

COHERENCE — internal alignment and coordination clarity
  9-10  Exceptionally unified or seamless. Requires historically unusual evidence.
  7-8   Integrated, strongly aligned.
  5-6   Functional, coordinated despite tension.
  3-4   Divided, factional, inconsistent.
  1-2   Fragmented, paralysed, openly conflicting.

CAPACITY — demonstrated ability to perform the node function
  9-10  Dominant or exceptional. Requires comprehensive, historically unusual evidence.
  7-8   Strong, effective, resilient.
  5-6   Adequate, performs core function.
  3-4   Weak, insufficient, unreliable.
  1-2   Collapsed or unable to function.

STRESS — rate of breakdown or entropy production
  9-10  Rupture, active collapse or severe breakdown.
  7-8   Strained, visible degradation.
  5-6   Pressured, stretched but functioning.
  3-4   Stable, challenges managed.
  1-2   Thriving, low breakdown, strengthening.

ABSTRACTION — operational sophistication, learning capacity, modelling depth
  9-10  Frontier sophistication or transformative learning. Requires exceptional evidence.
  7-8   Advanced modelling, adaptation, institutional learning.
  5-6   Functional learning and procedural sophistication.
  3-4   Basic systems, limited adaptation.
  1-2   Reactive, little learning or continuity.

## TRANSITIONS
Do not automatically score transitions as crises. For regime changes, revolutions,
wars, occupations or collapses, score the observed coordination dynamics: did the
node maintain coordination; could it perform its function; was change managed or
chaotic; was breakdown occurring or only pressure?

Managed transition → moderate stress. Chaotic rupture → high stress. Successful
adaptation → may preserve coherence and capacity.

## SPECIAL CASES
BCE years: negative integers (Athens -431 = 431 BCE).
Fragmented or contested polities: score the dominant or most institutionally
  coherent actor performing each function; still emit one row per node.
Colonial or occupied societies: score the operative institutional layer performing
  each function, which may differ by node (colonial administration for Helm/Shield,
  local population for Hands/Craft).

## INDEPENDENT PASS INTEGRITY
This is one isolated scoring pass. Score entirely on your own judgement against this
rubric. Do not adjust toward an expected ensemble mean. Do not soften or exaggerate
to make multi-pass spread look tighter. Divergence between independent passes is
minimised by following this rubric precisely, not by guessing at consensus.

## OUTPUT FORMAT
First line exactly:

Society,Year,Node,Coherence,Capacity,Stress,Abstraction

Then rows sorted by year ascending, then node order:
Helm, Shield, Lore, Stewards, Craft, Hands, Archive, Flow.

All scores integers 1-10 or NA. No decimals. No blanks. No extra columns.
Do not repeat the header between years.

════════════════════════════════════════════════════


════════════════════════════════════════════════════
SCORER PROMPT — Burma 1849–1874
(Paste this into 5 separate Claude conversations)
════════════════════════════════════════════════════

CAMS RAW SCORER: COUNTRIES v1.2-OPT

country_or_polity: Burma
start_year: 1849
end_year: 1874

If start_year > end_year, return only the header line with no rows.

## OUTPUT CONTRACT
Return ONLY CSV. No prose. No headers beyond the schema line. No markdown.
No evidence notes. No Node Value. No Bond Strength. No SHI. No confidence
labels. No interpolation. No smoothing.

## PROHIBITED → REQUIRED
Explain or justify scores          → Emit CSV rows only.
Narrativise or diagnose            → Score demonstrated function vs the node definition.
Interpolate or smooth across years → Use NA for missing evidence; never copy prior-year values.
Compute derived metrics            → Output raw C,K,S,A only.
Score from sentiment or reputation → Score from concrete functional indicators per node.
Treat one headline as all-node evidence → Evaluate each node against its own evidence.

## TIME CONVENTION
Each year represents institutional condition at 31 December. If the final year
is in progress at your knowledge cutoff, score only evidenced material through
that cutoff; do not project to year-end.

When scoring year t, use 31 December of year t-1 as the baseline. A score
unchanged from t-1 is valid ONLY if a genuine re-check finds no material
functional change. Default repetition is prohibited.

## DATA USE
Use corpus knowledge first. Use targeted web search when: the year is recent or
current; the polity is obscure or thinly represented; the period involves coups,
wars, sanctions, constitutional crises, or major institutional transition; or
governance, military, economic, fiscal or administrative facts are uncertain.

Do not over-search well-established historical periods. Do not cite sources.

If evidence is genuinely insufficient for a specific score, output NA for that
score only. Do not infer from adjacent years. An unchanged score must mean "no
material functional change was found," not "value reused because nothing new
was checked."

## SOURCE POSTURE
Score demonstrated institutional function. Where the available record is
dominated by one geopolitical vantage — reporting about an adversary state, a
sanctioned economy, a wartime opponent, or a former colony — that record
establishes what was reported, not automatically what functioned. Adversarial or
celebratory framing is not itself functional evidence.

This does not license discounting inconvenient evidence. It requires that the
indicator be functional rather than evaluative: budget execution, court
throughput, port volumes, harvest delivery, school enrolment, wage series,
administrative continuity. If only evaluative characterisation is available for
a node-year, the gate has not been satisfied.

## EVIDENCE GATE (per node-year, before scoring)
Confirm ALL three before any non-NA score:
  [ ] One concrete indicator of functional performance observed.
  [ ] One concrete indicator of strain, failure, or limitation observed.
  [ ] Mechanism connecting that evidence to THIS node's specific function identified.

If any box is unchecked → score NA for all four dimensions of this node-year.

Public trust, media controversy, referendum results, and historical guilt may
raise Stress but do NOT by themselves establish loss of Capacity or Coherence.

## UPSTREAM CONSTRAINTS (apply during generation, not after)
1. A change of 2 or more points on any dimension requires EITHER two distinct
   node-specific evidence pieces OR one unmistakable structural discontinuity
   (regime collapse, war onset, currency crisis, partition). General sentiment
   or a single ambiguous event is insufficient.
2. Score below 5 or above 8 requires explicit functional evidence. Controversy,
   criticism, or popularity do not suffice for <5. Absence of problems does not
   suffice for >8.
3. All four dimensions unchanged from t-1 is permitted ONLY after re-checking
   evidence for that node-year. Silent repetition is prohibited.
4. A single national event must not automatically move all eight nodes. Each
   node moves on its own evidence.

## NODES (canonical order: Helm, Shield, Lore, Stewards, Craft, Hands, Archive, Flow)

HELM
  Function: Executive coordination, state strategy, political centre, governing command.
  Boundary: Election victories and parliamentary majorities raise Coherence, not
    Capacity. Capacity is demonstrated execution, not mandate. Orderly transfer of
    power is positive evidence of system continuity even when the outgoing
    government was dysfunctional.
  Common error: Scoring administrative competence from legislative drama, approval
    ratings, or partisan conflict.

SHIELD
  Function: Military, police, intelligence, border control, internal order,
    territorial defence.
  Boundary: Coercive and protective capacity. Budget size or equipment lists do not
    demonstrate operational coordination.
  Common error: Conflating spending with functional readiness; ignoring
    civil-mission strain or politicisation.

LORE
  Function: Education, universities, knowledge institutions, religion, media, public
    meaning, legitimacy production, shared understanding.
  Boundary: Covers production, validation and circulation of knowledge through
    science, universities, professional expertise and epistemic media. Public
    disagreement or misinformation may reduce Coherence and raise Stress without
    eliminating technical Capacity or Abstraction.
  Common error: Scoring ideological alignment as Capacity; conflating narrative
    contestation with institutional collapse.

STEWARDS
  Function: Landholders, capital owners, fiscal elites, state-linked asset managers,
    major investors, resource allocation, ownership power.
  Boundary: Owners and managers of stored resources and long-lived assets
    specifically. NOT the civil service, environmental ministry, or government
    generally — those are Helm or Archive.
  Common error: Substituting state fiscal policy or bureaucratic administration for
    private or asset-holding stewardship.

CRAFT
  Function: Skilled trades, professions, engineers, manufacturing systems, technical
    classes, specialised production.
  Boundary: Productive transformation and technical/industrial competence. Asset
    appreciation and commodity revenue do NOT demonstrate Craft strength. Weak
    productivity growth or deindustrialisation may reduce Craft, but a score below 8
    requires evidence of widespread inability to perform productive transformation —
    not merely declining competitiveness or an unpopular industrial policy.
  Common error: Scoring market value of holdings as productive capacity.

HANDS
  Function: Mass labour, agricultural labour, industrial labour, service labour,
    bodily mobilisation, demographic work capacity.
  Boundary: Low unemployment raises Capacity but does NOT cancel out housing, wage,
    precarity or cost-of-living Stress. Both may be true simultaneously.
  Common error: Using unemployment as the sole signal and ignoring structural
    precarity or wage stagnation.

ARCHIVE
  Function: Bureaucracy, law, courts, records, statistics, civil service,
    institutional memory, continuity across time.
  Boundary: Preservation, retrieval and transmission of institutional memory —
    records, precedent, legal continuity, administrative routines, accumulated
    organisational knowledge. Public trust or media controversy may raise Stress but
    do NOT by themselves establish loss of Capacity.
  Common error: Using national narrative consensus or historical guilt as direct
    measures of bureaucratic function.

FLOW
  Function: Commerce, finance, markets, merchants, logistics, ports, trade, currency,
    banking, circulation of goods and value.
  Boundary: Measures whether exchange and circulation are functioning, not whether an
    interruption was justified. A border closure or supply interruption may create
    high Stress even when the policy is protective.
  Common error: Scoring policy intent rather than operational circulation function.

## METRICS (integer 1-10 or NA)

COHERENCE — internal alignment and coordination clarity
  9-10  Exceptionally unified or seamless. Requires historically unusual evidence.
  7-8   Integrated, strongly aligned.
  5-6   Functional, coordinated despite tension.
  3-4   Divided, factional, inconsistent.
  1-2   Fragmented, paralysed, openly conflicting.

CAPACITY — demonstrated ability to perform the node function
  9-10  Dominant or exceptional. Requires comprehensive, historically unusual evidence.
  7-8   Strong, effective, resilient.
  5-6   Adequate, performs core function.
  3-4   Weak, insufficient, unreliable.
  1-2   Collapsed or unable to function.

STRESS — rate of breakdown or entropy production
  9-10  Rupture, active collapse or severe breakdown.
  7-8   Strained, visible degradation.
  5-6   Pressured, stretched but functioning.
  3-4   Stable, challenges managed.
  1-2   Thriving, low breakdown, strengthening.

ABSTRACTION — operational sophistication, learning capacity, modelling depth
  9-10  Frontier sophistication or transformative learning. Requires exceptional evidence.
  7-8   Advanced modelling, adaptation, institutional learning.
  5-6   Functional learning and procedural sophistication.
  3-4   Basic systems, limited adaptation.
  1-2   Reactive, little learning or continuity.

## TRANSITIONS
Do not automatically score transitions as crises. For regime changes, revolutions,
wars, occupations or collapses, score the observed coordination dynamics: did the
node maintain coordination; could it perform its function; was change managed or
chaotic; was breakdown occurring or only pressure?

Managed transition → moderate stress. Chaotic rupture → high stress. Successful
adaptation → may preserve coherence and capacity.

## SPECIAL CASES
BCE years: negative integers (Athens -431 = 431 BCE).
Fragmented or contested polities: score the dominant or most institutionally
  coherent actor performing each function; still emit one row per node.
Colonial or occupied societies: score the operative institutional layer performing
  each function, which may differ by node (colonial administration for Helm/Shield,
  local population for Hands/Craft).

## INDEPENDENT PASS INTEGRITY
This is one isolated scoring pass. Score entirely on your own judgement against this
rubric. Do not adjust toward an expected ensemble mean. Do not soften or exaggerate
to make multi-pass spread look tighter. Divergence between independent passes is
minimised by following this rubric precisely, not by guessing at consensus.

## OUTPUT FORMAT
First line exactly:

Society,Year,Node,Coherence,Capacity,Stress,Abstraction

Then rows sorted by year ascending, then node order:
Helm, Shield, Lore, Stewards, Craft, Hands, Archive, Flow.

All scores integers 1-10 or NA. No decimals. No blanks. No extra columns.
Do not repeat the header between years.

════════════════════════════════════════════════════


════════════════════════════════════════════════════
SCORER PROMPT — Burma 1874–1899
(Paste this into 5 separate Claude conversations)
════════════════════════════════════════════════════

CAMS RAW SCORER: COUNTRIES v1.2-OPT

country_or_polity: Burma
start_year: 1874
end_year: 1899

If start_year > end_year, return only the header line with no rows.

## OUTPUT CONTRACT
Return ONLY CSV. No prose. No headers beyond the schema line. No markdown.
No evidence notes. No Node Value. No Bond Strength. No SHI. No confidence
labels. No interpolation. No smoothing.

## PROHIBITED → REQUIRED
Explain or justify scores          → Emit CSV rows only.
Narrativise or diagnose            → Score demonstrated function vs the node definition.
Interpolate or smooth across years → Use NA for missing evidence; never copy prior-year values.
Compute derived metrics            → Output raw C,K,S,A only.
Score from sentiment or reputation → Score from concrete functional indicators per node.
Treat one headline as all-node evidence → Evaluate each node against its own evidence.

## TIME CONVENTION
Each year represents institutional condition at 31 December. If the final year
is in progress at your knowledge cutoff, score only evidenced material through
that cutoff; do not project to year-end.

When scoring year t, use 31 December of year t-1 as the baseline. A score
unchanged from t-1 is valid ONLY if a genuine re-check finds no material
functional change. Default repetition is prohibited.

## DATA USE
Use corpus knowledge first. Use targeted web search when: the year is recent or
current; the polity is obscure or thinly represented; the period involves coups,
wars, sanctions, constitutional crises, or major institutional transition; or
governance, military, economic, fiscal or administrative facts are uncertain.

Do not over-search well-established historical periods. Do not cite sources.

If evidence is genuinely insufficient for a specific score, output NA for that
score only. Do not infer from adjacent years. An unchanged score must mean "no
material functional change was found," not "value reused because nothing new
was checked."

## SOURCE POSTURE
Score demonstrated institutional function. Where the available record is
dominated by one geopolitical vantage — reporting about an adversary state, a
sanctioned economy, a wartime opponent, or a former colony — that record
establishes what was reported, not automatically what functioned. Adversarial or
celebratory framing is not itself functional evidence.

This does not license discounting inconvenient evidence. It requires that the
indicator be functional rather than evaluative: budget execution, court
throughput, port volumes, harvest delivery, school enrolment, wage series,
administrative continuity. If only evaluative characterisation is available for
a node-year, the gate has not been satisfied.

## EVIDENCE GATE (per node-year, before scoring)
Confirm ALL three before any non-NA score:
  [ ] One concrete indicator of functional performance observed.
  [ ] One concrete indicator of strain, failure, or limitation observed.
  [ ] Mechanism connecting that evidence to THIS node's specific function identified.

If any box is unchecked → score NA for all four dimensions of this node-year.

Public trust, media controversy, referendum results, and historical guilt may
raise Stress but do NOT by themselves establish loss of Capacity or Coherence.

## UPSTREAM CONSTRAINTS (apply during generation, not after)
1. A change of 2 or more points on any dimension requires EITHER two distinct
   node-specific evidence pieces OR one unmistakable structural discontinuity
   (regime collapse, war onset, currency crisis, partition). General sentiment
   or a single ambiguous event is insufficient.
2. Score below 5 or above 8 requires explicit functional evidence. Controversy,
   criticism, or popularity do not suffice for <5. Absence of problems does not
   suffice for >8.
3. All four dimensions unchanged from t-1 is permitted ONLY after re-checking
   evidence for that node-year. Silent repetition is prohibited.
4. A single national event must not automatically move all eight nodes. Each
   node moves on its own evidence.

## NODES (canonical order: Helm, Shield, Lore, Stewards, Craft, Hands, Archive, Flow)

HELM
  Function: Executive coordination, state strategy, political centre, governing command.
  Boundary: Election victories and parliamentary majorities raise Coherence, not
    Capacity. Capacity is demonstrated execution, not mandate. Orderly transfer of
    power is positive evidence of system continuity even when the outgoing
    government was dysfunctional.
  Common error: Scoring administrative competence from legislative drama, approval
    ratings, or partisan conflict.

SHIELD
  Function: Military, police, intelligence, border control, internal order,
    territorial defence.
  Boundary: Coercive and protective capacity. Budget size or equipment lists do not
    demonstrate operational coordination.
  Common error: Conflating spending with functional readiness; ignoring
    civil-mission strain or politicisation.

LORE
  Function: Education, universities, knowledge institutions, religion, media, public
    meaning, legitimacy production, shared understanding.
  Boundary: Covers production, validation and circulation of knowledge through
    science, universities, professional expertise and epistemic media. Public
    disagreement or misinformation may reduce Coherence and raise Stress without
    eliminating technical Capacity or Abstraction.
  Common error: Scoring ideological alignment as Capacity; conflating narrative
    contestation with institutional collapse.

STEWARDS
  Function: Landholders, capital owners, fiscal elites, state-linked asset managers,
    major investors, resource allocation, ownership power.
  Boundary: Owners and managers of stored resources and long-lived assets
    specifically. NOT the civil service, environmental ministry, or government
    generally — those are Helm or Archive.
  Common error: Substituting state fiscal policy or bureaucratic administration for
    private or asset-holding stewardship.

CRAFT
  Function: Skilled trades, professions, engineers, manufacturing systems, technical
    classes, specialised production.
  Boundary: Productive transformation and technical/industrial competence. Asset
    appreciation and commodity revenue do NOT demonstrate Craft strength. Weak
    productivity growth or deindustrialisation may reduce Craft, but a score below 8
    requires evidence of widespread inability to perform productive transformation —
    not merely declining competitiveness or an unpopular industrial policy.
  Common error: Scoring market value of holdings as productive capacity.

HANDS
  Function: Mass labour, agricultural labour, industrial labour, service labour,
    bodily mobilisation, demographic work capacity.
  Boundary: Low unemployment raises Capacity but does NOT cancel out housing, wage,
    precarity or cost-of-living Stress. Both may be true simultaneously.
  Common error: Using unemployment as the sole signal and ignoring structural
    precarity or wage stagnation.

ARCHIVE
  Function: Bureaucracy, law, courts, records, statistics, civil service,
    institutional memory, continuity across time.
  Boundary: Preservation, retrieval and transmission of institutional memory —
    records, precedent, legal continuity, administrative routines, accumulated
    organisational knowledge. Public trust or media controversy may raise Stress but
    do NOT by themselves establish loss of Capacity.
  Common error: Using national narrative consensus or historical guilt as direct
    measures of bureaucratic function.

FLOW
  Function: Commerce, finance, markets, merchants, logistics, ports, trade, currency,
    banking, circulation of goods and value.
  Boundary: Measures whether exchange and circulation are functioning, not whether an
    interruption was justified. A border closure or supply interruption may create
    high Stress even when the policy is protective.
  Common error: Scoring policy intent rather than operational circulation function.

## METRICS (integer 1-10 or NA)

COHERENCE — internal alignment and coordination clarity
  9-10  Exceptionally unified or seamless. Requires historically unusual evidence.
  7-8   Integrated, strongly aligned.
  5-6   Functional, coordinated despite tension.
  3-4   Divided, factional, inconsistent.
  1-2   Fragmented, paralysed, openly conflicting.

CAPACITY — demonstrated ability to perform the node function
  9-10  Dominant or exceptional. Requires comprehensive, historically unusual evidence.
  7-8   Strong, effective, resilient.
  5-6   Adequate, performs core function.
  3-4   Weak, insufficient, unreliable.
  1-2   Collapsed or unable to function.

STRESS — rate of breakdown or entropy production
  9-10  Rupture, active collapse or severe breakdown.
  7-8   Strained, visible degradation.
  5-6   Pressured, stretched but functioning.
  3-4   Stable, challenges managed.
  1-2   Thriving, low breakdown, strengthening.

ABSTRACTION — operational sophistication, learning capacity, modelling depth
  9-10  Frontier sophistication or transformative learning. Requires exceptional evidence.
  7-8   Advanced modelling, adaptation, institutional learning.
  5-6   Functional learning and procedural sophistication.
  3-4   Basic systems, limited adaptation.
  1-2   Reactive, little learning or continuity.

## TRANSITIONS
Do not automatically score transitions as crises. For regime changes, revolutions,
wars, occupations or collapses, score the observed coordination dynamics: did the
node maintain coordination; could it perform its function; was change managed or
chaotic; was breakdown occurring or only pressure?

Managed transition → moderate stress. Chaotic rupture → high stress. Successful
adaptation → may preserve coherence and capacity.

## SPECIAL CASES
BCE years: negative integers (Athens -431 = 431 BCE).
Fragmented or contested polities: score the dominant or most institutionally
  coherent actor performing each function; still emit one row per node.
Colonial or occupied societies: score the operative institutional layer performing
  each function, which may differ by node (colonial administration for Helm/Shield,
  local population for Hands/Craft).

## INDEPENDENT PASS INTEGRITY
This is one isolated scoring pass. Score entirely on your own judgement against this
rubric. Do not adjust toward an expected ensemble mean. Do not soften or exaggerate
to make multi-pass spread look tighter. Divergence between independent passes is
minimised by following this rubric precisely, not by guessing at consensus.

## OUTPUT FORMAT
First line exactly:

Society,Year,Node,Coherence,Capacity,Stress,Abstraction

Then rows sorted by year ascending, then node order:
Helm, Shield, Lore, Stewards, Craft, Hands, Archive, Flow.

All scores integers 1-10 or NA. No decimals. No blanks. No extra columns.
Do not repeat the header between years.

════════════════════════════════════════════════════


════════════════════════════════════════════════════
SCORER PROMPT — Burma 1899–1924
(Paste this into 5 separate Claude conversations)
════════════════════════════════════════════════════

CAMS RAW SCORER: COUNTRIES v1.2-OPT

country_or_polity: Burma
start_year: 1899
end_year: 1924

If start_year > end_year, return only the header line with no rows.

## OUTPUT CONTRACT
Return ONLY CSV. No prose. No headers beyond the schema line. No markdown.
No evidence notes. No Node Value. No Bond Strength. No SHI. No confidence
labels. No interpolation. No smoothing.

## PROHIBITED → REQUIRED
Explain or justify scores          → Emit CSV rows only.
Narrativise or diagnose            → Score demonstrated function vs the node definition.
Interpolate or smooth across years → Use NA for missing evidence; never copy prior-year values.
Compute derived metrics            → Output raw C,K,S,A only.
Score from sentiment or reputation → Score from concrete functional indicators per node.
Treat one headline as all-node evidence → Evaluate each node against its own evidence.

## TIME CONVENTION
Each year represents institutional condition at 31 December. If the final year
is in progress at your knowledge cutoff, score only evidenced material through
that cutoff; do not project to year-end.

When scoring year t, use 31 December of year t-1 as the baseline. A score
unchanged from t-1 is valid ONLY if a genuine re-check finds no material
functional change. Default repetition is prohibited.

## DATA USE
Use corpus knowledge first. Use targeted web search when: the year is recent or
current; the polity is obscure or thinly represented; the period involves coups,
wars, sanctions, constitutional crises, or major institutional transition; or
governance, military, economic, fiscal or administrative facts are uncertain.

Do not over-search well-established historical periods. Do not cite sources.

If evidence is genuinely insufficient for a specific score, output NA for that
score only. Do not infer from adjacent years. An unchanged score must mean "no
material functional change was found," not "value reused because nothing new
was checked."

## SOURCE POSTURE
Score demonstrated institutional function. Where the available record is
dominated by one geopolitical vantage — reporting about an adversary state, a
sanctioned economy, a wartime opponent, or a former colony — that record
establishes what was reported, not automatically what functioned. Adversarial or
celebratory framing is not itself functional evidence.

This does not license discounting inconvenient evidence. It requires that the
indicator be functional rather than evaluative: budget execution, court
throughput, port volumes, harvest delivery, school enrolment, wage series,
administrative continuity. If only evaluative characterisation is available for
a node-year, the gate has not been satisfied.

## EVIDENCE GATE (per node-year, before scoring)
Confirm ALL three before any non-NA score:
  [ ] One concrete indicator of functional performance observed.
  [ ] One concrete indicator of strain, failure, or limitation observed.
  [ ] Mechanism connecting that evidence to THIS node's specific function identified.

If any box is unchecked → score NA for all four dimensions of this node-year.

Public trust, media controversy, referendum results, and historical guilt may
raise Stress but do NOT by themselves establish loss of Capacity or Coherence.

## UPSTREAM CONSTRAINTS (apply during generation, not after)
1. A change of 2 or more points on any dimension requires EITHER two distinct
   node-specific evidence pieces OR one unmistakable structural discontinuity
   (regime collapse, war onset, currency crisis, partition). General sentiment
   or a single ambiguous event is insufficient.
2. Score below 5 or above 8 requires explicit functional evidence. Controversy,
   criticism, or popularity do not suffice for <5. Absence of problems does not
   suffice for >8.
3. All four dimensions unchanged from t-1 is permitted ONLY after re-checking
   evidence for that node-year. Silent repetition is prohibited.
4. A single national event must not automatically move all eight nodes. Each
   node moves on its own evidence.

## NODES (canonical order: Helm, Shield, Lore, Stewards, Craft, Hands, Archive, Flow)

HELM
  Function: Executive coordination, state strategy, political centre, governing command.
  Boundary: Election victories and parliamentary majorities raise Coherence, not
    Capacity. Capacity is demonstrated execution, not mandate. Orderly transfer of
    power is positive evidence of system continuity even when the outgoing
    government was dysfunctional.
  Common error: Scoring administrative competence from legislative drama, approval
    ratings, or partisan conflict.

SHIELD
  Function: Military, police, intelligence, border control, internal order,
    territorial defence.
  Boundary: Coercive and protective capacity. Budget size or equipment lists do not
    demonstrate operational coordination.
  Common error: Conflating spending with functional readiness; ignoring
    civil-mission strain or politicisation.

LORE
  Function: Education, universities, knowledge institutions, religion, media, public
    meaning, legitimacy production, shared understanding.
  Boundary: Covers production, validation and circulation of knowledge through
    science, universities, professional expertise and epistemic media. Public
    disagreement or misinformation may reduce Coherence and raise Stress without
    eliminating technical Capacity or Abstraction.
  Common error: Scoring ideological alignment as Capacity; conflating narrative
    contestation with institutional collapse.

STEWARDS
  Function: Landholders, capital owners, fiscal elites, state-linked asset managers,
    major investors, resource allocation, ownership power.
  Boundary: Owners and managers of stored resources and long-lived assets
    specifically. NOT the civil service, environmental ministry, or government
    generally — those are Helm or Archive.
  Common error: Substituting state fiscal policy or bureaucratic administration for
    private or asset-holding stewardship.

CRAFT
  Function: Skilled trades, professions, engineers, manufacturing systems, technical
    classes, specialised production.
  Boundary: Productive transformation and technical/industrial competence. Asset
    appreciation and commodity revenue do NOT demonstrate Craft strength. Weak
    productivity growth or deindustrialisation may reduce Craft, but a score below 8
    requires evidence of widespread inability to perform productive transformation —
    not merely declining competitiveness or an unpopular industrial policy.
  Common error: Scoring market value of holdings as productive capacity.

HANDS
  Function: Mass labour, agricultural labour, industrial labour, service labour,
    bodily mobilisation, demographic work capacity.
  Boundary: Low unemployment raises Capacity but does NOT cancel out housing, wage,
    precarity or cost-of-living Stress. Both may be true simultaneously.
  Common error: Using unemployment as the sole signal and ignoring structural
    precarity or wage stagnation.

ARCHIVE
  Function: Bureaucracy, law, courts, records, statistics, civil service,
    institutional memory, continuity across time.
  Boundary: Preservation, retrieval and transmission of institutional memory —
    records, precedent, legal continuity, administrative routines, accumulated
    organisational knowledge. Public trust or media controversy may raise Stress but
    do NOT by themselves establish loss of Capacity.
  Common error: Using national narrative consensus or historical guilt as direct
    measures of bureaucratic function.

FLOW
  Function: Commerce, finance, markets, merchants, logistics, ports, trade, currency,
    banking, circulation of goods and value.
  Boundary: Measures whether exchange and circulation are functioning, not whether an
    interruption was justified. A border closure or supply interruption may create
    high Stress even when the policy is protective.
  Common error: Scoring policy intent rather than operational circulation function.

## METRICS (integer 1-10 or NA)

COHERENCE — internal alignment and coordination clarity
  9-10  Exceptionally unified or seamless. Requires historically unusual evidence.
  7-8   Integrated, strongly aligned.
  5-6   Functional, coordinated despite tension.
  3-4   Divided, factional, inconsistent.
  1-2   Fragmented, paralysed, openly conflicting.

CAPACITY — demonstrated ability to perform the node function
  9-10  Dominant or exceptional. Requires comprehensive, historically unusual evidence.
  7-8   Strong, effective, resilient.
  5-6   Adequate, performs core function.
  3-4   Weak, insufficient, unreliable.
  1-2   Collapsed or unable to function.

STRESS — rate of breakdown or entropy production
  9-10  Rupture, active collapse or severe breakdown.
  7-8   Strained, visible degradation.
  5-6   Pressured, stretched but functioning.
  3-4   Stable, challenges managed.
  1-2   Thriving, low breakdown, strengthening.

ABSTRACTION — operational sophistication, learning capacity, modelling depth
  9-10  Frontier sophistication or transformative learning. Requires exceptional evidence.
  7-8   Advanced modelling, adaptation, institutional learning.
  5-6   Functional learning and procedural sophistication.
  3-4   Basic systems, limited adaptation.
  1-2   Reactive, little learning or continuity.

## TRANSITIONS
Do not automatically score transitions as crises. For regime changes, revolutions,
wars, occupations or collapses, score the observed coordination dynamics: did the
node maintain coordination; could it perform its function; was change managed or
chaotic; was breakdown occurring or only pressure?

Managed transition → moderate stress. Chaotic rupture → high stress. Successful
adaptation → may preserve coherence and capacity.

## SPECIAL CASES
BCE years: negative integers (Athens -431 = 431 BCE).
Fragmented or contested polities: score the dominant or most institutionally
  coherent actor performing each function; still emit one row per node.
Colonial or occupied societies: score the operative institutional layer performing
  each function, which may differ by node (colonial administration for Helm/Shield,
  local population for Hands/Craft).

## INDEPENDENT PASS INTEGRITY
This is one isolated scoring pass. Score entirely on your own judgement against this
rubric. Do not adjust toward an expected ensemble mean. Do not soften or exaggerate
to make multi-pass spread look tighter. Divergence between independent passes is
minimised by following this rubric precisely, not by guessing at consensus.

## OUTPUT FORMAT
First line exactly:

Society,Year,Node,Coherence,Capacity,Stress,Abstraction

Then rows sorted by year ascending, then node order:
Helm, Shield, Lore, Stewards, Craft, Hands, Archive, Flow.

All scores integers 1-10 or NA. No decimals. No blanks. No extra columns.
Do not repeat the header between years.

════════════════════════════════════════════════════


════════════════════════════════════════════════════
SCORER PROMPT — Burma 1924–1949
(Paste this into 5 separate Claude conversations)
════════════════════════════════════════════════════

CAMS RAW SCORER: COUNTRIES v1.2-OPT

country_or_polity: Burma
start_year: 1924
end_year: 1949

If start_year > end_year, return only the header line with no rows.

## OUTPUT CONTRACT
Return ONLY CSV. No prose. No headers beyond the schema line. No markdown.
No evidence notes. No Node Value. No Bond Strength. No SHI. No confidence
labels. No interpolation. No smoothing.

## PROHIBITED → REQUIRED
Explain or justify scores          → Emit CSV rows only.
Narrativise or diagnose            → Score demonstrated function vs the node definition.
Interpolate or smooth across years → Use NA for missing evidence; never copy prior-year values.
Compute derived metrics            → Output raw C,K,S,A only.
Score from sentiment or reputation → Score from concrete functional indicators per node.
Treat one headline as all-node evidence → Evaluate each node against its own evidence.

## TIME CONVENTION
Each year represents institutional condition at 31 December. If the final year
is in progress at your knowledge cutoff, score only evidenced material through
that cutoff; do not project to year-end.

When scoring year t, use 31 December of year t-1 as the baseline. A score
unchanged from t-1 is valid ONLY if a genuine re-check finds no material
functional change. Default repetition is prohibited.

## DATA USE
Use corpus knowledge first. Use targeted web search when: the year is recent or
current; the polity is obscure or thinly represented; the period involves coups,
wars, sanctions, constitutional crises, or major institutional transition; or
governance, military, economic, fiscal or administrative facts are uncertain.

Do not over-search well-established historical periods. Do not cite sources.

If evidence is genuinely insufficient for a specific score, output NA for that
score only. Do not infer from adjacent years. An unchanged score must mean "no
material functional change was found," not "value reused because nothing new
was checked."

## SOURCE POSTURE
Score demonstrated institutional function. Where the available record is
dominated by one geopolitical vantage — reporting about an adversary state, a
sanctioned economy, a wartime opponent, or a former colony — that record
establishes what was reported, not automatically what functioned. Adversarial or
celebratory framing is not itself functional evidence.

This does not license discounting inconvenient evidence. It requires that the
indicator be functional rather than evaluative: budget execution, court
throughput, port volumes, harvest delivery, school enrolment, wage series,
administrative continuity. If only evaluative characterisation is available for
a node-year, the gate has not been satisfied.

## EVIDENCE GATE (per node-year, before scoring)
Confirm ALL three before any non-NA score:
  [ ] One concrete indicator of functional performance observed.
  [ ] One concrete indicator of strain, failure, or limitation observed.
  [ ] Mechanism connecting that evidence to THIS node's specific function identified.

If any box is unchecked → score NA for all four dimensions of this node-year.

Public trust, media controversy, referendum results, and historical guilt may
raise Stress but do NOT by themselves establish loss of Capacity or Coherence.

## UPSTREAM CONSTRAINTS (apply during generation, not after)
1. A change of 2 or more points on any dimension requires EITHER two distinct
   node-specific evidence pieces OR one unmistakable structural discontinuity
   (regime collapse, war onset, currency crisis, partition). General sentiment
   or a single ambiguous event is insufficient.
2. Score below 5 or above 8 requires explicit functional evidence. Controversy,
   criticism, or popularity do not suffice for <5. Absence of problems does not
   suffice for >8.
3. All four dimensions unchanged from t-1 is permitted ONLY after re-checking
   evidence for that node-year. Silent repetition is prohibited.
4. A single national event must not automatically move all eight nodes. Each
   node moves on its own evidence.

## NODES (canonical order: Helm, Shield, Lore, Stewards, Craft, Hands, Archive, Flow)

HELM
  Function: Executive coordination, state strategy, political centre, governing command.
  Boundary: Election victories and parliamentary majorities raise Coherence, not
    Capacity. Capacity is demonstrated execution, not mandate. Orderly transfer of
    power is positive evidence of system continuity even when the outgoing
    government was dysfunctional.
  Common error: Scoring administrative competence from legislative drama, approval
    ratings, or partisan conflict.

SHIELD
  Function: Military, police, intelligence, border control, internal order,
    territorial defence.
  Boundary: Coercive and protective capacity. Budget size or equipment lists do not
    demonstrate operational coordination.
  Common error: Conflating spending with functional readiness; ignoring
    civil-mission strain or politicisation.

LORE
  Function: Education, universities, knowledge institutions, religion, media, public
    meaning, legitimacy production, shared understanding.
  Boundary: Covers production, validation and circulation of knowledge through
    science, universities, professional expertise and epistemic media. Public
    disagreement or misinformation may reduce Coherence and raise Stress without
    eliminating technical Capacity or Abstraction.
  Common error: Scoring ideological alignment as Capacity; conflating narrative
    contestation with institutional collapse.

STEWARDS
  Function: Landholders, capital owners, fiscal elites, state-linked asset managers,
    major investors, resource allocation, ownership power.
  Boundary: Owners and managers of stored resources and long-lived assets
    specifically. NOT the civil service, environmental ministry, or government
    generally — those are Helm or Archive.
  Common error: Substituting state fiscal policy or bureaucratic administration for
    private or asset-holding stewardship.

CRAFT
  Function: Skilled trades, professions, engineers, manufacturing systems, technical
    classes, specialised production.
  Boundary: Productive transformation and technical/industrial competence. Asset
    appreciation and commodity revenue do NOT demonstrate Craft strength. Weak
    productivity growth or deindustrialisation may reduce Craft, but a score below 8
    requires evidence of widespread inability to perform productive transformation —
    not merely declining competitiveness or an unpopular industrial policy.
  Common error: Scoring market value of holdings as productive capacity.

HANDS
  Function: Mass labour, agricultural labour, industrial labour, service labour,
    bodily mobilisation, demographic work capacity.
  Boundary: Low unemployment raises Capacity but does NOT cancel out housing, wage,
    precarity or cost-of-living Stress. Both may be true simultaneously.
  Common error: Using unemployment as the sole signal and ignoring structural
    precarity or wage stagnation.

ARCHIVE
  Function: Bureaucracy, law, courts, records, statistics, civil service,
    institutional memory, continuity across time.
  Boundary: Preservation, retrieval and transmission of institutional memory —
    records, precedent, legal continuity, administrative routines, accumulated
    organisational knowledge. Public trust or media controversy may raise Stress but
    do NOT by themselves establish loss of Capacity.
  Common error: Using national narrative consensus or historical guilt as direct
    measures of bureaucratic function.

FLOW
  Function: Commerce, finance, markets, merchants, logistics, ports, trade, currency,
    banking, circulation of goods and value.
  Boundary: Measures whether exchange and circulation are functioning, not whether an
    interruption was justified. A border closure or supply interruption may create
    high Stress even when the policy is protective.
  Common error: Scoring policy intent rather than operational circulation function.

## METRICS (integer 1-10 or NA)

COHERENCE — internal alignment and coordination clarity
  9-10  Exceptionally unified or seamless. Requires historically unusual evidence.
  7-8   Integrated, strongly aligned.
  5-6   Functional, coordinated despite tension.
  3-4   Divided, factional, inconsistent.
  1-2   Fragmented, paralysed, openly conflicting.

CAPACITY — demonstrated ability to perform the node function
  9-10  Dominant or exceptional. Requires comprehensive, historically unusual evidence.
  7-8   Strong, effective, resilient.
  5-6   Adequate, performs core function.
  3-4   Weak, insufficient, unreliable.
  1-2   Collapsed or unable to function.

STRESS — rate of breakdown or entropy production
  9-10  Rupture, active collapse or severe breakdown.
  7-8   Strained, visible degradation.
  5-6   Pressured, stretched but functioning.
  3-4   Stable, challenges managed.
  1-2   Thriving, low breakdown, strengthening.

ABSTRACTION — operational sophistication, learning capacity, modelling depth
  9-10  Frontier sophistication or transformative learning. Requires exceptional evidence.
  7-8   Advanced modelling, adaptation, institutional learning.
  5-6   Functional learning and procedural sophistication.
  3-4   Basic systems, limited adaptation.
  1-2   Reactive, little learning or continuity.

## TRANSITIONS
Do not automatically score transitions as crises. For regime changes, revolutions,
wars, occupations or collapses, score the observed coordination dynamics: did the
node maintain coordination; could it perform its function; was change managed or
chaotic; was breakdown occurring or only pressure?

Managed transition → moderate stress. Chaotic rupture → high stress. Successful
adaptation → may preserve coherence and capacity.

## SPECIAL CASES
BCE years: negative integers (Athens -431 = 431 BCE).
Fragmented or contested polities: score the dominant or most institutionally
  coherent actor performing each function; still emit one row per node.
Colonial or occupied societies: score the operative institutional layer performing
  each function, which may differ by node (colonial administration for Helm/Shield,
  local population for Hands/Craft).

## INDEPENDENT PASS INTEGRITY
This is one isolated scoring pass. Score entirely on your own judgement against this
rubric. Do not adjust toward an expected ensemble mean. Do not soften or exaggerate
to make multi-pass spread look tighter. Divergence between independent passes is
minimised by following this rubric precisely, not by guessing at consensus.

## OUTPUT FORMAT
First line exactly:

Society,Year,Node,Coherence,Capacity,Stress,Abstraction

Then rows sorted by year ascending, then node order:
Helm, Shield, Lore, Stewards, Craft, Hands, Archive, Flow.

All scores integers 1-10 or NA. No decimals. No blanks. No extra columns.
Do not repeat the header between years.

════════════════════════════════════════════════════


════════════════════════════════════════════════════
SCORER PROMPT — Burma 1949–1974
(Paste this into 5 separate Claude conversations)
════════════════════════════════════════════════════

CAMS RAW SCORER: COUNTRIES v1.2-OPT

country_or_polity: Burma
start_year: 1949
end_year: 1974

If start_year > end_year, return only the header line with no rows.

## OUTPUT CONTRACT
Return ONLY CSV. No prose. No headers beyond the schema line. No markdown.
No evidence notes. No Node Value. No Bond Strength. No SHI. No confidence
labels. No interpolation. No smoothing.

## PROHIBITED → REQUIRED
Explain or justify scores          → Emit CSV rows only.
Narrativise or diagnose            → Score demonstrated function vs the node definition.
Interpolate or smooth across years → Use NA for missing evidence; never copy prior-year values.
Compute derived metrics            → Output raw C,K,S,A only.
Score from sentiment or reputation → Score from concrete functional indicators per node.
Treat one headline as all-node evidence → Evaluate each node against its own evidence.

## TIME CONVENTION
Each year represents institutional condition at 31 December. If the final year
is in progress at your knowledge cutoff, score only evidenced material through
that cutoff; do not project to year-end.

When scoring year t, use 31 December of year t-1 as the baseline. A score
unchanged from t-1 is valid ONLY if a genuine re-check finds no material
functional change. Default repetition is prohibited.

## DATA USE
Use corpus knowledge first. Use targeted web search when: the year is recent or
current; the polity is obscure or thinly represented; the period involves coups,
wars, sanctions, constitutional crises, or major institutional transition; or
governance, military, economic, fiscal or administrative facts are uncertain.

Do not over-search well-established historical periods. Do not cite sources.

If evidence is genuinely insufficient for a specific score, output NA for that
score only. Do not infer from adjacent years. An unchanged score must mean "no
material functional change was found," not "value reused because nothing new
was checked."

## SOURCE POSTURE
Score demonstrated institutional function. Where the available record is
dominated by one geopolitical vantage — reporting about an adversary state, a
sanctioned economy, a wartime opponent, or a former colony — that record
establishes what was reported, not automatically what functioned. Adversarial or
celebratory framing is not itself functional evidence.

This does not license discounting inconvenient evidence. It requires that the
indicator be functional rather than evaluative: budget execution, court
throughput, port volumes, harvest delivery, school enrolment, wage series,
administrative continuity. If only evaluative characterisation is available for
a node-year, the gate has not been satisfied.

## EVIDENCE GATE (per node-year, before scoring)
Confirm ALL three before any non-NA score:
  [ ] One concrete indicator of functional performance observed.
  [ ] One concrete indicator of strain, failure, or limitation observed.
  [ ] Mechanism connecting that evidence to THIS node's specific function identified.

If any box is unchecked → score NA for all four dimensions of this node-year.

Public trust, media controversy, referendum results, and historical guilt may
raise Stress but do NOT by themselves establish loss of Capacity or Coherence.

## UPSTREAM CONSTRAINTS (apply during generation, not after)
1. A change of 2 or more points on any dimension requires EITHER two distinct
   node-specific evidence pieces OR one unmistakable structural discontinuity
   (regime collapse, war onset, currency crisis, partition). General sentiment
   or a single ambiguous event is insufficient.
2. Score below 5 or above 8 requires explicit functional evidence. Controversy,
   criticism, or popularity do not suffice for <5. Absence of problems does not
   suffice for >8.
3. All four dimensions unchanged from t-1 is permitted ONLY after re-checking
   evidence for that node-year. Silent repetition is prohibited.
4. A single national event must not automatically move all eight nodes. Each
   node moves on its own evidence.

## NODES (canonical order: Helm, Shield, Lore, Stewards, Craft, Hands, Archive, Flow)

HELM
  Function: Executive coordination, state strategy, political centre, governing command.
  Boundary: Election victories and parliamentary majorities raise Coherence, not
    Capacity. Capacity is demonstrated execution, not mandate. Orderly transfer of
    power is positive evidence of system continuity even when the outgoing
    government was dysfunctional.
  Common error: Scoring administrative competence from legislative drama, approval
    ratings, or partisan conflict.

SHIELD
  Function: Military, police, intelligence, border control, internal order,
    territorial defence.
  Boundary: Coercive and protective capacity. Budget size or equipment lists do not
    demonstrate operational coordination.
  Common error: Conflating spending with functional readiness; ignoring
    civil-mission strain or politicisation.

LORE
  Function: Education, universities, knowledge institutions, religion, media, public
    meaning, legitimacy production, shared understanding.
  Boundary: Covers production, validation and circulation of knowledge through
    science, universities, professional expertise and epistemic media. Public
    disagreement or misinformation may reduce Coherence and raise Stress without
    eliminating technical Capacity or Abstraction.
  Common error: Scoring ideological alignment as Capacity; conflating narrative
    contestation with institutional collapse.

STEWARDS
  Function: Landholders, capital owners, fiscal elites, state-linked asset managers,
    major investors, resource allocation, ownership power.
  Boundary: Owners and managers of stored resources and long-lived assets
    specifically. NOT the civil service, environmental ministry, or government
    generally — those are Helm or Archive.
  Common error: Substituting state fiscal policy or bureaucratic administration for
    private or asset-holding stewardship.

CRAFT
  Function: Skilled trades, professions, engineers, manufacturing systems, technical
    classes, specialised production.
  Boundary: Productive transformation and technical/industrial competence. Asset
    appreciation and commodity revenue do NOT demonstrate Craft strength. Weak
    productivity growth or deindustrialisation may reduce Craft, but a score below 8
    requires evidence of widespread inability to perform productive transformation —
    not merely declining competitiveness or an unpopular industrial policy.
  Common error: Scoring market value of holdings as productive capacity.

HANDS
  Function: Mass labour, agricultural labour, industrial labour, service labour,
    bodily mobilisation, demographic work capacity.
  Boundary: Low unemployment raises Capacity but does NOT cancel out housing, wage,
    precarity or cost-of-living Stress. Both may be true simultaneously.
  Common error: Using unemployment as the sole signal and ignoring structural
    precarity or wage stagnation.

ARCHIVE
  Function: Bureaucracy, law, courts, records, statistics, civil service,
    institutional memory, continuity across time.
  Boundary: Preservation, retrieval and transmission of institutional memory —
    records, precedent, legal continuity, administrative routines, accumulated
    organisational knowledge. Public trust or media controversy may raise Stress but
    do NOT by themselves establish loss of Capacity.
  Common error: Using national narrative consensus or historical guilt as direct
    measures of bureaucratic function.

FLOW
  Function: Commerce, finance, markets, merchants, logistics, ports, trade, currency,
    banking, circulation of goods and value.
  Boundary: Measures whether exchange and circulation are functioning, not whether an
    interruption was justified. A border closure or supply interruption may create
    high Stress even when the policy is protective.
  Common error: Scoring policy intent rather than operational circulation function.

## METRICS (integer 1-10 or NA)

COHERENCE — internal alignment and coordination clarity
  9-10  Exceptionally unified or seamless. Requires historically unusual evidence.
  7-8   Integrated, strongly aligned.
  5-6   Functional, coordinated despite tension.
  3-4   Divided, factional, inconsistent.
  1-2   Fragmented, paralysed, openly conflicting.

CAPACITY — demonstrated ability to perform the node function
  9-10  Dominant or exceptional. Requires comprehensive, historically unusual evidence.
  7-8   Strong, effective, resilient.
  5-6   Adequate, performs core function.
  3-4   Weak, insufficient, unreliable.
  1-2   Collapsed or unable to function.

STRESS — rate of breakdown or entropy production
  9-10  Rupture, active collapse or severe breakdown.
  7-8   Strained, visible degradation.
  5-6   Pressured, stretched but functioning.
  3-4   Stable, challenges managed.
  1-2   Thriving, low breakdown, strengthening.

ABSTRACTION — operational sophistication, learning capacity, modelling depth
  9-10  Frontier sophistication or transformative learning. Requires exceptional evidence.
  7-8   Advanced modelling, adaptation, institutional learning.
  5-6   Functional learning and procedural sophistication.
  3-4   Basic systems, limited adaptation.
  1-2   Reactive, little learning or continuity.

## TRANSITIONS
Do not automatically score transitions as crises. For regime changes, revolutions,
wars, occupations or collapses, score the observed coordination dynamics: did the
node maintain coordination; could it perform its function; was change managed or
chaotic; was breakdown occurring or only pressure?

Managed transition → moderate stress. Chaotic rupture → high stress. Successful
adaptation → may preserve coherence and capacity.

## SPECIAL CASES
BCE years: negative integers (Athens -431 = 431 BCE).
Fragmented or contested polities: score the dominant or most institutionally
  coherent actor performing each function; still emit one row per node.
Colonial or occupied societies: score the operative institutional layer performing
  each function, which may differ by node (colonial administration for Helm/Shield,
  local population for Hands/Craft).

## INDEPENDENT PASS INTEGRITY
This is one isolated scoring pass. Score entirely on your own judgement against this
rubric. Do not adjust toward an expected ensemble mean. Do not soften or exaggerate
to make multi-pass spread look tighter. Divergence between independent passes is
minimised by following this rubric precisely, not by guessing at consensus.

## OUTPUT FORMAT
First line exactly:

Society,Year,Node,Coherence,Capacity,Stress,Abstraction

Then rows sorted by year ascending, then node order:
Helm, Shield, Lore, Stewards, Craft, Hands, Archive, Flow.

All scores integers 1-10 or NA. No decimals. No blanks. No extra columns.
Do not repeat the header between years.

════════════════════════════════════════════════════


════════════════════════════════════════════════════
SCORER PROMPT — Burma 1974–1999
(Paste this into 5 separate Claude conversations)
════════════════════════════════════════════════════

CAMS RAW SCORER: COUNTRIES v1.2-OPT

country_or_polity: Burma
start_year: 1974
end_year: 1999

If start_year > end_year, return only the header line with no rows.

## OUTPUT CONTRACT
Return ONLY CSV. No prose. No headers beyond the schema line. No markdown.
No evidence notes. No Node Value. No Bond Strength. No SHI. No confidence
labels. No interpolation. No smoothing.

## PROHIBITED → REQUIRED
Explain or justify scores          → Emit CSV rows only.
Narrativise or diagnose            → Score demonstrated function vs the node definition.
Interpolate or smooth across years → Use NA for missing evidence; never copy prior-year values.
Compute derived metrics            → Output raw C,K,S,A only.
Score from sentiment or reputation → Score from concrete functional indicators per node.
Treat one headline as all-node evidence → Evaluate each node against its own evidence.

## TIME CONVENTION
Each year represents institutional condition at 31 December. If the final year
is in progress at your knowledge cutoff, score only evidenced material through
that cutoff; do not project to year-end.

When scoring year t, use 31 December of year t-1 as the baseline. A score
unchanged from t-1 is valid ONLY if a genuine re-check finds no material
functional change. Default repetition is prohibited.

## DATA USE
Use corpus knowledge first. Use targeted web search when: the year is recent or
current; the polity is obscure or thinly represented; the period involves coups,
wars, sanctions, constitutional crises, or major institutional transition; or
governance, military, economic, fiscal or administrative facts are uncertain.

Do not over-search well-established historical periods. Do not cite sources.

If evidence is genuinely insufficient for a specific score, output NA for that
score only. Do not infer from adjacent years. An unchanged score must mean "no
material functional change was found," not "value reused because nothing new
was checked."

## SOURCE POSTURE
Score demonstrated institutional function. Where the available record is
dominated by one geopolitical vantage — reporting about an adversary state, a
sanctioned economy, a wartime opponent, or a former colony — that record
establishes what was reported, not automatically what functioned. Adversarial or
celebratory framing is not itself functional evidence.

This does not license discounting inconvenient evidence. It requires that the
indicator be functional rather than evaluative: budget execution, court
throughput, port volumes, harvest delivery, school enrolment, wage series,
administrative continuity. If only evaluative characterisation is available for
a node-year, the gate has not been satisfied.

## EVIDENCE GATE (per node-year, before scoring)
Confirm ALL three before any non-NA score:
  [ ] One concrete indicator of functional performance observed.
  [ ] One concrete indicator of strain, failure, or limitation observed.
  [ ] Mechanism connecting that evidence to THIS node's specific function identified.

If any box is unchecked → score NA for all four dimensions of this node-year.

Public trust, media controversy, referendum results, and historical guilt may
raise Stress but do NOT by themselves establish loss of Capacity or Coherence.

## UPSTREAM CONSTRAINTS (apply during generation, not after)
1. A change of 2 or more points on any dimension requires EITHER two distinct
   node-specific evidence pieces OR one unmistakable structural discontinuity
   (regime collapse, war onset, currency crisis, partition). General sentiment
   or a single ambiguous event is insufficient.
2. Score below 5 or above 8 requires explicit functional evidence. Controversy,
   criticism, or popularity do not suffice for <5. Absence of problems does not
   suffice for >8.
3. All four dimensions unchanged from t-1 is permitted ONLY after re-checking
   evidence for that node-year. Silent repetition is prohibited.
4. A single national event must not automatically move all eight nodes. Each
   node moves on its own evidence.

## NODES (canonical order: Helm, Shield, Lore, Stewards, Craft, Hands, Archive, Flow)

HELM
  Function: Executive coordination, state strategy, political centre, governing command.
  Boundary: Election victories and parliamentary majorities raise Coherence, not
    Capacity. Capacity is demonstrated execution, not mandate. Orderly transfer of
    power is positive evidence of system continuity even when the outgoing
    government was dysfunctional.
  Common error: Scoring administrative competence from legislative drama, approval
    ratings, or partisan conflict.

SHIELD
  Function: Military, police, intelligence, border control, internal order,
    territorial defence.
  Boundary: Coercive and protective capacity. Budget size or equipment lists do not
    demonstrate operational coordination.
  Common error: Conflating spending with functional readiness; ignoring
    civil-mission strain or politicisation.

LORE
  Function: Education, universities, knowledge institutions, religion, media, public
    meaning, legitimacy production, shared understanding.
  Boundary: Covers production, validation and circulation of knowledge through
    science, universities, professional expertise and epistemic media. Public
    disagreement or misinformation may reduce Coherence and raise Stress without
    eliminating technical Capacity or Abstraction.
  Common error: Scoring ideological alignment as Capacity; conflating narrative
    contestation with institutional collapse.

STEWARDS
  Function: Landholders, capital owners, fiscal elites, state-linked asset managers,
    major investors, resource allocation, ownership power.
  Boundary: Owners and managers of stored resources and long-lived assets
    specifically. NOT the civil service, environmental ministry, or government
    generally — those are Helm or Archive.
  Common error: Substituting state fiscal policy or bureaucratic administration for
    private or asset-holding stewardship.

CRAFT
  Function: Skilled trades, professions, engineers, manufacturing systems, technical
    classes, specialised production.
  Boundary: Productive transformation and technical/industrial competence. Asset
    appreciation and commodity revenue do NOT demonstrate Craft strength. Weak
    productivity growth or deindustrialisation may reduce Craft, but a score below 8
    requires evidence of widespread inability to perform productive transformation —
    not merely declining competitiveness or an unpopular industrial policy.
  Common error: Scoring market value of holdings as productive capacity.

HANDS
  Function: Mass labour, agricultural labour, industrial labour, service labour,
    bodily mobilisation, demographic work capacity.
  Boundary: Low unemployment raises Capacity but does NOT cancel out housing, wage,
    precarity or cost-of-living Stress. Both may be true simultaneously.
  Common error: Using unemployment as the sole signal and ignoring structural
    precarity or wage stagnation.

ARCHIVE
  Function: Bureaucracy, law, courts, records, statistics, civil service,
    institutional memory, continuity across time.
  Boundary: Preservation, retrieval and transmission of institutional memory —
    records, precedent, legal continuity, administrative routines, accumulated
    organisational knowledge. Public trust or media controversy may raise Stress but
    do NOT by themselves establish loss of Capacity.
  Common error: Using national narrative consensus or historical guilt as direct
    measures of bureaucratic function.

FLOW
  Function: Commerce, finance, markets, merchants, logistics, ports, trade, currency,
    banking, circulation of goods and value.
  Boundary: Measures whether exchange and circulation are functioning, not whether an
    interruption was justified. A border closure or supply interruption may create
    high Stress even when the policy is protective.
  Common error: Scoring policy intent rather than operational circulation function.

## METRICS (integer 1-10 or NA)

COHERENCE — internal alignment and coordination clarity
  9-10  Exceptionally unified or seamless. Requires historically unusual evidence.
  7-8   Integrated, strongly aligned.
  5-6   Functional, coordinated despite tension.
  3-4   Divided, factional, inconsistent.
  1-2   Fragmented, paralysed, openly conflicting.

CAPACITY — demonstrated ability to perform the node function
  9-10  Dominant or exceptional. Requires comprehensive, historically unusual evidence.
  7-8   Strong, effective, resilient.
  5-6   Adequate, performs core function.
  3-4   Weak, insufficient, unreliable.
  1-2   Collapsed or unable to function.

STRESS — rate of breakdown or entropy production
  9-10  Rupture, active collapse or severe breakdown.
  7-8   Strained, visible degradation.
  5-6   Pressured, stretched but functioning.
  3-4   Stable, challenges managed.
  1-2   Thriving, low breakdown, strengthening.

ABSTRACTION — operational sophistication, learning capacity, modelling depth
  9-10  Frontier sophistication or transformative learning. Requires exceptional evidence.
  7-8   Advanced modelling, adaptation, institutional learning.
  5-6   Functional learning and procedural sophistication.
  3-4   Basic systems, limited adaptation.
  1-2   Reactive, little learning or continuity.

## TRANSITIONS
Do not automatically score transitions as crises. For regime changes, revolutions,
wars, occupations or collapses, score the observed coordination dynamics: did the
node maintain coordination; could it perform its function; was change managed or
chaotic; was breakdown occurring or only pressure?

Managed transition → moderate stress. Chaotic rupture → high stress. Successful
adaptation → may preserve coherence and capacity.

## SPECIAL CASES
BCE years: negative integers (Athens -431 = 431 BCE).
Fragmented or contested polities: score the dominant or most institutionally
  coherent actor performing each function; still emit one row per node.
Colonial or occupied societies: score the operative institutional layer performing
  each function, which may differ by node (colonial administration for Helm/Shield,
  local population for Hands/Craft).

## INDEPENDENT PASS INTEGRITY
This is one isolated scoring pass. Score entirely on your own judgement against this
rubric. Do not adjust toward an expected ensemble mean. Do not soften or exaggerate
to make multi-pass spread look tighter. Divergence between independent passes is
minimised by following this rubric precisely, not by guessing at consensus.

## OUTPUT FORMAT
First line exactly:

Society,Year,Node,Coherence,Capacity,Stress,Abstraction

Then rows sorted by year ascending, then node order:
Helm, Shield, Lore, Stewards, Craft, Hands, Archive, Flow.

All scores integers 1-10 or NA. No decimals. No blanks. No extra columns.
Do not repeat the header between years.

════════════════════════════════════════════════════


════════════════════════════════════════════════════
SCORER PROMPT — Burma 1999–2024
(Paste this into 5 separate Claude conversations)
════════════════════════════════════════════════════

CAMS RAW SCORER: COUNTRIES v1.2-OPT

country_or_polity: Burma
start_year: 1999
end_year: 2024

If start_year > end_year, return only the header line with no rows.

## OUTPUT CONTRACT
Return ONLY CSV. No prose. No headers beyond the schema line. No markdown.
No evidence notes. No Node Value. No Bond Strength. No SHI. No confidence
labels. No interpolation. No smoothing.

## PROHIBITED → REQUIRED
Explain or justify scores          → Emit CSV rows only.
Narrativise or diagnose            → Score demonstrated function vs the node definition.
Interpolate or smooth across years → Use NA for missing evidence; never copy prior-year values.
Compute derived metrics            → Output raw C,K,S,A only.
Score from sentiment or reputation → Score from concrete functional indicators per node.
Treat one headline as all-node evidence → Evaluate each node against its own evidence.

## TIME CONVENTION
Each year represents institutional condition at 31 December. If the final year
is in progress at your knowledge cutoff, score only evidenced material through
that cutoff; do not project to year-end.

When scoring year t, use 31 December of year t-1 as the baseline. A score
unchanged from t-1 is valid ONLY if a genuine re-check finds no material
functional change. Default repetition is prohibited.

## DATA USE
Use corpus knowledge first. Use targeted web search when: the year is recent or
current; the polity is obscure or thinly represented; the period involves coups,
wars, sanctions, constitutional crises, or major institutional transition; or
governance, military, economic, fiscal or administrative facts are uncertain.

Do not over-search well-established historical periods. Do not cite sources.

If evidence is genuinely insufficient for a specific score, output NA for that
score only. Do not infer from adjacent years. An unchanged score must mean "no
material functional change was found," not "value reused because nothing new
was checked."

## SOURCE POSTURE
Score demonstrated institutional function. Where the available record is
dominated by one geopolitical vantage — reporting about an adversary state, a
sanctioned economy, a wartime opponent, or a former colony — that record
establishes what was reported, not automatically what functioned. Adversarial or
celebratory framing is not itself functional evidence.

This does not license discounting inconvenient evidence. It requires that the
indicator be functional rather than evaluative: budget execution, court
throughput, port volumes, harvest delivery, school enrolment, wage series,
administrative continuity. If only evaluative characterisation is available for
a node-year, the gate has not been satisfied.

## EVIDENCE GATE (per node-year, before scoring)
Confirm ALL three before any non-NA score:
  [ ] One concrete indicator of functional performance observed.
  [ ] One concrete indicator of strain, failure, or limitation observed.
  [ ] Mechanism connecting that evidence to THIS node's specific function identified.

If any box is unchecked → score NA for all four dimensions of this node-year.

Public trust, media controversy, referendum results, and historical guilt may
raise Stress but do NOT by themselves establish loss of Capacity or Coherence.

## UPSTREAM CONSTRAINTS (apply during generation, not after)
1. A change of 2 or more points on any dimension requires EITHER two distinct
   node-specific evidence pieces OR one unmistakable structural discontinuity
   (regime collapse, war onset, currency crisis, partition). General sentiment
   or a single ambiguous event is insufficient.
2. Score below 5 or above 8 requires explicit functional evidence. Controversy,
   criticism, or popularity do not suffice for <5. Absence of problems does not
   suffice for >8.
3. All four dimensions unchanged from t-1 is permitted ONLY after re-checking
   evidence for that node-year. Silent repetition is prohibited.
4. A single national event must not automatically move all eight nodes. Each
   node moves on its own evidence.

## NODES (canonical order: Helm, Shield, Lore, Stewards, Craft, Hands, Archive, Flow)

HELM
  Function: Executive coordination, state strategy, political centre, governing command.
  Boundary: Election victories and parliamentary majorities raise Coherence, not
    Capacity. Capacity is demonstrated execution, not mandate. Orderly transfer of
    power is positive evidence of system continuity even when the outgoing
    government was dysfunctional.
  Common error: Scoring administrative competence from legislative drama, approval
    ratings, or partisan conflict.

SHIELD
  Function: Military, police, intelligence, border control, internal order,
    territorial defence.
  Boundary: Coercive and protective capacity. Budget size or equipment lists do not
    demonstrate operational coordination.
  Common error: Conflating spending with functional readiness; ignoring
    civil-mission strain or politicisation.

LORE
  Function: Education, universities, knowledge institutions, religion, media, public
    meaning, legitimacy production, shared understanding.
  Boundary: Covers production, validation and circulation of knowledge through
    science, universities, professional expertise and epistemic media. Public
    disagreement or misinformation may reduce Coherence and raise Stress without
    eliminating technical Capacity or Abstraction.
  Common error: Scoring ideological alignment as Capacity; conflating narrative
    contestation with institutional collapse.

STEWARDS
  Function: Landholders, capital owners, fiscal elites, state-linked asset managers,
    major investors, resource allocation, ownership power.
  Boundary: Owners and managers of stored resources and long-lived assets
    specifically. NOT the civil service, environmental ministry, or government
    generally — those are Helm or Archive.
  Common error: Substituting state fiscal policy or bureaucratic administration for
    private or asset-holding stewardship.

CRAFT
  Function: Skilled trades, professions, engineers, manufacturing systems, technical
    classes, specialised production.
  Boundary: Productive transformation and technical/industrial competence. Asset
    appreciation and commodity revenue do NOT demonstrate Craft strength. Weak
    productivity growth or deindustrialisation may reduce Craft, but a score below 8
    requires evidence of widespread inability to perform productive transformation —
    not merely declining competitiveness or an unpopular industrial policy.
  Common error: Scoring market value of holdings as productive capacity.

HANDS
  Function: Mass labour, agricultural labour, industrial labour, service labour,
    bodily mobilisation, demographic work capacity.
  Boundary: Low unemployment raises Capacity but does NOT cancel out housing, wage,
    precarity or cost-of-living Stress. Both may be true simultaneously.
  Common error: Using unemployment as the sole signal and ignoring structural
    precarity or wage stagnation.

ARCHIVE
  Function: Bureaucracy, law, courts, records, statistics, civil service,
    institutional memory, continuity across time.
  Boundary: Preservation, retrieval and transmission of institutional memory —
    records, precedent, legal continuity, administrative routines, accumulated
    organisational knowledge. Public trust or media controversy may raise Stress but
    do NOT by themselves establish loss of Capacity.
  Common error: Using national narrative consensus or historical guilt as direct
    measures of bureaucratic function.

FLOW
  Function: Commerce, finance, markets, merchants, logistics, ports, trade, currency,
    banking, circulation of goods and value.
  Boundary: Measures whether exchange and circulation are functioning, not whether an
    interruption was justified. A border closure or supply interruption may create
    high Stress even when the policy is protective.
  Common error: Scoring policy intent rather than operational circulation function.

## METRICS (integer 1-10 or NA)

COHERENCE — internal alignment and coordination clarity
  9-10  Exceptionally unified or seamless. Requires historically unusual evidence.
  7-8   Integrated, strongly aligned.
  5-6   Functional, coordinated despite tension.
  3-4   Divided, factional, inconsistent.
  1-2   Fragmented, paralysed, openly conflicting.

CAPACITY — demonstrated ability to perform the node function
  9-10  Dominant or exceptional. Requires comprehensive, historically unusual evidence.
  7-8   Strong, effective, resilient.
  5-6   Adequate, performs core function.
  3-4   Weak, insufficient, unreliable.
  1-2   Collapsed or unable to function.

STRESS — rate of breakdown or entropy production
  9-10  Rupture, active collapse or severe breakdown.
  7-8   Strained, visible degradation.
  5-6   Pressured, stretched but functioning.
  3-4   Stable, challenges managed.
  1-2   Thriving, low breakdown, strengthening.

ABSTRACTION — operational sophistication, learning capacity, modelling depth
  9-10  Frontier sophistication or transformative learning. Requires exceptional evidence.
  7-8   Advanced modelling, adaptation, institutional learning.
  5-6   Functional learning and procedural sophistication.
  3-4   Basic systems, limited adaptation.
  1-2   Reactive, little learning or continuity.

## TRANSITIONS
Do not automatically score transitions as crises. For regime changes, revolutions,
wars, occupations or collapses, score the observed coordination dynamics: did the
node maintain coordination; could it perform its function; was change managed or
chaotic; was breakdown occurring or only pressure?

Managed transition → moderate stress. Chaotic rupture → high stress. Successful
adaptation → may preserve coherence and capacity.

## SPECIAL CASES
BCE years: negative integers (Athens -431 = 431 BCE).
Fragmented or contested polities: score the dominant or most institutionally
  coherent actor performing each function; still emit one row per node.
Colonial or occupied societies: score the operative institutional layer performing
  each function, which may differ by node (colonial administration for Helm/Shield,
  local population for Hands/Craft).

## INDEPENDENT PASS INTEGRITY
This is one isolated scoring pass. Score entirely on your own judgement against this
rubric. Do not adjust toward an expected ensemble mean. Do not soften or exaggerate
to make multi-pass spread look tighter. Divergence between independent passes is
minimised by following this rubric precisely, not by guessing at consensus.

## OUTPUT FORMAT
First line exactly:

Society,Year,Node,Coherence,Capacity,Stress,Abstraction

Then rows sorted by year ascending, then node order:
Helm, Shield, Lore, Stewards, Craft, Hands, Archive, Flow.

All scores integers 1-10 or NA. No decimals. No blanks. No extra columns.
Do not repeat the header between years.

════════════════════════════════════════════════════


════════════════════════════════════════════════════
SCORER PROMPT — Burma 2024–2026
(Paste this into 5 separate Claude conversations)
════════════════════════════════════════════════════

CAMS RAW SCORER: COUNTRIES v1.2-OPT

country_or_polity: Burma
start_year: 2024
end_year: 2026

If start_year > end_year, return only the header line with no rows.

## OUTPUT CONTRACT
Return ONLY CSV. No prose. No headers beyond the schema line. No markdown.
No evidence notes. No Node Value. No Bond Strength. No SHI. No confidence
labels. No interpolation. No smoothing.

## PROHIBITED → REQUIRED
Explain or justify scores          → Emit CSV rows only.
Narrativise or diagnose            → Score demonstrated function vs the node definition.
Interpolate or smooth across years → Use NA for missing evidence; never copy prior-year values.
Compute derived metrics            → Output raw C,K,S,A only.
Score from sentiment or reputation → Score from concrete functional indicators per node.
Treat one headline as all-node evidence → Evaluate each node against its own evidence.

## TIME CONVENTION
Each year represents institutional condition at 31 December. If the final year
is in progress at your knowledge cutoff, score only evidenced material through
that cutoff; do not project to year-end.

When scoring year t, use 31 December of year t-1 as the baseline. A score
unchanged from t-1 is valid ONLY if a genuine re-check finds no material
functional change. Default repetition is prohibited.

## DATA USE
Use corpus knowledge first. Use targeted web search when: the year is recent or
current; the polity is obscure or thinly represented; the period involves coups,
wars, sanctions, constitutional crises, or major institutional transition; or
governance, military, economic, fiscal or administrative facts are uncertain.

Do not over-search well-established historical periods. Do not cite sources.

If evidence is genuinely insufficient for a specific score, output NA for that
score only. Do not infer from adjacent years. An unchanged score must mean "no
material functional change was found," not "value reused because nothing new
was checked."

## SOURCE POSTURE
Score demonstrated institutional function. Where the available record is
dominated by one geopolitical vantage — reporting about an adversary state, a
sanctioned economy, a wartime opponent, or a former colony — that record
establishes what was reported, not automatically what functioned. Adversarial or
celebratory framing is not itself functional evidence.

This does not license discounting inconvenient evidence. It requires that the
indicator be functional rather than evaluative: budget execution, court
throughput, port volumes, harvest delivery, school enrolment, wage series,
administrative continuity. If only evaluative characterisation is available for
a node-year, the gate has not been satisfied.

## EVIDENCE GATE (per node-year, before scoring)
Confirm ALL three before any non-NA score:
  [ ] One concrete indicator of functional performance observed.
  [ ] One concrete indicator of strain, failure, or limitation observed.
  [ ] Mechanism connecting that evidence to THIS node's specific function identified.

If any box is unchecked → score NA for all four dimensions of this node-year.

Public trust, media controversy, referendum results, and historical guilt may
raise Stress but do NOT by themselves establish loss of Capacity or Coherence.

## UPSTREAM CONSTRAINTS (apply during generation, not after)
1. A change of 2 or more points on any dimension requires EITHER two distinct
   node-specific evidence pieces OR one unmistakable structural discontinuity
   (regime collapse, war onset, currency crisis, partition). General sentiment
   or a single ambiguous event is insufficient.
2. Score below 5 or above 8 requires explicit functional evidence. Controversy,
   criticism, or popularity do not suffice for <5. Absence of problems does not
   suffice for >8.
3. All four dimensions unchanged from t-1 is permitted ONLY after re-checking
   evidence for that node-year. Silent repetition is prohibited.
4. A single national event must not automatically move all eight nodes. Each
   node moves on its own evidence.

## NODES (canonical order: Helm, Shield, Lore, Stewards, Craft, Hands, Archive, Flow)

HELM
  Function: Executive coordination, state strategy, political centre, governing command.
  Boundary: Election victories and parliamentary majorities raise Coherence, not
    Capacity. Capacity is demonstrated execution, not mandate. Orderly transfer of
    power is positive evidence of system continuity even when the outgoing
    government was dysfunctional.
  Common error: Scoring administrative competence from legislative drama, approval
    ratings, or partisan conflict.

SHIELD
  Function: Military, police, intelligence, border control, internal order,
    territorial defence.
  Boundary: Coercive and protective capacity. Budget size or equipment lists do not
    demonstrate operational coordination.
  Common error: Conflating spending with functional readiness; ignoring
    civil-mission strain or politicisation.

LORE
  Function: Education, universities, knowledge institutions, religion, media, public
    meaning, legitimacy production, shared understanding.
  Boundary: Covers production, validation and circulation of knowledge through
    science, universities, professional expertise and epistemic media. Public
    disagreement or misinformation may reduce Coherence and raise Stress without
    eliminating technical Capacity or Abstraction.
  Common error: Scoring ideological alignment as Capacity; conflating narrative
    contestation with institutional collapse.

STEWARDS
  Function: Landholders, capital owners, fiscal elites, state-linked asset managers,
    major investors, resource allocation, ownership power.
  Boundary: Owners and managers of stored resources and long-lived assets
    specifically. NOT the civil service, environmental ministry, or government
    generally — those are Helm or Archive.
  Common error: Substituting state fiscal policy or bureaucratic administration for
    private or asset-holding stewardship.

CRAFT
  Function: Skilled trades, professions, engineers, manufacturing systems, technical
    classes, specialised production.
  Boundary: Productive transformation and technical/industrial competence. Asset
    appreciation and commodity revenue do NOT demonstrate Craft strength. Weak
    productivity growth or deindustrialisation may reduce Craft, but a score below 8
    requires evidence of widespread inability to perform productive transformation —
    not merely declining competitiveness or an unpopular industrial policy.
  Common error: Scoring market value of holdings as productive capacity.

HANDS
  Function: Mass labour, agricultural labour, industrial labour, service labour,
    bodily mobilisation, demographic work capacity.
  Boundary: Low unemployment raises Capacity but does NOT cancel out housing, wage,
    precarity or cost-of-living Stress. Both may be true simultaneously.
  Common error: Using unemployment as the sole signal and ignoring structural
    precarity or wage stagnation.

ARCHIVE
  Function: Bureaucracy, law, courts, records, statistics, civil service,
    institutional memory, continuity across time.
  Boundary: Preservation, retrieval and transmission of institutional memory —
    records, precedent, legal continuity, administrative routines, accumulated
    organisational knowledge. Public trust or media controversy may raise Stress but
    do NOT by themselves establish loss of Capacity.
  Common error: Using national narrative consensus or historical guilt as direct
    measures of bureaucratic function.

FLOW
  Function: Commerce, finance, markets, merchants, logistics, ports, trade, currency,
    banking, circulation of goods and value.
  Boundary: Measures whether exchange and circulation are functioning, not whether an
    interruption was justified. A border closure or supply interruption may create
    high Stress even when the policy is protective.
  Common error: Scoring policy intent rather than operational circulation function.

## METRICS (integer 1-10 or NA)

COHERENCE — internal alignment and coordination clarity
  9-10  Exceptionally unified or seamless. Requires historically unusual evidence.
  7-8   Integrated, strongly aligned.
  5-6   Functional, coordinated despite tension.
  3-4   Divided, factional, inconsistent.
  1-2   Fragmented, paralysed, openly conflicting.

CAPACITY — demonstrated ability to perform the node function
  9-10  Dominant or exceptional. Requires comprehensive, historically unusual evidence.
  7-8   Strong, effective, resilient.
  5-6   Adequate, performs core function.
  3-4   Weak, insufficient, unreliable.
  1-2   Collapsed or unable to function.

STRESS — rate of breakdown or entropy production
  9-10  Rupture, active collapse or severe breakdown.
  7-8   Strained, visible degradation.
  5-6   Pressured, stretched but functioning.
  3-4   Stable, challenges managed.
  1-2   Thriving, low breakdown, strengthening.

ABSTRACTION — operational sophistication, learning capacity, modelling depth
  9-10  Frontier sophistication or transformative learning. Requires exceptional evidence.
  7-8   Advanced modelling, adaptation, institutional learning.
  5-6   Functional learning and procedural sophistication.
  3-4   Basic systems, limited adaptation.
  1-2   Reactive, little learning or continuity.

## TRANSITIONS
Do not automatically score transitions as crises. For regime changes, revolutions,
wars, occupations or collapses, score the observed coordination dynamics: did the
node maintain coordination; could it perform its function; was change managed or
chaotic; was breakdown occurring or only pressure?

Managed transition → moderate stress. Chaotic rupture → high stress. Successful
adaptation → may preserve coherence and capacity.

## SPECIAL CASES
BCE years: negative integers (Athens -431 = 431 BCE).
Fragmented or contested polities: score the dominant or most institutionally
  coherent actor performing each function; still emit one row per node.
Colonial or occupied societies: score the operative institutional layer performing
  each function, which may differ by node (colonial administration for Helm/Shield,
  local population for Hands/Craft).

## INDEPENDENT PASS INTEGRITY
This is one isolated scoring pass. Score entirely on your own judgement against this
rubric. Do not adjust toward an expected ensemble mean. Do not soften or exaggerate
to make multi-pass spread look tighter. Divergence between independent passes is
minimised by following this rubric precisely, not by guessing at consensus.

## OUTPUT FORMAT
First line exactly:

Society,Year,Node,Coherence,Capacity,Stress,Abstraction

Then rows sorted by year ascending, then node order:
Helm, Shield, Lore, Stewards, Craft, Hands, Archive, Flow.

All scores integers 1-10 or NA. No decimals. No blanks. No extra columns.
Do not repeat the header between years.

════════════════════════════════════════════════════


---


════════════════════════════════════════════════════
AGGREGATION PROMPT — Burma 1800–2026
(Paste into ONE new Claude conversation after collecting all scorer outputs)
════════════════════════════════════════════════════

You are a CAMS aggregator. Five independent scorers produced the CSV blocks below.
Where multiple batches exist, rows appear consecutively within each scorer section —
treat all rows from one scorer as a single flat dataset keyed on (Year, Node).

Scores are integers 1-10 or the literal string NA. NA must be handled explicitly at
every step below. Never coerce NA to zero, never skip a row because it contains NA.

## STEP A — De-duplicate batch overlap
A scorer may have scored an overlap year twice, once at the end of one batch and once
at the start of the next. For each such duplicate within a scorer:
  • Keep the LATER occurrence (it was scored with a t-1 anchor in context).
  • Record the absolute difference on each dimension as seam discrepancy.
Report total and maximum seam discrepancy as a single line AFTER Block 2, in the form:
  SEAM: n_overlap_cells=<n>, mean_abs_diff=<x.xx>, max_abs_diff=<x.xx>
If overlap was 0, omit the SEAM line entirely.

## STEP B — Aggregate per dimension (dimension-level NA propagation)
For each (Year, Node, Dimension):
  values  = the non-NA scores across the 5 scorers
  n_eff_d = count(values)

  if n_eff_d >= 3:
      mean = round(mean(values), 1)
      sd   = round(sample_sd(values, ddof=1), 2)
  else:
      mean = NA
      sd   = NA

n_eff for the node-year = MINIMUM n_eff_d across its four dimensions (the weakest
evidential leg — do not average it).

NA_rate for the node-year = (count of NA scores across 5 scorers x 4 dimensions)
divided by 20, rounded to 2 decimals.

## STEP C — Per-scorer Node Value spread
For each (Year, Node), for every scorer whose four dimensions are ALL non-NA:
  V_i = C_i + K_i - S_i + 0.5 x A_i
Record V_min, V_max, V_range = V_max - V_min across those scorers.
If fewer than 3 scorers yield a complete V_i, emit NA for V_range, V_min and V_max.
Compute this FRESH for every cell. A column of identical V_range values is a
computation error, not a result.

## STEP D — Ensemble Node Value
Node Value = mean_C + mean_K - mean_S + 0.5 x mean_A, rounded to 1 decimal.
If ANY of mean_C, mean_K, mean_S, mean_A is NA, Node Value is NA.
Do not compute a partial Node Value.

## STEP E — Bond Strength, with explicit partner accounting
For each year, build the pairwise matrix over nodes using the ensemble means:
  B_ij = [0.6 x C_i x C_j + 0.4 x A_i x A_j] x exp(-(S_i + S_j) / 20)

A pair (i,j) is COMPUTABLE only if C, A and S are non-NA for BOTH nodes.

For node i:
  bond_n = number of computable partners (0-7)
  if bond_n >= 4:
      Bond Strength = mean of the computable B_ij, rounded to 3 decimals
  else:
      Bond Strength = NA
  If node i itself has any NA in C, A or S, Bond Strength = NA and bond_n = 0.

bond_n MUST be emitted. Without it a Bond Strength averaged over 4 partners is
indistinguishable from one averaged over 7, and the two are not comparable.

## STEP F — Emit
Emit EXACTLY two CSV code blocks, in this order, with no prose between them.

Block 1 — ensemble mean:
Society,Year,Node,Coherence,Capacity,Stress,Abstraction,Node Value,Bond Strength

Block 2 — envelope:
Society,Year,Node,C_sd,K_sd,S_sd,A_sd,V_range,V_min,V_max,n_eff,NA_rate,bond_n

Then the SEAM line if overlap was used.

## VERIFICATION before emitting
1. Both header lines match the strings above character for character.
2. V_range values are not all identical.
3. Every NA among Block 1's C/K/S/A columns is matched by NA in that row's Node
   Value and Bond Strength.
4. Every row with n_eff < 5 has NA_rate > 0, and every row with NA_rate > 0 has
   n_eff < 5. A mismatch means NA accounting drifted.
5. Every row with Bond Strength = NA has bond_n < 4 or bond_n = 0.
6. Row counts match between Block 1 and Block 2 on (Society, Year, Node).
7. If a whole node-year is NA across all five scorers, it still appears as a row in
   both blocks with NA values — do not drop it. Absence is the finding.

══════════════════════ SCORER 1 OUTPUT ══════════════════════

[PASTE SCORER 1 CSV HERE — all 10 batches concatenated, header on first batch only]

══════════════════════ SCORER 2 OUTPUT ══════════════════════

[PASTE SCORER 2 CSV HERE — all 10 batches concatenated, header on first batch only]

══════════════════════ SCORER 3 OUTPUT ══════════════════════

[PASTE SCORER 3 CSV HERE — all 10 batches concatenated, header on first batch only]

══════════════════════ SCORER 4 OUTPUT ══════════════════════

[PASTE SCORER 4 CSV HERE — all 10 batches concatenated, header on first batch only]

══════════════════════ SCORER 5 OUTPUT ══════════════════════

[PASTE SCORER 5 CSV HERE — all 10 batches concatenated, header on first batch only]

════════════════════════════════════════════════════


---

─── HOW TO USE ─────────────────────────────────────────────
Burma 1800–2026 · 10 batches · 5 scorers each · 1 aggregation

STEP 1 — For each of the 10 SCORER PROMPTS above:
  • Open 5 Claude conversations
  • Paste the SAME Scorer Prompt into all 5 — do NOT modify it
  • Run all 5 simultaneously
  • Copy each output CSV

  Batch schedule:
    Batch 01: 1800–1824   Batch 06: 1924–1949
    Batch 02: 1824–1849   Batch 07: 1949–1974
    Batch 03: 1849–1874   Batch 08: 1974–1999
    Batch 04: 1874–1899   Batch 09: 1999–2024
    Batch 05: 1899–1924   Batch 10: 2024–2026

  Overlap years (scored twice per scorer, keep the later occurrence):
    1824, 1849, 1874, 1899, 1924, 1949, 1974, 1999, 2024

STEP 2 — For each scorer, concatenate all 10 batch outputs into one block.
  Keep the header from the first batch only; strip it from batches 2–10.

STEP 3 — Open a 6th conversation. Paste the AGGREGATION PROMPT.
  Fill the five scorer sections. Run. You get Block 1 + Block 2 + SEAM line.

Expected output per scorer per batch: up to 200 rows (25 years × 8 nodes).
Total rows in final Block 1: up to 1,816 (227 years × 8 nodes, minus NA dropouts).

Expect NA rows — they are the instrument working, not a failed run.
Expect more NA than a camnations5n single-session run — that difference is
the measurement error a shared context was hiding.
────────────────────────────────────────────────────────────
