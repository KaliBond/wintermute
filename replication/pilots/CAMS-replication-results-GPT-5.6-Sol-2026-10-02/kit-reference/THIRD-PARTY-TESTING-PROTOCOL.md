# CAMS Third-Party Testing Protocol — TP-1 (2026-10-02)

*Governing document for independent replication of CAMS scoring runs by a third party who is not the model author and not the packet assembler. Companion to REPLICATION-KIT v1.0; where documents conflict, this protocol governs conduct and UIS v1.4 governs scoring.*

## 1. Purpose and Design

Experiment Run A produced CAMS scores for one node (Helm) of two historical societies (anonymised as BLIND-A, 100 BCE and BLIND-B, 100 CE) using five passes by a single AI model that had also assembled the evidence packets. Two threats to validity follow: scorer-assembler identity, and same-model correlation. Third-party testing removes both by handing the packets to scorers the author does not control, on platforms the author does not choose, under rules the tester enforces.

Design: independent replications, one packet per scorer session, no cross-session visibility, verbatim artefact return, hash-verified evidence.

## 2. Roles

- **Tester** — the third party. Recruits scorers, runs sessions, collects artefacts, certifies the contamination controls. The Tester must not have seen Run A's scores.
- **Scorer** — an AI model on any platform, or a human expert, engaged fresh for one packet. Scorers may recognise the historical entity from the facts; recognition is recorded, not disqualifying.
- **Author** (K. McKern / lab) — supplies the sealed kit, receives artefacts, performs recombination. The Author has no contact with scorers during testing.

## 3. Materials (all in REPLICATION-KIT)

| File | Use |
|---|---|
| SCORER-PROMPT.md | pasted first into every scorer session |
| UIS-v1.4-summary.md | pasted second — the scoring rules |
| packet-Helm-BLIND-A.redacted.md | evidence, entity A |
| packet-Helm-BLIND-B.redacted.md | evidence, entity B |
| PACKET-HASHES.txt | evidence-integrity check |

The Tester uses **redacted packets only**. If a session interface truncates a packet, the Tester re-sends it in parts and records the truncation.

## 4. Procedure

1. **Briefing isolation.** Do not tell any scorer what Run A scored, what the model is "supposed" to find, or which societies are involved. Do not discuss CAMS theory with the scorer.
2. **One packet per session.** Open a fresh session (new chat, no history). Paste SCORER-PROMPT, then UIS-v1.4-summary, then one packet. Never place both packets in one session; never reuse a session for the second packet.
3. **No coaching.** If the scorer asks clarifying questions, answer only by quoting the packet or the rules summary. If the scorer refuses to score, record REFUSED for the affected metrics and end the session — refusal is data.
4. **Record verbatim.** Save the scorer's CSV row and protocol note exactly as returned, unedited. Also record: platform, model name and version as the platform reports them, date and timezone, packet version (redacted), whether the scorer stated it recognised the entity, and any deviations from this protocol.
5. **Integrity check.** Hash the packet file actually sent (SHA-256) against PACKET-HASHES.txt. Record match or mismatch.
6. **Minimum design.** At least three independent scorer draws per packet, from at least two distinct model lineages or human experts. More is better; correlated re-runs on one platform count as one lineage.
7. **Return.** Send all artefacts to the Author as files, not screenshots. Include a signed Tester's declaration (§6).

## 5. Prohibited Contaminations (any of these voids a draw)

- Scorer shown Run A scores, programme documents, or the CAMS Intellectual Spine before scoring.
- Both packets in one session, or the second packet scored in a session that saw the first.
- Editing, "improving", or arguing with the scorer's output before return.
- Selecting which draws to return (all completed draws must be returned, including outliers and refusals).

## 6. Tester's Declaration (template)

```
I, <name>, conducted <N> scorer sessions under CAMS Protocol TP-1 between <dates>.
Platforms/models used: <list, one line each, with version as reported>.
I confirm: redacted packets only; one packet per fresh session; no scorer saw Run A
scores or CAMS programme material; all completed draws are returned verbatim;
no draw was excluded after inspection of its scores.
Deviations from protocol: <list or "none">.
Signature / date.
```

## 7. What the Author Does on Receipt (fixed in advance)

1. Log each draw in the LAB-NOTEBOOK as its own run entry with timestamp; seal artefacts by SHA-256.
2. Pool foreign draws per entity per metric; compute median and envelope (sd, range, min, max, N). Keep Run A's same-model block separate and labelled.
3. Flag any metric where |third-party median − Run A median| ≥ 2 for packet review. The flag triggers re-examination of the packet evidence and wording — not pressure on scorers.
4. Publish the pooled result, the spread, and all verbatim scorer artefacts. Spread is reported and left standing; inconsistency is data.

## 8. Interpretation Rules (agreed before any scores are seen)

- **Convergence** (medians within 1 point on all four metrics, both entities): the packet-plus-rules instrument transmits stably across scorers; credit to the instrument, not to any model.
- **Structured divergence** (one metric diverges across many lineages): the packet's evidence for that metric is ambiguous — revise the packet, re-run as a new experiment.
- **Unstructured divergence** (scorers scatter on everything): the rules are not yet teachable as written; UIS v1.4 needs revision before further historical scoring.
- No outcome counts as failure. A null or scattering result is a measurement of the instrument, which is what this test is for.
