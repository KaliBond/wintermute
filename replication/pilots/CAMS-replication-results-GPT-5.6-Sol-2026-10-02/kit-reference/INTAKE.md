# INTAKE — Returning Cross-Platform Scores

Save each foreign run's artefacts in this workspace under:

```
scores/replication/<PLATFORM>/<MODEL>/<ENTITY>/   e.g. scores/replication/chatgpt/gpt-5/BLIND-A/
  raw.csv            the single CSV row returned, verbatim, with header added:
                     Entity,Year,Node,Coherence,Capacity,Stress,Abstraction
  protocol.md        the returned protocol note, verbatim
  session-meta.md    platform, model + version as reported, date, operator,
                     which packet version was used (full / redacted),
                     confirmation the scorer saw no Run A scores beforehand,
                     REFUSED metrics if any
```

## Recombination rule (applied here after intake)

1. Pool all valid foreign draws per entity per metric. Run A's five same-model passes stay a separate provenance block — never silently pooled.
2. Cross-platform `ensemble-xp.csv`: median per metric over foreign draws (one decimal permitted).
3. Cross-platform `envelope-xp.csv`: sd, range, min, max, N, with provenance column listing pipe-separated `platform/model` identities.
4. Divergence flag: |foreign median − Run A median| ≥ 2 on any metric → packet review. The disagreement is a finding, not an error to be smoothed.
5. Each completed replication run gets its own LAB-NOTEBOOK entry (Run C, D, …) with timestamp and seal, extending `scores/ensemble-2026-10-02/LAB-SEAL.txt` — never editing it.

## Verification

Before accepting a foreign run, hash the packet file that was actually sent against `PACKET-HASHES.txt`. If the packet was modified in transit (truncation by a chat interface is the common case), record the modification in session-meta.md — a truncated packet is a different evidence draw and must be flagged.
