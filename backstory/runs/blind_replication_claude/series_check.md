# Series hypothesis check

Input: `runs/blind_replication_claude/series/node_metrics.csv`

## Trajectories (mean Energy, mean Cognition, Helm-Lore coherence)

| Series | Window | Year | Energy | Cognition | Helm-Lore C |
| --- | --- | --- | --- | --- | --- |
| Egypt | B04 | -2500 | 4.10 | 46.9 | 8.0 |
| Egypt | B13 | -2400 | 2.52 | 38.5 | 6.7 |
| Egypt | B14 | -2250 | -0.20 | 25.1 | 5.0 |
| Egypt | B05 | -2150 | -3.08 | 15.2 | 3.6 |
| Egypt | B15 | -1900 | 3.40 | 46.6 | 7.2 |
| Aegean | B16 | -1350 | 2.40 | 33.3 | nan |
| Aegean | B08 | -1250 | 0.80 | 32.9 | 5.7 |
| Aegean | B09 | -1175 | -4.85 | 9.5 | 3.1 |
| Aegean | B17 | -1000 | -1.17 | 13.3 | 3.7 |
| Aegean | B18 | -750 | 0.82 | 27.9 | 5.5 |

## H2 Hysteresis: recovery slower than fall
- Egypt: Energy fall -2.88/century vs recovery +2.59/century; Cognition fall -9.9 vs recovery +12.6/century -> against
  - Last window vs last pre-collapse window: Energy +3.60, Cognition +21.5
- Aegean: Energy fall -7.53/century vs recovery +2.10/century; Cognition fall -31.1 vs recovery +2.1/century -> supports
  - Last window vs last pre-collapse window: Energy +0.02, Cognition -5.0

## H3 Helm-Lore coherence falls before breakdown
- Egypt: Helm-Lore C 8.0 (B04) -> 5.0 (B14), change -3.0; pre-collapse windows with negative Energy: ['B14'] -> supports
- Aegean: Helm-Lore C nan (B16) -> 5.7 (B08), change +nan; pre-collapse windows with negative Energy: none -> not testable (Helm or Lore NA)

## H4 Relief devices hold Hands' stress lower for longer
- Mesopotamian Hands Stress by window: {'B19': 6.6, 'B06': 6.0, 'B20': 7.0}; comparator mean 5.73
- Windows below comparator: 0 of 3 -> against
