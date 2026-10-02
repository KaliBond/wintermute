# Series hypothesis check

Input: `runs/replication_R6/series/node_metrics.csv`

## Trajectories (mean Energy, mean Cognition, Helm-Lore coherence)

| Series | Window | Year | Energy | Cognition | Helm-Lore C |
| --- | --- | --- | --- | --- | --- |
| Egypt | B04 | -2500 | 4.10 | 49.8 | 8.1 |
| Egypt | B13 | -2400 | 2.93 | 44.1 | 7.2 |
| Egypt | B14 | -2250 | -0.15 | 29.4 | 5.2 |
| Egypt | B05 | -2150 | -3.02 | 15.7 | 3.8 |
| Egypt | B15 | -1900 | 3.50 | 50.5 | 7.6 |
| Aegean | B16 | -1350 | 2.34 | 34.1 | 5.5 |
| Aegean | B08 | -1250 | 0.87 | 34.0 | 5.8 |
| Aegean | B09 | -1175 | -4.45 | 10.5 | 3.1 |
| Aegean | B17 | -1000 | -1.19 | 14.1 | 3.8 |
| Aegean | B18 | -750 | 0.73 | 28.0 | 5.4 |

## H2 Hysteresis: recovery slower than fall
- Egypt: Energy fall -2.88/century vs recovery +2.61/century; Cognition fall -13.6 vs recovery +13.9/century -> against
  - Last window vs last pre-collapse window: Energy +3.65, Cognition +21.1
- Aegean: Energy fall -7.10/century vs recovery +1.87/century; Cognition fall -31.3 vs recovery +2.1/century -> supports
  - Last window vs last pre-collapse window: Energy -0.15, Cognition -6.0

## H3 Helm-Lore coherence falls before breakdown
- Egypt: Helm-Lore C 8.1 (B04) -> 5.2 (B14), change -2.9; pre-collapse windows with negative Energy: ['B14'] -> supports
- Aegean: Helm-Lore C 5.5 (B16) -> 5.8 (B08), change +0.3; pre-collapse windows with negative Energy: none -> against

## H4 Relief devices hold Hands' stress lower for longer
- Mesopotamian Hands Stress by window: {'B19': 6.6, 'B06': 6.2, 'B20': 7.0}; comparator mean 5.73
- Windows below comparator: 0 of 3 -> against
