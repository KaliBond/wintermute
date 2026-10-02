# Series hypothesis check

Input: `runs/kimi_blind/series/node_metrics.csv`

## Trajectories (mean Energy, mean Cognition, Helm-Lore coherence)

| Series | Window | Year | Energy | Cognition | Helm-Lore C |
| --- | --- | --- | --- | --- | --- |
| Egypt | B04 | -2500 | 3.50 | 44.2 | 8.5 |
| Egypt | B13 | -2400 | 2.50 | 41.9 | 7.5 |
| Egypt | B14 | -2250 | -0.12 | 27.1 | 5.5 |
| Egypt | B05 | -2150 | -1.75 | 18.1 | 4.5 |
| Egypt | B15 | -1900 | 3.62 | 46.0 | 8.0 |
| Aegean | B16 | -1350 | 2.25 | 38.8 | 7.0 |
| Aegean | B08 | -1250 | 2.00 | 37.9 | 7.0 |
| Aegean | B09 | -1175 | -3.12 | 11.0 | 4.0 |
| Aegean | B17 | -1000 | -3.12 | 7.6 | 3.5 |
| Aegean | B18 | -750 | 0.88 | 28.5 | 6.5 |

## H2 Hysteresis: recovery slower than fall
- Egypt: Energy fall -1.62/century vs recovery +2.15/century; Cognition fall -9.0 vs recovery +11.2/century -> against
  - Last window vs last pre-collapse window: Energy +3.75, Cognition +18.9
- Aegean: Energy fall -6.83/century vs recovery +0.00/century; Cognition fall -35.8 vs recovery -1.9/century -> supports
  - Last window vs last pre-collapse window: Energy -1.12, Cognition -9.4

## H3 Helm-Lore coherence falls before breakdown
- Egypt: Helm-Lore C 8.5 (B04) -> 5.5 (B14), change -3.0; pre-collapse windows with negative Energy: ['B14'] -> supports
- Aegean: Helm-Lore C 7.0 (B16) -> 7.0 (B08), change +0.0; pre-collapse windows with negative Energy: none -> against

## H4 Relief devices hold Hands' stress lower for longer
- Mesopotamian Hands Stress by window: {'B19': 6.0, 'B06': 6.0, 'B20': 7.0}; comparator mean 5.33
- Windows below comparator: 0 of 3 -> against
