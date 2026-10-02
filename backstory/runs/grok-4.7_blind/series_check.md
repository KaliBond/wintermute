# Series hypothesis check

Input: `runs/grok-4.7_blind/series/node_metrics.csv`

## Trajectories (mean Energy, mean Cognition, Helm-Lore coherence)

| Series | Window | Year | Energy | Cognition | Helm-Lore C |
| --- | --- | --- | --- | --- | --- |
| Egypt | B04 | -2500 | 3.75 | 47.6 | 8.0 |
| Egypt | B13 | -2400 | 2.75 | 44.0 | 7.0 |
| Egypt | B14 | -2250 | 0.12 | 27.8 | 5.5 |
| Egypt | B05 | -2150 | -2.00 | 15.5 | 3.5 |
| Egypt | B15 | -1900 | 3.25 | 51.9 | 7.5 |
| Aegean | B16 | -1350 | 1.50 | 36.9 | 6.0 |
| Aegean | B08 | -1250 | 1.62 | 39.5 | 6.5 |
| Aegean | B09 | -1175 | -3.71 | 11.4 | 3.5 |
| Aegean | B17 | -1000 | -0.33 | 19.7 | 4.5 |
| Aegean | B18 | -750 | 0.88 | 31.9 | 5.5 |

## H2 Hysteresis: recovery slower than fall
- Egypt: Energy fall -2.12/century vs recovery +2.10/century; Cognition fall -12.2 vs recovery +14.5/century -> against
  - Last window vs last pre-collapse window: Energy +3.12, Cognition +24.1
- Aegean: Energy fall -7.12/century vs recovery +1.93/century; Cognition fall -37.4 vs recovery +4.7/century -> supports
  - Last window vs last pre-collapse window: Energy -0.75, Cognition -7.6

## H3 Helm-Lore coherence falls before breakdown
- Egypt: Helm-Lore C 8.0 (B04) -> 5.5 (B14), change -2.5; pre-collapse windows with negative Energy: none -> supports
- Aegean: Helm-Lore C 6.0 (B16) -> 6.5 (B08), change +0.5; pre-collapse windows with negative Energy: none -> against

## H4 Relief devices hold Hands' stress lower for longer
- Mesopotamian Hands Stress by window: {'B19': 6.0, 'B06': 7.0, 'B20': 8.0}; comparator mean 5.50
- Windows below comparator: 0 of 3 -> against
