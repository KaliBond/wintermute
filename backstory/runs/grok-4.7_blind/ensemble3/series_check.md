# Series hypothesis check

Input: `runs/grok-4.7_blind/ensemble3/series/node_metrics.csv`

## Trajectories (mean Energy, mean Cognition, Helm-Lore coherence)

| Series | Window | Year | Energy | Cognition | Helm-Lore C |
| --- | --- | --- | --- | --- | --- |
| Egypt | B04 | -2500 | 3.50 | 48.0 | 8.0 |
| Egypt | B13 | -2400 | 2.35 | 43.5 | 7.0 |
| Egypt | B14 | -2250 | 0.07 | 29.0 | 5.5 |
| Egypt | B05 | -2150 | -2.20 | 15.7 | 3.5 |
| Egypt | B15 | -1900 | 3.00 | 52.5 | 7.2 |
| Aegean | B16 | -1350 | 1.52 | 38.3 | 6.3 |
| Aegean | B08 | -1250 | 1.53 | 40.1 | 6.5 |
| Aegean | B09 | -1175 | -3.41 | 12.7 | 3.4 |
| Aegean | B17 | -1000 | -0.85 | 18.6 | 4.5 |
| Aegean | B18 | -750 | 0.75 | 32.0 | 5.5 |

## H2 Hysteresis: recovery slower than fall
- Egypt: Energy fall -2.28/century vs recovery +2.08/century; Cognition fall -13.4 vs recovery +14.7/century -> against
  - Last window vs last pre-collapse window: Energy +2.92, Cognition +23.5
- Aegean: Energy fall -6.59/century vs recovery +1.47/century; Cognition fall -36.6 vs recovery +3.4/century -> supports
  - Last window vs last pre-collapse window: Energy -0.78, Cognition -8.1

## H3 Helm-Lore coherence falls before breakdown
- Egypt: Helm-Lore C 8.0 (B04) -> 5.5 (B14), change -2.5; pre-collapse windows with negative Energy: none -> supports
- Aegean: Helm-Lore C 6.3 (B16) -> 6.5 (B08), change +0.2; pre-collapse windows with negative Energy: none -> against

## H4 Relief devices hold Hands' stress lower for longer
- Mesopotamian Hands Stress by window: {'B19': 6.7, 'B06': 6.7, 'B20': 7.7}; comparator mean 5.72
- Windows below comparator: 0 of 3 -> against
