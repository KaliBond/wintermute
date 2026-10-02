# Series hypothesis check

Input: `runs/claude_grand/series/node_metrics.csv`

## Trajectories (mean Energy, mean Cognition, Helm-Lore coherence)

| Series | Window | Year | Energy | Cognition | Helm-Lore C |
| --- | --- | --- | --- | --- | --- |
| Egypt | B04 | -2500 | 3.95 | 47.1 | 8.1 |
| Egypt | B13 | -2400 | 2.69 | 40.0 | 6.8 |
| Egypt | B14 | -2250 | -0.11 | 27.0 | 5.1 |
| Egypt | B05 | -2150 | -2.98 | 15.4 | 3.6 |
| Egypt | B15 | -1900 | 3.43 | 47.3 | 7.2 |
| Aegean | B16 | -1350 | 2.23 | 32.6 | 5.5 |
| Aegean | B08 | -1250 | 0.80 | 33.2 | 5.7 |
| Aegean | B09 | -1175 | -4.65 | 10.0 | 3.0 |
| Aegean | B17 | -1000 | -1.24 | 13.6 | 3.8 |
| Aegean | B18 | -750 | 0.74 | 27.5 | 5.4 |

## H2 Hysteresis: recovery slower than fall
- Egypt: Energy fall -2.86/century vs recovery +2.56/century; Cognition fall -11.6 vs recovery +12.7/century -> against
  - Last window vs last pre-collapse window: Energy +3.54, Cognition +20.3
- Aegean: Energy fall -7.27/century vs recovery +1.95/century; Cognition fall -30.9 vs recovery +2.1/century -> supports
  - Last window vs last pre-collapse window: Energy -0.06, Cognition -5.6

## H3 Helm-Lore coherence falls before breakdown
- Egypt: Helm-Lore C 8.1 (B04) -> 5.1 (B14), change -3.0; pre-collapse windows with negative Energy: ['B14'] -> supports
- Aegean: Helm-Lore C 5.5 (B16) -> 5.7 (B08), change +0.2; pre-collapse windows with negative Energy: none -> against

## H4 Relief devices hold Hands' stress lower for longer
- Mesopotamian Hands Stress by window: {'B19': 6.7, 'B06': 6.1, 'B20': 7.1}; comparator mean 5.82
- Windows below comparator: 0 of 3 -> against
