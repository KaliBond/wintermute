# Series hypothesis check

Input: `runs/grok-4.7/series/node_metrics.csv`

## Trajectories (mean Energy, mean Cognition, Helm-Lore coherence)

| Series | Window | Year | Energy | Cognition | Helm-Lore C |
| --- | --- | --- | --- | --- | --- |
| Egypt | B04 | -2500 | 3.50 | 46.8 | 7.5 |
| Egypt | B13 | -2400 | 2.00 | 39.4 | 6.5 |
| Egypt | B14 | -2250 | 0.50 | 31.1 | 5.5 |
| Egypt | B05 | -2150 | -2.38 | 15.5 | 3.5 |
| Egypt | B15 | -1900 | 3.00 | 44.9 | 7.0 |
| Aegean | B16 | -1350 | 1.75 | 36.1 | 6.0 |
| Aegean | B08 | -1250 | 1.62 | 41.6 | 6.5 |
| Aegean | B09 | -1175 | -3.14 | 13.0 | 3.5 |
| Aegean | B17 | -1000 | -0.29 | 17.0 | 4.0 |
| Aegean | B18 | -750 | 0.88 | 30.5 | 5.5 |

## H2 Hysteresis: recovery slower than fall
- Egypt: Energy fall -2.88/century vs recovery +2.15/century; Cognition fall -15.6 vs recovery +11.8/century -> supports
  - Last window vs last pre-collapse window: Energy +2.50, Cognition +13.8
- Aegean: Energy fall -6.36/century vs recovery +1.63/century; Cognition fall -38.2 vs recovery +2.3/century -> supports
  - Last window vs last pre-collapse window: Energy -0.75, Cognition -11.1

## H3 Helm-Lore coherence falls before breakdown
- Egypt: Helm-Lore C 7.5 (B04) -> 5.5 (B14), change -2.0; pre-collapse windows with negative Energy: none -> supports
- Aegean: Helm-Lore C 6.0 (B16) -> 6.5 (B08), change +0.5; pre-collapse windows with negative Energy: none -> against

## H4 Relief devices hold Hands' stress lower for longer
- Mesopotamian Hands Stress by window: {'B19': 7.0, 'B06': 7.0, 'B20': 7.0}; comparator mean 5.33
- Windows below comparator: 0 of 3 -> against
