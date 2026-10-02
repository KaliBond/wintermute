# Series hypothesis check

Input: `runs/grok-4.7_blind/ensemble4/series/node_metrics.csv`

## Trajectories (mean Energy, mean Cognition, Helm-Lore coherence)

| Series | Window | Year | Energy | Cognition | Helm-Lore C |
| --- | --- | --- | --- | --- | --- |
| Egypt | B04 | -2500 | 3.51 | 48.8 | 8.0 |
| Egypt | B13 | -2400 | 2.38 | 42.7 | 7.0 |
| Egypt | B14 | -2250 | 0.09 | 28.6 | 5.4 |
| Egypt | B05 | -2150 | -1.94 | 17.7 | 3.9 |
| Egypt | B15 | -1900 | 3.08 | 51.1 | 7.1 |
| Aegean | B16 | -1350 | 1.58 | 37.0 | 6.2 |
| Aegean | B08 | -1250 | 1.50 | 40.0 | 6.5 |
| Aegean | B09 | -1175 | -3.14 | 13.6 | 3.5 |
| Aegean | B17 | -1000 | -0.92 | 18.4 | 4.5 |
| Aegean | B18 | -750 | 0.75 | 32.3 | 5.5 |

## H2 Hysteresis: recovery slower than fall
- Egypt: Energy fall -2.02/century vs recovery +2.01/century; Cognition fall -11.0 vs recovery +13.4/century -> against
  - Last window vs last pre-collapse window: Energy +2.99, Cognition +22.4
- Aegean: Energy fall -6.19/century vs recovery +1.27/century; Cognition fall -35.2 vs recovery +2.8/century -> supports
  - Last window vs last pre-collapse window: Energy -0.75, Cognition -7.7

## H3 Helm-Lore coherence falls before breakdown
- Egypt: Helm-Lore C 8.0 (B04) -> 5.4 (B14), change -2.6; pre-collapse windows with negative Energy: none -> supports
- Aegean: Helm-Lore C 6.2 (B16) -> 6.5 (B08), change +0.3; pre-collapse windows with negative Energy: none -> against

## H4 Relief devices hold Hands' stress lower for longer
- Mesopotamian Hands Stress by window: {'B19': 6.8, 'B06': 6.8, 'B20': 7.8}; comparator mean 5.70
- Windows below comparator: 0 of 3 -> against
