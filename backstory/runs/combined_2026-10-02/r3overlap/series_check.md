# Series hypothesis check

Input: `runs/combined_2026-10-02/r3overlap/node_metrics.csv`

## Trajectories (mean Energy, mean Cognition, Helm-Lore coherence)

| Series | Window | Year | Energy | Cognition | Helm-Lore C |
| --- | --- | --- | --- | --- | --- |
| Egypt | B04 | -2500 | 3.65 | 44.4 | 8.0 |
| Egypt | B13 | -2400 | 2.60 | 37.8 | 6.6 |
| Egypt | B14 | -2250 | 0.03 | 26.5 | 5.0 |
| Egypt | B05 | -2150 | -3.18 | 12.9 | 3.0 |
| Egypt | B15 | -1900 | 3.33 | 45.0 | 7.0 |
| Aegean | B16 | -1350 | 2.12 | 31.6 | 5.5 |
| Aegean | B08 | -1250 | 0.70 | 32.5 | 5.6 |
| Aegean | B09 | -1175 | -3.73 | 10.9 | 3.2 |
| Aegean | B17 | -1000 | -1.15 | 14.2 | 3.9 |
| Aegean | B18 | -750 | 0.65 | 26.8 | 5.4 |

## H2 Hysteresis: recovery slower than fall
- Egypt: Energy fall -3.20/century vs recovery +2.60/century; Cognition fall -13.5 vs recovery +12.8/century -> supports
  - Last window vs last pre-collapse window: Energy +3.30, Cognition +18.5
- Aegean: Energy fall -5.90/century vs recovery +1.47/century; Cognition fall -28.7 vs recovery +1.9/century -> supports
  - Last window vs last pre-collapse window: Energy -0.05, Cognition -5.7

## H3 Helm-Lore coherence falls before breakdown
- Egypt: Helm-Lore C 8.0 (B04) -> 5.0 (B14), change -3.0; pre-collapse windows with negative Energy: none -> supports
- Aegean: Helm-Lore C 5.5 (B16) -> 5.6 (B08), change +0.1; pre-collapse windows with negative Energy: none -> against

## H4 Relief devices hold Hands' stress lower for longer
- Mesopotamian Hands Stress by window: {'B19': 7.0, 'B06': 6.2, 'B20': 7.2}; comparator mean 5.93
- Windows below comparator: 0 of 3 -> against
