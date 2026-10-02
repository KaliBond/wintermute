# Series hypothesis check

Input: `runs/three_passes_unlabelled/series/node_metrics.csv`

## Trajectories (mean Energy, mean Cognition, Helm-Lore coherence)

| Series | Window | Year | Energy | Cognition | Helm-Lore C |
| --- | --- | --- | --- | --- | --- |
| Egypt | B04 | -2500 | 2.79 | 58.5 | 8.0 |
| Egypt | B13 | -2400 | 2.25 | 56.0 | 7.2 |
| Egypt | B14 | -2250 | 0.82 | 46.3 | 6.5 |
| Egypt | B05 | -2150 | -0.93 | 34.4 | 5.5 |
| Egypt | B15 | -1900 | 2.54 | 60.4 | 7.7 |
| Aegean | B16 | -1350 | 2.12 | 49.5 | 6.8 |
| Aegean | B08 | -1250 | 1.22 | 51.0 | 6.8 |
| Aegean | B09 | -1175 | -1.90 | 27.7 | 5.0 |
| Aegean | B17 | -1000 | 0.88 | 39.2 | nan |
| Aegean | B18 | -750 | 1.45 | 46.3 | 6.5 |

## H2 Hysteresis: recovery slower than fall
- Egypt: Energy fall -1.75/century vs recovery +1.39/century; Cognition fall -11.9 vs recovery +10.4/century -> supports
  - Last window vs last pre-collapse window: Energy +1.71, Cognition +14.1
- Aegean: Energy fall -4.17/century vs recovery +1.59/century; Cognition fall -31.1 vs recovery +6.6/century -> supports
  - Last window vs last pre-collapse window: Energy +0.23, Cognition -4.7

## H3 Helm-Lore coherence falls before breakdown
- Egypt: Helm-Lore C 8.0 (B04) -> 6.5 (B14), change -1.5; pre-collapse windows with negative Energy: none -> supports
- Aegean: Helm-Lore C 6.8 (B16) -> 6.8 (B08), change +0.0; pre-collapse windows with negative Energy: none -> against

## H4 Relief devices hold Hands' stress lower for longer
- Mesopotamian Hands Stress by window: {'B19': 7.7, 'B06': 7.3, 'B20': 7.3}; comparator mean 6.82
- Windows below comparator: 0 of 3 -> against
