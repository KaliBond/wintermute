# JUNO Unified Dataset v3 (2026-10-19)
41,914 rows, 42 societies. Columns: Society, Year, Node, Coherence, Capacity, Stress, Abstraction, Node_Value_v2 (C+K+A/2-S), Source_File, Source_Note. Legacy _Calc columns dropped. No duplicate Society-Year-Node rows.

Added: New Zealand, Greece, Laos, Cambodia, Mongolia, Palestine, Finland (Gemini+Grok mean).
Replaced: Netherlands (m5n ensemble 1850-2025, real Stress), Ukraine (1950-2026 ensemble), Hong Kong, Singapore, USA (annual 1900-2026 CAMNATIONS5 ensemble, was 26 sampled years), Thailand (5-model 5-yearly ensemble; flat single-model annual file rejected).
Patched: Israel, Saudi Arabia, Iran 2015-2026 = annual means of quarterly files. Palestine: 3 duplicate years averaged.
Removed: UAE (ceiling scores 10/10/3-4/10 in every source) pending rescore.
Flagged in Source_Note for blind rescore: Denmark (Stress=0, 870 rows), Israel pre-2015 (flat), India and Turkey (26-31 sampled years; India Hands=0 anomaly).
See JUNO_blind_rescore_list.md.

Patched 2026-10-19 (after audit): Palestine legacy node names mapped to canonical 8; Iraq years 20/21 -> 1920/1921.
