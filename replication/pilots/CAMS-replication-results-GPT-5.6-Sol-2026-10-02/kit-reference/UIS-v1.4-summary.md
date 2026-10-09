# UIS v1.4 — Scoring Rules Summary (for foreign-platform scorers)

Condensed from the CAMS Universal Instruction Set v1.4 (22 September 2026). The full text governs; this summary is what a replication scorer needs in one screen.

**Core stance.** Score the named function only. Do not assemble a system story. Do not use one node to adjust another. Do not identify the entity; if the packet names offices or famous persons, ignore prestige — score performance of the function. If the packet implies what the whole system "is", ignore the label; score the function at the scale the packet itself describes.

**Year.** Terminal-year rule: the score represents the evidenced condition at the packet's year. Later events must not be back-cast. Facts the packet lists as exclusions are inadmissible.

**Metrics (integer 1–10 or NA).**

| Score | Coherence — acts as one? | Capacity — can it do the job at this scale? | Stress — how hard pressed? | Abstraction — method or reflex? |
|---|---|---|---|---|
| 10 | Unusual unity | Dominant surplus | Rupture | Frontier learning |
| 8 | Strong alignment | Strong, with margin | Visible degradation | Used plans and learning |
| 6 | Works despite tension | Core job done, little margin | Stretched, still working | Plans and reflex alternate |
| 5 | Mixed signals, still working | Barely reliable | Pressure present and felt | Models often ignored |
| 4 | Division impedes output | Chronic shortage | Challenges inside routine | Plans exist, rarely govern |
| 2 | Open conflict | Core job unreliable | Strengthening or idle | Almost no foresight |
| 1 | Schism or no agent | Absent or destructive | Negligible load | Reflex only |

Anchors are defined only for 10, 8, 6, 5, 4, 2, 1. Scores of 9, 7, 3 are permitted integers judged between adjacent anchors; do not invent anchor text for them.

**Hard constraints.**

- No decimals. No zero.
- Stress is load; never invert it. High Stress with high Capacity is overload of a working function, not collapse.
- Title, budget, headcount, vote, price and fame are not Capacity.
- Financial facts are hostile witnesses: a score whose anchor fact is a financial token counts only with material corroboration (a physical, organisational or behavioural fact about the same function).
- NA if the packet lacks both a performance fact and a strain fact for the function.
- A move of 2 or more points, or any score <5 or >8, needs a fact in the packet that is about this function.
- Do not score by counting institutions: institutions that existed without producing usable output move nothing.

**Output.** One CSV row `Entity,Year,Node,Coherence,Capacity,Stress,Abstraction` (integers or NA), then the protocol note. Nothing else.
