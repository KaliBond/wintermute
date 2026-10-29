# ENV file note (second upload; not used in any test)
Header: Society,Year,Node,C_sd,K_sd,S_sd,A_sd,V_range,V_min,V_max. Society "USA". No scores. 1008 data rows = 126 year-blocks of 8 nodes. Read in full with Read; not copied to the box (Shell cannot see attachments).

Data-quality findings (from the Read output):
- Years 1927 and 1928 are missing.
- Year 1907 appears twice, out of sequence, with conflicting values. The first block (in 1900-1926 order) is the flat default; the second block (after 1926, before 1929) has Helm, Stewards and Flow wider (SDs 0.6/0.4/0.3/0.3, V_range 2.0, V_min -1.0, V_max 1.0). It looks like a misplaced 1907 or a template overwrite.
- 1900-1926 is a flat default (SDs 0.3, Hands 0.4/0.4/0.3/0.4, Archive K_sd and A_sd 0.0, V_range 1.0, V_min -0.5, V_max 0.5) apart from the second 1907 block.
- From 1929 the SDs vary by node and period: wide around 1929-33 (V_range up to 3.0 for Helm, Stewards, Flow), 1934-45 about 0.5/0.5/0.4/0.3, flat 0.3 over 1946-67, wider in 1968, 1969-79 about 0.4, Helm wider 1973-74, Hands wider 1980-82, flat 0.3 over 1983-2000, wider in 2001-03 (Shield), 2008-09 (Helm, Stewards, Flow), 2016-19 (Helm), and 2020-26 (all nodes, Helm and Lore widest).
- V_min and V_max are not symmetric around zero everywhere.
So the envelope is informative mainly around crises and transition-adjacent years and flat elsewhere. That is a possible circularity if used to weight or flag transition windows.

Possible uses (none alter the frozen rules; all exploratory, to be pre-labelled):
1. Uncertainty on Q: by propagation Var(Q) ~ (C*A_sd)^2 + (A*C_sd)^2 + (A_sd*C_sd)^2 + K_sd^2 + S_sd^2 (independence assumed, which is a strong assumption). Gives approximate SD bands per node-year, and a check of whether node-year acute flags (|z| >= 1.96) exceed the stated SD.
2. Robustness of H2: draw Q from Normal(Q, SD) (or uniform within V_min..V_max) N times, recompute X, the P90 threshold, S_lead and p_N1, and report the spread of S_lead. Only meaningful once 1907, 1927 and 1928 are fixed.
3. Down-weighting wide-envelope node-years in H1 profile correlations, as a sensitivity row.
4. Flagging transition-adjacent years with wide envelopes (1929-33, 1973-74, 2008-09, 2020-26) as a descriptive table.
Needs before use: confirm which 1907 block is right, supply 1927-28, and confirm that the 1900-1926 flat values are an intended default and not measured spread.
