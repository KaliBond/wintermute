#!/bin/bash
# Verifier v2, layer B: source checks on lab and note (see criteria.md).
cd "$(dirname "$0")"
LAB=../../../shannon-laboratory.html
NOTE=../../../papers/CAMS_Informational_Perspective_Research_Note.md
fails=0
chk() { # id, description, command-result
  if [ "$3" -ge 1 ]; then echo "PASS $1 $2"; else echo "FAIL $1 $2"; fails=$((fails+1)); fi
}
absent() { if [ "$3" -eq 0 ]; then echo "PASS $1 $2"; else echo "FAIL $1 $2"; fails=$((fails+1)); fi }

chk B1a "lab: eta_loop as Library Attractor formula" "$(grep -c 'BS<sub>Lore</sub>' $LAB)"
absent B1b "lab: no eta_loop <-> TE mapping" "$(grep -ci 'Directed transfer entropy to Helm and Lore' $LAB)"
chk B2 "lab: Stress = environmental entropy rate, not noise" "$(grep -c 'Rate of change' $LAB)"
chk B3a "lab: coherence split row" "$(grep -c 'Coherence C (within a node)' $LAB)"
chk B3b "lab: semantic information (Kolchinsky" "$(grep -c 'Kolchinsky' $LAB)"
chk B4a "lab: muted-source baseline ticks in TE chart" "$(grep -c 'edgesMuted' $LAB)"
chk B4b "lab: capacity tile labelled BSC model" "$(grep -c 'Gate channel capacity (BSC model)' $LAB)"
chk B5 "lab: authority-echo panel" "$(grep -c 'The authority echo' $LAB)"
chk B6 "lab: memory panel with out-of-phase control" "$(grep -c 'misaligned' $LAB)"
chk B7a "note: dDIG cited as pre-existing register" "$(grep -c 'ddig_analysis.py' $NOTE)"
chk B7b "note: ESD cited" "$(grep -c 'cams_analyzer.py' $NOTE)"
absent B7c "note: no until-now claim" "$(grep -ci 'supplied, until now' $NOTE)"
chk B8a "note: eta_loop = (B_Lore x B_Archive)/S_Hands" "$(grep -c '(B_Lore × B_Archive)/S_Hands' $NOTE)"
chk B8b "note: mislabel correction logged" "$(grep -c 'Correction logged' $NOTE)"
chk B9a "note: common-drive attenuation (24%/48%)" "$(grep -c '24% of Society A' $NOTE)"
chk B9b "note: A inversion under conditioning" "$(grep -c 'inverts in A' $NOTE)"
chk B9c "note: authority echo figures" "$(grep -c '0.161 bits of agreement' $NOTE)"
chk B9d "note: memory worth 0.218 bits" "$(grep -c '0.218 bits' $NOTE)"
chk B10 "note: new references (9)" "$(grep -cE 'Ashby|Tishby|Bialek|Kolchinsky|Rosas|Flack, J. C., & Krakauer|Shin, J|Heilmann|Wang, S' $NOTE)"
chk B11 "note: project-ledger provenance marks" "$(grep -c 'deposit pending' $NOTE)"
chk B12 "note: gate-as-regime caveat (Heilmann; Wang & Yang)" "$(grep -c 'channels, not regime type' $NOTE)"
chk B13a "note: INFO-2 revised to conditional TE" "$(grep -c 'common drive and is withdrawn' $NOTE)"
chk B13b "note: coding fork (INFO-6)" "$(grep -c 'INFO-6' $NOTE)"
chk B13c "note: echo test (INFO-7)" "$(grep -c 'INFO-7' $NOTE)"
chk B13d "note: memory/feedback substitutes (INFO-8)" "$(grep -c 'INFO-8' $NOTE)"
chk B13e "note: dDIG repairs registered" "$(grep -c 'condition on the target' $NOTE)"
absent B14 "note: no em-dashes (style contract)" "$(grep -c '—' $NOTE)"

echo "---"
node numeric_checks.js || fails=$((fails+1))
echo "---"
if [ $fails -eq 0 ]; then echo "V2 ALL CHECKS PASS"; else echo "V2 FAILURES: $fails"; exit 1; fi
