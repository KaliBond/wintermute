#!/bin/bash
# Verifier v1 structural checks for the CAMS informational-perspective research note
F=../../../papers/CAMS_Informational_Perspective_Research_Note.md
fail=0
chk(){ if eval "$2"; then echo "PASS: $1"; else echo "FAIL: $1"; fail=1; fi; }
chk "1. deliverable exists and is non-trivial (>15KB)" "[ -s $F ] && [ $(wc -c < $F) -gt 15000 ]"
chk "2a. lineage anchor Shannon present" "grep -q 'Shannon, C. E. (1948)' $F"
chk "2b. lineage anchor Wiener present" "grep -q 'Wiener, N. (1948)' $F"
chk "2c. lineage anchor van der Leeuw & Dirks 2025 present" "grep -q 'van der Leeuw, S., & Dirks, G. (2025)' $F"
chk "2d. lineage anchor Hayek/Galbraith present" "grep -q 'Hayek, F. A. (1945)' $F && grep -q 'Galbraith, J. R. (1974)' $F"
chk "3. canonical eta_loop formula matches framework (B_Lore x B_Archive)/S_Hands" "grep -q 'B_Lore × B_Archive)/S_Hands' $F"
chk "4. no invented V_i variant (no redefinition of node value)" "! grep -qE 'V_i ?=[^ ]' $F"
chk "5a. lab continuity: TE asymmetry numbers cited" "grep -q '0.156' $F && grep -q '0.108' $F"
chk "5b. lab continuity: Archive silencing gain cited" "grep -q '0.031 bits' $F"
chk "6. >=3 pre-registrable hypotheses with bars" "[ $(grep -c '^\*\*INFO-' $F) -ge 3 ] && grep -q 'Bar:' $F"
chk "7a. red line: no 8-node proven-minimum claim" "grep -qi 'not claim the eight-node architecture is minimal' $F"
chk "7b. red line: Lore/Archive distinctness stated" "grep -q 'Lore and Archive are treated as distinct' $F"
chk "7c. critique: James-Barnett-Crutchfield caveat included" "grep -q 'James, R. G., Barnett, N., & Crutchfield' $F"
echo "---"; [ $fail -eq 0 ] && echo "ALL CHECKS PASS" || echo "CHECKS FAILED"
exit $fail
