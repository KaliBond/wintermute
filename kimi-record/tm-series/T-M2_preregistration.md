# Pre-Registration T-M2: Displacement Morphology as Predictor of Failure Morphology

**Date:** 2026-10-16 · **Status:** REGISTERED before outcome computation · Dataset: JUNO_Unified_Dataset.csv (canonical verification PASSED, drift 3.6e-15) · Same episode extraction as T-M1 (201 episodes, 149 usable with LOEO typicality, 33 societies).

## Rationale

T-M1 result: displacement morphology is society-level character (stability gate PASS, p=0.004) but does not predict recovery (AUC 0.492, FAIL). Reframed hypothesis: morphology predicts the *morphology of failure* — which function absorbs the damage — rather than recovery dynamics. This matches the June/July finding that the 8-node layer is required for failure attribution (residual layer = differential diagnosis) while aggregate recovery rides the common mode.

## Operational definitions

- Episodes and displacement signatures: identical to T-M1 (no re-derivation).
- **Failure locus** of episode e: the node with minimum V during the episode window (ties broken by largest ΔV from pre-onset baseline). Argmin over 8 nodes.
- **Predictors:** (a) society modal failure locus across OTHER episodes (LOEO, no leakage); (b) displacement signature of the episode itself — predicted locus = argmax of stress-uptake signature d_e (does the node that absorbs stress become the node that fails?); (c) rivals: marginal base-rate guess (always predict the globally most frequent locus); V-Dem libdem at onset as continuous predictor is not applicable to 8-class — regime rival tested via per-class logistic on locus∈{Helm, Hands} (the two politically interpretable loci).
- **Primary test T-M2a:** multi-class prediction accuracy of society-modal-locus guess vs permutation null (society-label shuffle, 500 permutations) and vs marginal base rate. Pass: accuracy ≥ 1.5× marginal base rate AND permutation p < 0.05.
- **Secondary T-M2b:** concordance between stress-uptake argmax and failure locus (argmin V): Cohen's κ vs chance. Pass: κ ≥ 0.20.
- **T-M2c:** does failure locus predict recovery (closing the loop — maybe locus, not typicality, carries outcome signal)? Per-locus recovery rates, χ² test. Exploratory unless omnibus p < 0.05.

## Minimum-n gate

Proceed if ≥5 societies have ≥3 episodes each (needed for LOEO modal locus) and ≥100 episodes total. Else exploratory.

## Registered interpretations

- T-M2a PASS → morphology is differential diagnosis: societies fail in characteristic places; the "identity" conjecture survives relocated from recovery to failure-locus.
- T-M2a FAIL + T-M2b PASS → stress-uptake determines failure within episodes but societies don't differ; morphology is universal mechanics, not identity.
- Both FAIL → the residual layer's society-level stability (T-M1c) has no behavioural consequence at episode level; morphology is scoring habit, not structure. Strong restriction on the identity conjecture.
