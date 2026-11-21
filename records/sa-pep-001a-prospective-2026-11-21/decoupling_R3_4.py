"""SA-PEP-001 R3.4: the restated 001a predictor, Helm decoupling (fixed before any R3.4 simulation).

Definitions (canonical bond formula, pipeline_R3.bond_strength):
  BS_i(t)  = mean of the 7 bonds w_ij(t) incident on node i
  dBS_i    = BS_i(t) - BS_i(t-2)                       (change over the predictor window)
  D_Helm   = mean_{j != Helm} dBS_j  -  dBS_Helm
           = -( dBS_Helm - mean_{j != Helm} dBS_j )
Sign convention: D_Helm > 0 means Helm's bond strength FELL relative to the
mean change in the other seven bonds (Helm decoupled).  001a predicts that
higher D_Helm raises the probability of the RIGIDITY class in t+1..t+3.
The same function applied to pseudo-node p of an arbitrary partition gives
the decoupling-style signal used in scenario (c).

Feature-set decision: D_Helm is an exact linear combination of the existing
CAMS window features (the 8 dBS values are already features 17 to 24 of the
R3.1 32-feature set), so it is DERIVABLE and is NOT added as a new feature.
This keeps CAMS and every arbitrary partition at equal complexity (32
features each), which the percentile comparison requires.  The aggregate
baseline (4 means at t and their 4 changes) does not receive it and cannot
express it.
"""
import numpy as np
from pipeline_R3 import apply_partition, bond_strength, CAMS_PARTITION

HELM = 0


def decoupling(X0, X2, P=CAMS_PARTITION, node=HELM):
    """X0 = matrices at t-2, X2 = at t, shape (n,8,4). Returns D for (pseudo-)node `node`."""
    dBS = bond_strength(apply_partition(X2, P)) - bond_strength(apply_partition(X0, P))
    others = np.delete(dBS, node, axis=1).mean(axis=1)
    return others - dBS[:, node]
