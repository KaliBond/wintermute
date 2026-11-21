"""SA-PEP-001 Revision 3: shared feature and model code (R3.1, R3.3, R3.5, R3.6).

Data layout: X has shape (n_eras, 8, 4); nodes in NODES order, attributes
in ATTRS order (C, K, A, S), values on the 0-10 scale.

Canonical bond formula (v0.2 item 2):
    q_i  = (0.6 C_i + 0.4 A_i) / 10
    w_ij = sqrt(q_i q_j) * 2^(-(S_i + S_j)/10),  i != j
    B_i  = mean of the 7 bonds incident on i
    V_i  = C_i + K_i + A_i/2 - S_i

Feature sets (equal size, 16, for CAMS and every random partition):
    8 node bond strengths B + 8 V values, computed on (pseudo-)nodes.
Aggregate-only baseline (4 features): system means of C, K, A, S.

Model: L2-penalised logistic regression, intercept unpenalised, features
standardised with training-fold statistics only, penalty 0.5*LAMBDA*||w||^2
on the summed log-likelihood (equivalent to C = 1/LAMBDA in common
libraries).  Fitted by Newton-Raphson (strictly convex problem, so the
solution is unique and deterministic), batched over feature sets.

Evaluation: out-of-sample Brier score from K-fold cross-validation with
folds grouped by society (cluster CV).  All feature sets in one replication
use the same folds, so comparisons are paired.

Decision rule (R3.6), implemented in decide():
    pct  = share of random partitions whose Brier is worse (higher) than
           CAMS's (ties count one half)
    O1   if pct > 0.95 AND Brier_CAMS < Brier_aggregate
    O2   else if median Brier of random partitions < Brier_aggregate
    O3   otherwise
"""
import numpy as np

NODES = ["Helm", "Shield", "Lore", "Stewards", "Craft", "Hands", "Archive", "Flow"]
ATTRS = ["C", "K", "A", "S"]
C_, K_, A_, S_ = 0, 1, 2, 3
CAMS_PARTITION = np.tile(np.arange(8)[:, None], (1, 4))  # identity

LAMBDA = 1.0          # fixed L2 strength (to be pre-registered)
K_FOLDS = 5           # cluster-CV folds (to be pre-registered)
NEWTON_MAX_IT = 50
NEWTON_TOL = 1e-10
PCT_THRESHOLD = 0.95


# ---------- features ----------

def apply_partition(X, P):
    """X (n,8,4), P (8,4) -> pseudo-node matrix (n,8,4)."""
    P = np.asarray(P)
    return np.stack([X[:, P[:, a], a] for a in range(4)], axis=2)


def bond_matrix(M):
    """M (n,8,4) -> w (n,8,8) with zero diagonal."""
    q = (0.6 * M[..., C_] + 0.4 * M[..., A_]) / 10.0
    q = np.clip(q, 0.0, None)
    S = M[..., S_]
    w = np.sqrt(q[:, :, None] * q[:, None, :]) * 2.0 ** (-(S[:, :, None] + S[:, None, :]) / 10.0)
    idx = np.arange(8)
    w[:, idx, idx] = 0.0
    return w


def bond_strength(M):
    return bond_matrix(M).sum(axis=2) / 7.0


def v_values(M):
    return M[..., C_] + M[..., K_] + M[..., A_] / 2.0 - M[..., S_]


def morph_features(X, P):
    M = apply_partition(X, P)
    return np.concatenate([bond_strength(M), v_values(M)], axis=1)  # (n,16)


def aggregate_features(X):
    return X.mean(axis=1)  # (n,4) system means of C,K,A,S


# ---------- model ----------

def _sigmoid(z):
    return 0.5 * (1.0 + np.tanh(0.5 * z))


def fit_logistic_batched(Xtr, ytr, lam=LAMBDA):
    """Xtr (B,n,d) standardised, ytr (n,). Returns beta (B,d+1), intercept first."""
    B, n, d = Xtr.shape
    Z = np.concatenate([np.ones((B, n, 1)), Xtr], axis=2)
    beta = np.zeros((B, d + 1))
    pen = np.full(d + 1, lam)
    pen[0] = 0.0
    for _ in range(NEWTON_MAX_IT):
        p = _sigmoid(np.einsum("bnd,bd->bn", Z, beta))
        g = np.einsum("bnd,bn->bd", Z, ytr[None, :] - p) - pen * beta
        W = p * (1 - p)
        H = np.einsum("bnd,bn,bne->bde", Z, W, Z) + np.diag(pen)[None]
        step = np.linalg.solve(H, g[..., None])[..., 0]
        beta = beta + step
        if np.max(np.abs(step)) < NEWTON_TOL:
            break
    return beta


def cluster_folds(groups, k, rng):
    """Assign whole societies to k folds at random (balanced counts)."""
    soc = np.unique(groups)
    perm = rng.permutation(soc)
    fold_of = {s: i % k for i, s in enumerate(perm)}
    return np.array([fold_of[g] for g in groups])


def cv_brier(F, y, fold_id, lam=LAMBDA):
    """F (B,n,d) feature sets, y (n,), fold_id (n,). Returns OOS Brier (B,)."""
    B, n, d = F.shape
    pred = np.zeros((B, n))
    for k in np.unique(fold_id):
        te = fold_id == k
        tr = ~te
        mu = F[:, tr].mean(axis=1, keepdims=True)
        sd = F[:, tr].std(axis=1, keepdims=True)
        sd = np.where(sd < 1e-12, 1.0, sd)
        Ftr = (F[:, tr] - mu) / sd
        Fte = (F[:, te] - mu) / sd
        beta = fit_logistic_batched(Ftr, y[tr], lam)
        pred[:, te] = _sigmoid(beta[:, :1] + np.einsum("bnd,bd->bn", Fte, beta[:, 1:]))
    return ((pred - y[None, :]) ** 2).mean(axis=1)


def evaluate(X, y, groups, partitions, rng, k=K_FOLDS, lam=LAMBDA):
    """Brier for CAMS, the aggregate baseline and each random partition, on shared folds."""
    fold_id = cluster_folds(groups, k, rng)
    F = np.stack([morph_features(X, CAMS_PARTITION)] + [morph_features(X, P) for P in partitions])
    briers = cv_brier(F, y, fold_id, lam)
    agg = cv_brier(aggregate_features(X)[None], y, fold_id, lam)[0]
    base = np.mean((y - np.array([y[fold_id != f].mean() for f in fold_id])) ** 2)
    return {"cams": briers[0], "parts": briers[1:], "agg": agg, "base_rate": base}


def percentile(cams, parts):
    parts = np.asarray(parts)
    return (np.sum(parts > cams) + 0.5 * np.sum(parts == cams)) / parts.size


def decide(cams, agg, parts, thr=PCT_THRESHOLD):
    pct = percentile(cams, parts)
    if pct > thr and cams < agg:
        return "O1", pct
    if np.median(parts) < agg:
        return "O2", pct
    return "O3", pct
