"""
Dual-model test for the stampede-as-phase-transition claim.

Model A: Explosive Kuramoto. Euler-Maruyama SDE integration (fixed dt, noise ~ sqrt(dt)).
         Coupling correlated with |omega_i| (Gomez-Gardenes mechanism) so a genuine
         first-order transition with hysteresis CAN appear -- or CAN fail to.
         Forward and backward sweeps of the control parameter are run separately
         and overlaid. Hysteresis is a MEASURED output, not an assumption.

Model B: Branching-process / probabilistic contagion (the honest alternative for a
         one-shot cascade, in the spirit of Poel et al. 2022). Stress modulates the
         branching ratio; we report cascade-size distributions and distance from the
         critical branching ratio b=1.

Falsification logic: if forward and backward sweep paths coincide (no enclosed area),
the first-order/hysteresis reading is NOT supported and the second-order reading stands.
"""

import numpy as np
import matplotlib.pyplot as plt

plt.rcParams.update({
    "figure.facecolor": "#0a0a0a", "axes.facecolor": "#0a0a0a",
    "savefig.facecolor": "#0a0a0a", "text.color": "#e8e8e8",
    "axes.labelcolor": "#e8e8e8", "xtick.color": "#e8e8e8",
    "ytick.color": "#e8e8e8", "axes.edgecolor": "#555555",
    "font.size": 11,
})
CYAN, GOLD, RED = "#00f0ff", "#ffcc00", "#ff3366"

rng = np.random.default_rng(20260606)


# ----------------------------------------------------------------------
# Model A: explosive Kuramoto via Euler-Maruyama
# ----------------------------------------------------------------------
def order_parameter(theta):
    return np.abs(np.mean(np.exp(1j * theta)))


def kuramoto_equilibrium_r(K_global, N=150, D=0.05, dt=0.01, t_relax=20.0,
                           t_measure=8.0, explosive=True, theta0=None, omega=None):
    """Integrate to steady state at fixed coupling; return time-averaged r and final theta.

    explosive=True  -> per-oscillator coupling K_i = K_global * |omega_i| / mean(|omega|)
                       (frequency-correlated coupling => candidate first-order transition)
    explosive=False -> uniform coupling (standard => second-order transition)
    """
    if omega is None:
        omega = rng.standard_normal(N)          # unimodal g(omega), N(0,1)
    if theta0 is None:
        theta = rng.uniform(0, 2 * np.pi, N)
    else:
        theta = theta0.copy()

    if explosive:
        w = np.abs(omega)
        Kfac = w / np.mean(w)                    # coupling weight per oscillator
    else:
        Kfac = np.ones(N)

    n_relax = int(t_relax / dt)
    n_meas = int(t_measure / dt)
    sqrt_dt = np.sqrt(dt)

    for _ in range(n_relax):
        diff = theta[None, :] - theta[:, None]
        coupling = (K_global / N) * Kfac * np.sum(np.sin(diff), axis=1)
        dtheta = omega + coupling
        theta = theta + dtheta * dt + np.sqrt(2 * D) * sqrt_dt * rng.standard_normal(N)

    rs = np.empty(n_meas)
    for k in range(n_meas):
        diff = theta[None, :] - theta[:, None]
        coupling = (K_global / N) * Kfac * np.sum(np.sin(diff), axis=1)
        dtheta = omega + coupling
        theta = theta + dtheta * dt + np.sqrt(2 * D) * sqrt_dt * rng.standard_normal(N)
        rs[k] = order_parameter(theta)
    return rs.mean(), theta, omega


def hysteresis_sweep(K_values, explosive=True, N=150, D=0.05):
    """Forward sweep (increasing K) then backward sweep (decreasing K), carrying
    the final state forward as the initial condition for the next K (adiabatic sweep).
    This is the configuration in which hysteresis, if it exists, will show."""
    # shared omega across the whole sweep so the comparison is clean
    omega = rng.standard_normal(N)

    # forward
    r_fwd = []
    theta = rng.uniform(0, 2 * np.pi, N)
    for K in K_values:
        r, theta, _ = kuramoto_equilibrium_r(K, N=N, D=D, explosive=explosive,
                                             theta0=theta, omega=omega)
        r_fwd.append(r)
    # backward, starting from the synchronised end state
    r_bwd = []
    for K in K_values[::-1]:
        r, theta, _ = kuramoto_equilibrium_r(K, N=N, D=D, explosive=explosive,
                                             theta0=theta, omega=omega)
        r_bwd.append(r)
    r_bwd = r_bwd[::-1]
    return np.array(r_fwd), np.array(r_bwd)


# ----------------------------------------------------------------------
# Model B: branching-process contagion (one-shot escape cascade)
# ----------------------------------------------------------------------
def branching_cascade(branching_ratio, n_trials=4000, max_gen=200, seed_size=1,
                      herd_size=500):
    """Galton-Watson cascade capped at a finite herd. Each alert animal recruits
    Poisson(branching_ratio) others, but the cascade cannot exceed herd_size
    (finite-population saturation). Returns array of total cascade sizes."""
    local = np.random.default_rng(seed_size * 7 + int(branching_ratio * 1000))
    sizes = np.empty(n_trials)
    for t in range(n_trials):
        active = seed_size
        total = seed_size
        gen = 0
        while active > 0 and gen < max_gen and total < herd_size:
            active = min(active, herd_size - total)
            offspring = local.poisson(branching_ratio, size=active).sum()
            total += offspring
            active = offspring
            gen += 1
        sizes[t] = min(total, herd_size)
    return sizes


def stress_to_branching(stress):
    """Map a stress scalar in [0,1] to a branching ratio that crosses b=1.
    Subcritical at low stress, supercritical at high stress."""
    return 0.6 + 0.7 * stress


# ----------------------------------------------------------------------
# Precursor analysis on an r time series
# ----------------------------------------------------------------------
def precursors(series, window=200):
    """Rolling variance and lag-1 autocorrelation."""
    s = np.asarray(series)
    var = np.array([s[max(0, i - window):i + 1].var() for i in range(len(s))])
    ac1 = np.full(len(s), np.nan)
    for i in range(window, len(s)):
        seg = s[i - window:i + 1]
        seg = seg - seg.mean()
        denom = np.sum(seg * seg)
        ac1[i] = np.sum(seg[1:] * seg[:-1]) / denom if denom > 1e-12 else np.nan
    return var, ac1


# ======================================================================
# RUN
# ======================================================================
print("=" * 64)
print("MODEL A: hysteresis test (explosive vs uniform coupling)")
print("=" * 64)
K_vals = np.linspace(0.2, 4.0, 16)

r_fwd_exp, r_bwd_exp = hysteresis_sweep(K_vals, explosive=True, D=0.05)
r_fwd_uni, r_bwd_uni = hysteresis_sweep(K_vals, explosive=False, D=0.05)

area_exp = np.trapezoid(np.abs(r_fwd_exp - r_bwd_exp), K_vals)
area_uni = np.trapezoid(np.abs(r_fwd_uni - r_bwd_uni), K_vals)
print(f"Enclosed hysteresis area (explosive, K_i~|omega_i|): {area_exp:.4f}")
print(f"Enclosed hysteresis area (uniform coupling)        : {area_uni:.4f}")
print(f"Max forward-backward gap (explosive): {np.max(np.abs(r_fwd_exp-r_bwd_exp)):.4f}")
print(f"Max forward-backward gap (uniform)  : {np.max(np.abs(r_fwd_uni-r_bwd_uni)):.4f}")

verdict = ("SUPPORTED: forward/backward paths diverge -> bistability/hysteresis present"
           if area_exp > 0.10 else
           "NOT SUPPORTED: paths effectively coincide -> no hysteresis at this D/N")
print(f"\nFirst-order (hysteresis) reading, explosive model: {verdict}")

print("\n" + "=" * 64)
print("MODEL B: branching cascade (one-shot escape)")
print("=" * 64)
for stress in [0.2, 0.5, 0.57, 0.8]:
    b = stress_to_branching(stress)
    sizes = branching_cascade(b, n_trials=3000)
    finite = sizes[sizes < 5000]
    print(f"stress={stress:.2f}  b={b:.3f}  mean cascade={sizes.mean():8.1f}  "
          f"median={np.median(sizes):5.0f}  P(size>50)={np.mean(sizes>50):.3f}")

# ---- figure 1: hysteresis ----
fig, ax = plt.subplots(1, 2, figsize=(12, 5))
ax[0].plot(K_vals, r_fwd_exp, "-o", color=CYAN, ms=4, label="forward (K rising)")
ax[0].plot(K_vals, r_bwd_exp, "-s", color=GOLD, ms=4, label="backward (K falling)")
ax[0].fill_between(K_vals, r_fwd_exp, r_bwd_exp, color=RED, alpha=0.18)
ax[0].set_title(f"Explosive coupling  $K_i\\propto|\\omega_i|$\nloop area={area_exp:.3f}",
                color="#e8e8e8")
ax[0].set_xlabel("global coupling K (≈ effective social coupling / stress)")
ax[0].set_ylabel("order parameter r  (≈ Coherence)")
ax[0].legend(facecolor="#1a1a1a", labelcolor="#e8e8e8", framealpha=0.6)

ax[1].plot(K_vals, r_fwd_uni, "-o", color=CYAN, ms=4, label="forward")
ax[1].plot(K_vals, r_bwd_uni, "-s", color=GOLD, ms=4, label="backward")
ax[1].fill_between(K_vals, r_fwd_uni, r_bwd_uni, color=RED, alpha=0.18)
ax[1].set_title(f"Uniform coupling (control)\nloop area={area_uni:.3f}",
                color="#e8e8e8")
ax[1].set_xlabel("global coupling K")
ax[1].set_ylabel("order parameter r")
ax[1].legend(facecolor="#1a1a1a", labelcolor="#e8e8e8", framealpha=0.6)
plt.tight_layout()
plt.savefig("/home/claude/explosive_kuramoto_hysteresis.png", dpi=150)
print("\nsaved explosive_kuramoto_hysteresis.png")

# ---- figure 2: branching cascade-size distributions ----
fig, ax = plt.subplots(figsize=(8, 5))
for stress, col in [(0.2, CYAN), (0.5, GOLD), (0.8, RED)]:
    b = stress_to_branching(stress)
    sizes = branching_cascade(b, n_trials=5000)
    sizes = sizes[sizes < 5000]
    ax.hist(sizes, bins=np.logspace(0, 2.7, 40), histtype="step", color=col, lw=2,
            label=f"stress={stress:.1f}  b={b:.2f}")
ax.set_xscale("log"); ax.set_yscale("log")
ax.set_xlabel("cascade size (animals fleeing)")
ax.set_ylabel("frequency")
ax.set_title("Branching-process escape cascades: size distribution vs stress",
             color="#e8e8e8")
ax.legend(facecolor="#1a1a1a", labelcolor="#e8e8e8", framealpha=0.6)
plt.tight_layout()
plt.savefig("/home/claude/branching_contagion.png", dpi=150)
print("saved branching_contagion.png")

# ---- precursor demonstration: lag-1 AC and variance rising into a transition ----
print("\n" + "=" * 64)
print("PRECURSOR CHECK: branching ratio ramped toward criticality")
print("=" * 64)
stress_ramp = np.linspace(0.1, 0.62, 60)   # ends just below b=1
series = []
for st in stress_ramp:
    b = stress_to_branching(st)
    series.append(branching_cascade(b, n_trials=120, herd_size=500).mean())
series = np.array(series)
var, ac1 = precursors(series, window=12)
# correlation of precursor with proximity to criticality
prox = stress_ramp
valid = ~np.isnan(ac1)
import numpy as _np
r_ac = _np.corrcoef(prox[valid], ac1[valid])[0, 1]
r_var = _np.corrcoef(prox, var)[0, 1]
print(f"corr(stress, lag-1 autocorr) = {r_ac:+.3f}")
print(f"corr(stress, variance)       = {r_var:+.3f}")
print("(positive => early-warning signatures strengthen approaching criticality)")

import csv
with open("/home/claude/precursor_series.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["stress", "branching_ratio", "mean_cascade", "roll_variance", "lag1_autocorr"])
    for i, st in enumerate(stress_ramp):
        w.writerow([f"{st:.4f}", f"{stress_to_branching(st):.4f}",
                    f"{series[i]:.3f}", f"{var[i]:.4f}",
                    "" if _np.isnan(ac1[i]) else f"{ac1[i]:.4f}"])
print("saved precursor_series.csv")
