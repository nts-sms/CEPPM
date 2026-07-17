"""
3D self-consistent bifurcation analysis for the CEPPM model.
=============================================================

Unlike the earlier 2D appendix script (bifurcation_analysis_h.py), which
fixed r_eff and alpha_eff at constant baseline values (r_base, alpha_base,
i.e. treated the EGT strategy layer as switched off), this script solves
the full coupled system self-consistently: at every stoat density S, the
behavioural strategy mix is assumed to sit at its replicator-dynamics
equilibrium x*(S), and that x*(S) is fed into r_eff(x) and alpha_eff(x)
before the K/S fixed points are found.

This produces "h_dynamic" (~0.1685) and an eradication threshold on the
coexistence branch (~0.324), which are DIFFERENT quantities from the
paper's own h_survival (0.1654) and h_erad (0.32):

  - h_survival (paper, Eqs 10-11): feasibility threshold. x is chosen by
    argmax_x[r_eff(x)/alpha_eff(x)] -- the best-case strategy bound, i.e.
    "does there exist a behavioural mix under which kiwi growth is
    possible". This is NOT reproduced here; it's a different construction
    (optimization over x, not a dynamical equilibrium in x).

  - h_erad (paper): the stoat-eradication threshold -- a property of the
    STOAT subsystem only, unrelated to kiwi growth feasibility.

  - h_dynamic (this script): realized threshold. x tracks the actual
    replicator equilibrium x*(S) induced by stoat density, and growth is
    evaluated along that self-consistent trajectory, not at the optimum.

Do not conflate h_dynamic with either h_survival or h_erad from the paper.
"""

import numpy as np
from scipy.optimize import brentq, fsolve
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------
# 1. Model parameters (from kiwi_conservation_analysis_v7.py)
# ---------------------------------------------------------------------
a, b = 0.35, 0.55      # base payoffs, open-foraging row
c, d = 0.45, 0.20      # base payoffs, cover-foraging row

r_base     = 0.05
alpha_base = 0.044
beta       = 0.0175
delta      = 0.35
r_stoat    = 0.60
S_max      = 8.0
K_max      = 150.0
H          = 0.1       # Holling handling time
kappa_learn_baseline = 1.5

open_vuln, cover_vuln = 1.3, 0.7
sig_k, S_mid          = 3.0, 1.0     # dynamic-payoff sigmoid params
open_max, cov_max     = 0.30, 0.15   # dynamic-payoff penalty caps

# Static ESS (base payoffs, S-independent)
denom_static = (a - b) + (d - c)
x_star_static = (d - b) / denom_static     # = 7/9 = 0.77778
PI_BAR_ESS = x_star_static * (x_star_static * a + (1 - x_star_static) * b) \
             + (1 - x_star_static) * (x_star_static * c + (1 - x_star_static) * d)

H_CRIT = r_stoat - delta                    # = 0.25, exact


# ---------------------------------------------------------------------
# 2. Self-consistent replicator equilibrium x*(S)
# ---------------------------------------------------------------------
def x_star_of_S(S):
    """
    Interior equilibrium of dx/dt=0 under DYNAMIC (S-dependent) payoffs.
    Derived analytically (open_pen and cov_pen cancel out of a-b and d-c
    respectively, since the sigmoid penalty is additive within each row):

        x*(S) = 7/9 - (1/3) * sigmoid(S)

    where sigmoid(S) = 1 / (1 + exp(-sig_k*(S - S_mid))).
    Verified by direct substitution into the dynamic payoff matrix.
    """
    sig = 1.0 / (1.0 + np.exp(-sig_k * (S - S_mid)))
    return x_star_static - (1.0 / 3.0) * sig


def r_eff_of_x(x):
    piA = x * a + (1 - x) * b
    piB = x * c + (1 - x) * d
    avg_reproduction = x * piA + (1 - x) * piB
    return r_base * (avg_reproduction / PI_BAR_ESS)


def alpha_eff_of_x(x):
    avg_vulnerability = x * open_vuln + (1 - x) * cover_vuln
    return alpha_base * avg_vulnerability


def f_K(alpha_eff, K):
    return (alpha_eff * K) / (1.0 + alpha_eff * H * K)


# ---------------------------------------------------------------------
# 3. Branch B (K=0, S-floor branch) -- closed form
# ---------------------------------------------------------------------
def S_branch_B(h):
    """K=0 => f_K=0, so dS/dt = r_stoat*S*(1-S/S_max) - delta*S - h*S = 0."""
    S = S_max * (1.0 - (delta + h) / r_stoat)
    return max(S, 0.0)


# ---------------------------------------------------------------------
# 4. Branch D (coexistence, K>0 & S>0) -- self-consistent root-find
# ---------------------------------------------------------------------
def branch_D_residual(S, h):
    """
    Reduce the 2-equation (K,S) fixed-point system to a single residual in S:
      (i)  dS/dt=0 (S>0) pins f_K(alpha_eff,K) = f_target(S,h)
      (ii) invert f_K(.) for K given f_target
      (iii) plug into dK/dt=0 and return the residual
    Returns None if no physically valid K exists at this S.
    """
    if S <= 1e-9 or S >= S_max:
        return None
    x = x_star_of_S(S)
    r_e = r_eff_of_x(x)
    al_e = alpha_eff_of_x(x)

    f_target = (delta + h - r_stoat * (1.0 - S / S_max)) / beta
    if f_target <= 1e-9:
        return None
    denom = al_e * (1.0 - f_target * H)
    if denom <= 1e-9:
        return None
    K = f_target / denom
    if K <= 0 or K > 5 * K_max:
        return None

    lhs = r_e * K * (1.0 - K / K_max)
    rhs = f_target * S
    return lhs - rhs, K


def solve_branch_D(h, S_grid=None):
    """Scan S for a sign change in the residual, then bisect. Returns (S*, K*) or None."""
    if S_grid is None:
        S_grid = np.linspace(1e-3, S_max - 1e-3, 400)
    prev_S, prev_val = None, None
    for S in S_grid:
        out = branch_D_residual(S, h)
        if out is None:
            prev_S, prev_val = None, None
            continue
        val, _ = out
        if prev_val is not None and np.sign(val) != np.sign(prev_val):
            def resid_only(SS):
                o = branch_D_residual(SS, h)
                return o[0] if o is not None else np.nan
            try:
                S_root = brentq(resid_only, prev_S, S, xtol=1e-8)
                out_root = branch_D_residual(S_root, h)
                if out_root is not None:
                    return S_root, out_root[1]
            except Exception:
                pass
        prev_S, prev_val = S, val
    return None


# ---------------------------------------------------------------------
# 5. h_dynamic: transcritical point where Branch D merges into Branch B
#    (K-direction stability of Branch B changes sign)
# ---------------------------------------------------------------------
def branch_B_K_growth_rate(h):
    """
    Linear (small-K) growth rate of K on Branch B:
    d(dK/dt)/dK |_{K=0} = r_eff(x_B) - alpha_eff(x_B) * S_B(h)
    Positive => K=0 unstable in the K-direction => coexistence branch exists.
    """
    S_B = S_branch_B(h)
    x_B = x_star_of_S(S_B)
    return r_eff_of_x(x_B) - alpha_eff_of_x(x_B) * S_B


def find_h_dynamic():
    h_lo, h_hi = 0.05, H_CRIT - 1e-6
    f_lo = branch_B_K_growth_rate(h_lo)
    f_hi = branch_B_K_growth_rate(h_hi)
    if np.sign(f_lo) == np.sign(f_hi):
        raise RuntimeError("No sign change found for h_dynamic in search range.")
    return brentq(branch_B_K_growth_rate, h_lo, h_hi, xtol=1e-8)


def find_h_erad_3D(h_dynamic):
    """Sweep h upward from just above h_crit until Branch D's S* root vanishes."""
    h_grid = np.linspace(H_CRIT + 1e-4, 0.5, 2000)
    last_good = None
    for h in h_grid:
        sol = solve_branch_D(h)
        if sol is None:
            if last_good is not None:
                return 0.5 * (last_good + h)
            continue
        last_good = h
    return last_good


# ---------------------------------------------------------------------
# 6. Full 3D Jacobian (finite-difference) for stability / Hopf checks
# ---------------------------------------------------------------------
def dynamic_payoff_matrix(S):
    sig = 1.0 / (1.0 + np.exp(-sig_k * (S - S_mid)))
    open_pen, cov_pen = open_max * sig, cov_max * sig
    return np.array([
        [max(a - open_pen, 0.01), max(b - open_pen, 0.01)],
        [max(c - cov_pen, 0.01),  max(d - cov_pen, 0.01)]
    ])


def full_rhs(y, h, kappa):
    x, K, S = y
    x = np.clip(x, 1e-6, 1 - 1e-6)
    K = max(K, 0.0)
    S = max(S, 0.0)

    pm = dynamic_payoff_matrix(S)
    piA = x * pm[0, 0] + (1 - x) * pm[0, 1]
    piB = x * pm[1, 0] + (1 - x) * pm[1, 1]
    pi_avg = x * piA + (1 - x) * piB
    dxdt = kappa * x * (piA - pi_avg)

    r_e = r_eff_of_x(x)
    al_e = alpha_eff_of_x(x)
    fK = f_K(al_e, K)
    dKdt = r_e * K * (1 - K / K_max) - fK * S

    natural = r_stoat * S * (1 - S / S_max) + beta * fK * S - delta * S
    dSdt = natural - h * S
    return np.array([dxdt, dKdt, dSdt])


def jacobian_fd(y, h, kappa, eps=1e-6):
    J = np.zeros((3, 3))
    f0 = full_rhs(y, h, kappa)
    for i in range(3):
        y_pert = np.array(y, dtype=float)
        y_pert[i] += eps
        f1 = full_rhs(y_pert, h, kappa)
        J[:, i] = (f1 - f0) / eps
    return J


def max_real_eigenvalue_branch_D(h, kappa):
    sol = solve_branch_D(h)
    if sol is None:
        return None
    S_star, K_star = sol
    x_star = x_star_of_S(S_star)
    J = jacobian_fd([x_star, K_star, S_star], h, kappa)
    eigvals = np.linalg.eigvals(J)
    return float(np.max(eigvals.real))


# ---------------------------------------------------------------------
# 7. Run the analysis
# ---------------------------------------------------------------------
if __name__ == "__main__":
    print("=" * 70)
    print("3D SELF-CONSISTENT BIFURCATION ANALYSIS")
    print("=" * 70)

    h_dynamic = find_h_dynamic()
    h_erad_3D = find_h_erad_3D(h_dynamic)

    print(f"\nh_crit          = {H_CRIT:.4f}   (exact, r_stoat - delta)")
    print(f"h_dynamic       = {h_dynamic:.4f}  (Branch B -> Branch D transcritical, "
          f"self-consistent x*(S))")
    print(f"h_erad (3D)     = {h_erad_3D:.4f}  (Branch D coexistence S* -> 0)")
    print("\nFor comparison (NOT reproduced by this script, see paper/appendix):")
    print("  h_survival (paper, Eqs 10-11, x_critical argmax bound) = 0.1654")
    print("  h_erad     (paper, stoat-only threshold)                = ~0.32")
    print("  h_fold     (2D appendix, fixed alpha_base, superseded)  = 0.1648")

    # ---- Sweep h to build the 3-panel state diagram --------------------
    h_vals = np.linspace(0.0, 0.45, 300)
    K_D, S_D, x_D, h_D_valid = [], [], [], []
    K_B, S_B_arr, x_B_arr, h_B_valid = [], [], [], []

    for h in h_vals:
        # Branch B always defined for h < h_crit (S_B > 0)
        S_B_val = S_branch_B(h)
        if S_B_val > 1e-9:
            h_B_valid.append(h)
            S_B_arr.append(S_B_val)
            K_B.append(0.0)
            x_B_arr.append(x_star_of_S(S_B_val))

        sol = solve_branch_D(h)
        if sol is not None:
            S_star, K_star = sol
            h_D_valid.append(h)
            S_D.append(S_star)
            K_D.append(K_star)
            x_D.append(x_star_of_S(S_star))

    # Branch C: K=K_max, S=0 (always exists for all h; stability changes at h_erad_3D)
    h_C = h_vals
    K_C = np.full_like(h_C, K_max)
    S_C = np.zeros_like(h_C)

    # ---- Figure 1: 3-panel state diagram --------------------------------
    fig, axes = plt.subplots(3, 1, figsize=(8, 11), sharex=True)

    axes[0].plot(h_C, K_C, color='#2ca02c', lw=2, label='Branch C (K=K_max, S=0)')
    axes[0].plot(h_B_valid, K_B, color='#d62728', lw=2, ls='--', label='Branch B (K=0)')
    axes[0].plot(h_D_valid, K_D, color='#1f77b4', lw=2, label='Branch D (coexistence)')
    axes[0].set_ylabel('K*')
    axes[0].legend(fontsize=8, loc='upper right')
    axes[0].set_title('3D self-consistent bifurcation diagram (x*(S) coupling)')

    axes[1].plot(h_C, S_C, color='#2ca02c', lw=2)
    axes[1].plot(h_B_valid, S_B_arr, color='#d62728', lw=2, ls='--')
    axes[1].plot(h_D_valid, S_D, color='#1f77b4', lw=2)
    axes[1].set_ylabel('S*')

    axes[2].plot(h_B_valid, [v * 100 for v in x_B_arr], color='#d62728', lw=2, ls='--')
    axes[2].plot(h_D_valid, [v * 100 for v in x_D], color='#1f77b4', lw=2)
    axes[2].axhline(x_star_static * 100, color='gray', lw=1, ls=':', label='static x*=77.78%')
    axes[2].set_ylabel('x* (%)')
    axes[2].set_xlabel('harvest rate h')
    axes[2].legend(fontsize=8, loc='lower right')

    for ax in axes:
        ax.axvline(H_CRIT, color='k', lw=0.8, ls=':')
        ax.axvline(h_dynamic, color='purple', lw=0.8, ls=':')
        ax.axvline(h_erad_3D, color='darkorange', lw=0.8, ls=':')

    axes[0].text(h_dynamic, axes[0].get_ylim()[1] * 0.9, ' h_dynamic', color='purple', fontsize=8)
    axes[0].text(H_CRIT, axes[0].get_ylim()[1] * 0.75, ' h_crit', color='k', fontsize=8)
    axes[0].text(h_erad_3D, axes[0].get_ylim()[1] * 0.6, ' h_erad(3D)', color='darkorange', fontsize=8)

    fig.tight_layout()
    fig.savefig('/mnt/user-data/outputs/3D_state_diagram_3panel_selfconsistent.png', dpi=150)
    print("\nSaved: 3D_state_diagram_3panel_selfconsistent.png")

    # ---- Figure 2: eigenvalue heatmap (h x kappa_learn) on Branch D -----
    h_heat = np.linspace(h_dynamic + 0.005, h_erad_3D - 0.005, 40)
    kappa_heat = np.logspace(-4, 3, 40)  # 0.0001 to 1000
    heat = np.full((len(kappa_heat), len(h_heat)), np.nan)

    for j, h in enumerate(h_heat):
        for i, kappa in enumerate(kappa_heat):
            val = max_real_eigenvalue_branch_D(h, kappa)
            if val is not None:
                heat[i, j] = val

    fig2, ax2 = plt.subplots(figsize=(8, 6))
    im = ax2.pcolormesh(h_heat, kappa_heat, heat, shading='auto', cmap='RdBu_r',
                         vmin=-np.nanmax(np.abs(heat)), vmax=np.nanmax(np.abs(heat)))
    ax2.set_yscale('log')
    ax2.set_xlabel('harvest rate h')
    ax2.set_ylabel('kappa_learn (log scale)')
    ax2.set_title('max Re(eigenvalue) on Branch D — no Hopf if never > 0')
    fig2.colorbar(im, ax=ax2, label='max Re(eigenvalue)')
    fig2.tight_layout()
    fig2.savefig('/mnt/user-data/outputs/eigenvalue_heatmap_selfconsistent.png', dpi=150)
    print("Saved: eigenvalue_heatmap_selfconsistent.png")

    max_overall = np.nanmax(heat)
    print(f"\nMax Re(eigenvalue) found across full (h, kappa_learn) grid: {max_overall:.6f}")
    print("(Negative/zero everywhere => no oscillatory (Hopf) instability under "
          "self-consistent x*(S) coupling, consistent with earlier 2D result.)")
