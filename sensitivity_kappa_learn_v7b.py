"""
Sensitivity analysis for the behavioural adaptation rate multiplier KAPPA_LEARN
CEPPM Kiwi Conservation Paper — Sierra M. Sharma & Nagaja T. Sanatkumar

CONTEXT (v7 model): payoffs reframed as dimensionless utility values
(reward minus crowding cost). Payoff matrix updated to v7:
[[0.35,0.55],[0.45,0.20]] → x*=7/9=77.78%, f'(x*)=-0.07778
κ_learn=1.5/yr gives EGT local relaxation timescale=5.94yr at x*.

KAPPA_LEARN is NOT directly measured for kiwi. It is bounded only loosely
by Dixon (2015) / Cunningham & Castro (2011)'s documented WITHIN-YEAR
(seasonal) habitat-use shifts -- which give a plausible half-life range of
roughly 2 weeks to ~14 years (the upper bound recovering the ORIGINAL,
unscaled model's implicit rate, kappa=0.48, as a special case).

This script sweeps KAPPA_LEARN across that full plausible range and checks:
  1. Does the t=200 (long-run reported) outcome change?              -> NO
  2. Does the TRAJECTORY / TIMING change?                              -> YES,
     substantially, for t < ~50 years.
  3. How does kappa_learn compare to the LV subsystem's own relaxation
     rate (computed via the Jacobian of the K-S subsystem)?  Confirms
     whether EGT is "faster than", "comparable to", or "slower than" LV
     across the plausible kappa_learn range.

Outputs:
  sensitivity_kappa_learn_trajectories.png  — x(t) trajectories across kappa
  sensitivity_kappa_learn_timescale.png     — EGT vs LV relaxation comparison
  sensitivity_x0_trajectories.png          — x(t), K(t) across x(0) values at h=0.40
  sensitivity_x0_klearn_heatmap.png        — 2D K(t=200) heatmap: x0 x kappa_learn at h=0.40
"""

import numpy as np
from scipy.integrate import solve_ivp
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ── Core parameters (v6: ORIGINAL payoff relativities, reframed as utility) ──
a, b, c, d   = 0.35, 0.55, 0.45, 0.20         # v7: c=0.45, d=0.20
v_open, v_cover = 1.3, 0.7
r_base, alpha_base, beta, delta = 0.05, 0.044, 0.0175, 0.35
K_max, r_stoat, S_max, H_HOLL   = 150.0, 0.60, 8.0, 0.1
p_open, p_cover  = 0.30, 0.15                  # p_open=0.30 (v7): floor margin = 0.35-0.30=0.05
k_sig, S_mid, floor = 3.0, 1.0, 0.01
x0, K0, S0       = 0.30, 20.0, 3.0

xs = (d - b) / ((a - b) + (d - c))             # standalone EGT x* = 7/9 = 0.7778

# PI_BAR_ESS: average payoff at x* using BASE payoffs — matches v7 main model
def _pi_bar_base(x):
    pA = a*x + b*(1-x); pB = c*x + d*(1-x)
    return x*pA + (1-x)*pB
PI_BAR_ESS = _pi_bar_base(xs)

def f_cubic(x):
    """Unscaled replicator RHS (without kappa_learn multiplier)."""
    pi_A = a*x + b*(1-x); pi_B = c*x + d*(1-x); pi_bar = x*pi_A + (1-x)*pi_B
    return x * (pi_A - pi_bar)

_eps = 1e-6
FPRIME_XSTAR = (f_cubic(xs+_eps) - f_cubic(xs-_eps)) / (2*_eps)   # ≈ -0.10312

def dynamic_pm(S):
    sig = 1 / (1 + np.exp(-k_sig * (S - S_mid)))
    op, cv = p_open*sig, p_cover*sig
    return np.array([
        [max(a-op, floor), max(b-op, floor)],
        [max(c-cv, floor), max(d-cv, floor)]
    ])

def make_rhs(kappa, h, int_yr):
    def rhs(t, y):
        x, K, S = y
        x = np.clip(x, 0.001, 0.999); K = max(K, 0); S = max(S, 0)
        pm   = dynamic_pm(S)
        piA  = pm[0,0]*x + pm[0,1]*(1-x)
        piB  = pm[1,0]*x + pm[1,1]*(1-x)
        piav = x*piA + (1-x)*piB
        dxdt = kappa * x * (piA - piav)
        avg_repr = a*x**2 + b*x*(1-x) + c*(1-x)*x + d*(1-x)**2   # BASE payoffs only
        avg_v    = v_open*x + v_cover*(1-x)
        r_eff    = r_base * (avg_repr / PI_BAR_ESS)
        a_eff    = alpha_base * avg_v
        f_K      = (a_eff*K) / (1 + a_eff*H_HOLL*K)
        dKdt     = r_eff*K*(1 - K/K_max) - f_K*S
        nat      = r_stoat*S*(1-S/S_max) + beta*f_K*S - delta*S
        Sfl      = S_max*(1 - delta/r_stoat)
        nk       = r_stoat*S*(1-S/S_max) - delta*S
        if S <= Sfl and nk < 0:
            nat = max(nat, 0.0)
        hv = h if (int_yr is not None and t >= int_yr) else 0.0
        return [dxdt, dKdt, nat - hv*S]
    return rhs

# ── kappa_learn sweep range ───────────────────────────────────────────────────
# Target half-lives spanning ~2 weeks to ~14 years (the slow end recovers
# the ORIGINAL unscaled model's implicit kappa=0.48 as a boundary case)
TARGET_HALFLIVES = np.array([0.04, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 14.0])
KAPPA_VALUES     = np.log(2) / (TARGET_HALFLIVES * abs(FPRIME_XSTAR))

print("="*70)
print("KAPPA_LEARN SWEEP RANGE")
print("="*70)
print(f"f'(x*) = {FPRIME_XSTAR:.5f}")
print(f"{'target half-life (yr)':>22}  {'kappa_learn':>14}")
for hl, k in zip(TARGET_HALFLIVES, KAPPA_VALUES):
    print(f"{hl:22.2f}  {k:14.3f}")

# ═══════════════════════════════════════════════════════════════════════════════
# PART 1 — Trajectory comparison across kappa_learn (Scenario 3a)
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("PART 1: x(t) trajectories across kappa_learn (Scenario 3a, h=0.40, yr10)")
print("="*70)

T_MAX = 200
t_checkpoints = [0, 5, 10, 15, 20, 30, 50, 100, 200]
trajectories = {}
for kappa, hl in zip(KAPPA_VALUES, TARGET_HALFLIVES):
    sol = solve_ivp(make_rhs(kappa, 0.40, 10), [0, T_MAX], [x0, K0, S0],
                    t_eval=np.linspace(0, T_MAX, 600),
                    method='RK45', rtol=1e-10, atol=1e-12)
    trajectories[hl] = sol

print(f"{'t':>5}  " + "  ".join(f"hl={hl:.2f}y" for hl in TARGET_HALFLIVES))
for tc in t_checkpoints:
    row = []
    for hl in TARGET_HALFLIVES:
        sol = trajectories[hl]
        idx = np.argmin(np.abs(sol.t - tc))
        row.append(sol.y[0, idx] * 100)
    print(f"{tc:5d}  " + "  ".join(f"{v:9.2f}" for v in row))

# ═══════════════════════════════════════════════════════════════════════════════
# PART 2 — LV subsystem relaxation rate (Jacobian), for timescale comparison
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("PART 2: LV subsystem relaxation rate (Jacobian near Sc3a end-state)")
print("="*70)

x_fixed = 0.7620  # converged CEPPM value near full suppression (Sc3a, t=200, v7: x*=77.78%)

def KS_rhs(K, S):
    avg_repr = a*x_fixed**2 + b*x_fixed*(1-x_fixed) + \
               c*(1-x_fixed)*x_fixed + d*(1-x_fixed)**2   # BASE payoffs
    avg_v = v_open*x_fixed + v_cover*(1-x_fixed)
    r_eff = r_base*(avg_repr/PI_BAR_ESS); a_eff = alpha_base*avg_v
    f_K = (a_eff*K)/(1+a_eff*H_HOLL*K)
    dK = r_eff*K*(1-K/K_max) - f_K*S
    nat = r_stoat*S*(1-S/S_max) + beta*f_K*S - delta*S
    return dK, nat

K_eq, S_eq = 149.7, 0.01
eps_jac = 0.5
dK_dK = (KS_rhs(K_eq+eps_jac,S_eq)[0]-KS_rhs(K_eq-eps_jac,S_eq)[0])/(2*eps_jac)
dK_dS = (KS_rhs(K_eq,S_eq+eps_jac)[0]-KS_rhs(K_eq,S_eq-eps_jac)[0])/(2*eps_jac)
dS_dK = (KS_rhs(K_eq+eps_jac,S_eq)[1]-KS_rhs(K_eq-eps_jac,S_eq)[1])/(2*eps_jac)
dS_dS = (KS_rhs(K_eq,S_eq+eps_jac)[1]-KS_rhs(K_eq,S_eq-eps_jac)[1])/(2*eps_jac)
J = np.array([[dK_dK, dK_dS],[dS_dK, dS_dS]])
eigvals = np.linalg.eigvals(J)
LV_SLOWEST_RATE = np.min(np.abs(eigvals.real))
LV_HALFLIFE = np.log(2) / LV_SLOWEST_RATE

print(f"Jacobian eigenvalues: {eigvals}")
print(f"LV subsystem slowest relaxation rate = {LV_SLOWEST_RATE:.4f} /yr")
print(f"LV subsystem half-life = {LV_HALFLIFE:.2f} years")
print()
print("EGT:LV half-life ratio across the kappa_learn sweep:")
for hl in TARGET_HALFLIVES:
    print(f"  EGT half-life={hl:6.2f}yr  ->  ratio to LV ({LV_HALFLIFE:.2f}yr) = {hl/LV_HALFLIFE:.3f}:1")

# ═══════════════════════════════════════════════════════════════════════════════
# FIGURES
# ═══════════════════════════════════════════════════════════════════════════════
BG = '#F7F6F2'
C_grid = '#DDDDDD'
cmap = plt.cm.viridis
colors = [cmap(i) for i in np.linspace(0, 1, len(TARGET_HALFLIVES))]

plt.rcParams.update({
    'font.family': 'DejaVu Sans', 'font.size': 10,
    'axes.spines.top': False, 'axes.spines.right': False,
    'axes.facecolor': BG, 'figure.facecolor': BG,
    'axes.grid': True, 'grid.color': C_grid, 'grid.linewidth': 0.6,
})

# Figure 1: trajectories
fig1, ax1 = plt.subplots(figsize=(9, 6.5), facecolor=BG)
for hl, color in zip(TARGET_HALFLIVES, colors):
    sol = trajectories[hl]
    label = f'half-life={hl:.2f}yr (κ={KAPPA_VALUES[list(TARGET_HALFLIVES).index(hl)]:.2f})'
    if abs(hl - 14.0) < 1e-6:
        label += '  [= original model]'
    ax1.plot(sol.t, sol.y[0]*100, color=color, lw=2.0, label=label)
ax1.axhline(xs*100, color='gray', lw=1.0, ls='--', alpha=0.6)
ax1.text(95, xs*100+0.3, f'standalone EGT x*={xs*100:.2f}% (7/9)',
         ha='right', fontsize=8, color='gray')
ax1.set_xlabel('Time (years)')
ax1.set_ylabel('Open foraging strategy x (%)')
ax1.set_title('Scenario 3a (h=0.40, intervention yr 10)\nx(t) across plausible kappa_learn / half-life values', pad=14)
ax1.legend(fontsize=7.5, loc='lower right')
ax1.set_xlim(0, 100)
plt.tight_layout()
plt.savefig('/mnt/user-data/outputs/sensitivity_kappa_learn_trajectories.png',
            dpi=150, bbox_inches='tight', facecolor=BG)
print("\nSaved: sensitivity_kappa_learn_trajectories.png")

# Figure 2: timescale comparison bar chart
fig2, ax2 = plt.subplots(figsize=(9, 5.5), facecolor=BG)
y_pos = np.arange(len(TARGET_HALFLIVES))
bars = ax2.barh(y_pos, TARGET_HALFLIVES, color=colors, edgecolor='white', height=0.6)
ax2.axvline(LV_HALFLIFE, color='#CC3030', lw=2.0, ls='--',
            label=f'LV subsystem half-life ({LV_HALFLIFE:.2f}yr)')
ax2.set_yticks(y_pos)
ax2.set_yticklabels([f'κ={k:.2f}\n(hl={hl:.2f}yr)' for k, hl in zip(KAPPA_VALUES, TARGET_HALFLIVES)],
                     fontsize=8)
ax2.set_xscale('log')
ax2.set_xlabel('Half-life (years, log scale)')
ax2.set_title('EGT relaxation half-life vs. LV subsystem half-life\n'
              '(bars right of red line = EGT slower than LV; left = EGT faster)')
ax2.legend(fontsize=9, loc='lower right')
plt.tight_layout()
plt.savefig('/mnt/user-data/outputs/sensitivity_kappa_learn_timescale.png',
            dpi=150, bbox_inches='tight', facecolor=BG)
print("Saved: sensitivity_kappa_learn_timescale.png")

# ═══════════════════════════════════════════════════════════════════════════════
# PART 3 — x(0) sensitivity at baseline kappa_learn=1.5, h=0.40
# Addresses Chippy's concern: what if x(0) is close to 0 or 1?
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("PART 3: x(0) sensitivity at kappa_learn=1.5, h=0.40, intervention yr 10")
print("="*70)

KAPPA_BASE = 1.5
H_SENS     = 0.40
INT_YR     = 10

# x(0) sweep — from near-cover-only to near-open-only
X0_VALUES  = np.array([0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80])
x0_trajs   = {}

# Also track: time for x to first cross x_survival (approx 40.56% in v7)
X_SURVIVAL = 0.4056

print(f"\n{'x0':>6}  {'x@50':>7}  {'x@100':>7}  {'x@200':>7}  {'K@200':>7}  "
      f"{'S@200':>7}  {'t_cross_xsurv':>15}")
for x0_val in X0_VALUES:
    sol = solve_ivp(
        make_rhs(KAPPA_BASE, H_SENS, INT_YR),
        [0, T_MAX], [x0_val, K0, S0],
        t_eval=np.linspace(0, T_MAX, 2000),
        method='RK45', rtol=1e-10, atol=1e-12
    )
    x0_trajs[x0_val] = sol

    def get_val(sol, t_target, row):
        idx = np.argmin(np.abs(sol.t - t_target))
        return sol.y[row, idx]

    x50  = get_val(sol, 50,  0) * 100
    x100 = get_val(sol, 100, 0) * 100
    x200 = get_val(sol, 200, 0) * 100
    K200 = get_val(sol, 200, 1)
    S200 = get_val(sol, 200, 2)

    # Time to first cross x_survival (only meaningful if x0 < x_survival)
    cross_idx = np.where(sol.y[0] >= X_SURVIVAL)[0]
    if len(cross_idx) > 0:
        t_cross = sol.t[cross_idx[0]]
        t_cross_str = f"{t_cross:>12.1f}yr"
    else:
        t_cross_str = "     never"

    print(f"{x0_val*100:5.0f}%  {x50:7.2f}%  {x100:7.2f}%  {x200:7.2f}%  "
          f"{K200:7.1f}  {S200:7.4f}  {t_cross_str}")

# Figure 3: x(t) and K(t) across x(0) values
cmap3  = plt.cm.plasma
cols3  = [cmap3(i) for i in np.linspace(0.1, 0.9, len(X0_VALUES))]

fig3, (ax3a, ax3b) = plt.subplots(1, 2, figsize=(13, 5.5), facecolor=BG)

for x0_val, col in zip(X0_VALUES, cols3):
    sol = x0_trajs[x0_val]
    lbl = f'x(0)={x0_val*100:.0f}%'
    ax3a.plot(sol.t, sol.y[0]*100, color=col, lw=1.8, label=lbl)
    ax3b.plot(sol.t, sol.y[1],     color=col, lw=1.8, label=lbl)

ax3a.axhline(xs*100,      color='gray',    lw=1.0, ls='--', alpha=0.6,
             label=f'EGT x*={xs*100:.1f}%')
ax3a.axhline(X_SURVIVAL*100, color='#CC3030', lw=1.0, ls=':',  alpha=0.8,
             label=f'x_survival≈{X_SURVIVAL*100:.1f}%')
ax3a.axvline(INT_YR, color='black', lw=0.8, ls=':', alpha=0.5)
ax3a.set_xlabel('Time (years)')
ax3a.set_ylabel('Open foraging strategy x (%)')
ax3a.set_title(f'x(t) across x(0) values\n(κ_learn={KAPPA_BASE}, h={H_SENS}, '
               f'intervention yr {INT_YR})', pad=10)
ax3a.legend(fontsize=7.5, loc='lower right')
ax3a.set_xlim(0, T_MAX)

ax3b.axvline(INT_YR, color='black', lw=0.8, ls=':', alpha=0.5,
             label=f'Intervention yr {INT_YR}')
ax3b.axhline(K0, color='gray', lw=1.0, ls='--', alpha=0.5, label=f'K₀={K0}')
ax3b.set_xlabel('Time (years)')
ax3b.set_ylabel('Kiwi population K (birds/1,000 ha)')
ax3b.set_title(f'K(t) across x(0) values\n(κ_learn={KAPPA_BASE}, h={H_SENS}, '
               f'intervention yr {INT_YR})', pad=10)
ax3b.legend(fontsize=7.5, loc='upper left')
ax3b.set_xlim(0, T_MAX)

plt.tight_layout()
plt.savefig('/mnt/user-data/outputs/sensitivity_x0_trajectories.png',
            dpi=150, bbox_inches='tight', facecolor=BG)
print("\nSaved: sensitivity_x0_trajectories.png")

# ═══════════════════════════════════════════════════════════════════════════════
# PART 4 — Joint x(0) x kappa_learn 2D heatmap, h=0.40
# The key robustness check: does K(t=200) depend on either?
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("PART 4: 2D heatmap K(t=200) across x(0) x kappa_learn at h=0.40")
print("="*70)

X0_GRID     = np.linspace(0.05, 0.80, 20)
KAPPA_GRID  = np.logspace(np.log10(0.10), np.log10(18.0), 22)  # spans full plausible range

K200_grid   = np.zeros((len(X0_GRID), len(KAPPA_GRID)))
x200_grid   = np.zeros_like(K200_grid)

print(f"Running {len(X0_GRID) * len(KAPPA_GRID)} simulations...")
for i, x0_val in enumerate(X0_GRID):
    for j, kappa in enumerate(KAPPA_GRID):
        sol = solve_ivp(
            make_rhs(kappa, H_SENS, INT_YR),
            [0, T_MAX], [x0_val, K0, S0],
            t_eval=[T_MAX],
            method='RK45', rtol=1e-8, atol=1e-10
        )
        K200_grid[i, j] = sol.y[1, -1]
        x200_grid[i, j] = sol.y[0, -1] * 100
    if (i+1) % 5 == 0:
        print(f"  x0 row {i+1}/{len(X0_GRID)} done")

print("\nK(t=200) summary across full grid:")
print(f"  min={K200_grid.min():.1f}  max={K200_grid.max():.1f}  "
      f"mean={K200_grid.mean():.1f}  std={K200_grid.std():.2f}")
print(f"  Range of variation: {K200_grid.max()-K200_grid.min():.1f} birds")

# Figure 4: 2D heatmaps side by side — K(200) and x(200)
fig4, (ax4a, ax4b) = plt.subplots(1, 2, figsize=(14, 5.5), facecolor=BG)

# K(200) heatmap
im4a = ax4a.contourf(
    np.log10(KAPPA_GRID), X0_GRID*100, K200_grid,
    levels=20, cmap='YlGn'
)
ax4a.contour(
    np.log10(KAPPA_GRID), X0_GRID*100, K200_grid,
    levels=[20, 50, 100, 130, 145], colors='black',
    linewidths=0.8, alpha=0.6
)
# Mark baseline x0=0.30, kappa=1.5
ax4a.plot(np.log10(KAPPA_BASE), 30, '*', color='red', ms=12, zorder=5,
          label=f'baseline (x₀=30%, κ={KAPPA_BASE})')
ax4a.axhline(30, color='red', lw=0.7, ls='--', alpha=0.4)
ax4a.axvline(np.log10(KAPPA_BASE), color='red', lw=0.7, ls='--', alpha=0.4)
plt.colorbar(im4a, ax=ax4a, label='K at t=200 (birds/1,000 ha)')
ax4a.set_xlabel('κ_learn (log scale)')
ax4a.set_ylabel('Initial open foraging x(0) (%)')
ax4a.set_title(f'K(t=200): x(0) × κ_learn\n(h={H_SENS}, intervention yr {INT_YR})', pad=10)
ax4a.set_xticks(np.log10([0.1, 0.3, 1.0, 3.0, 10.0]))
ax4a.set_xticklabels(['0.1', '0.3', '1.0', '3.0', '10.0'])
ax4a.legend(fontsize=8, loc='upper left')

# x(200) heatmap
im4b = ax4b.contourf(
    np.log10(KAPPA_GRID), X0_GRID*100, x200_grid,
    levels=20, cmap='PuBu'
)
ax4b.contour(
    np.log10(KAPPA_GRID), X0_GRID*100, x200_grid,
    levels=[40, 55, 65, 72, 76], colors='black',
    linewidths=0.8, alpha=0.6
)
ax4b.plot(np.log10(KAPPA_BASE), 30, '*', color='red', ms=12, zorder=5,
          label=f'baseline')
ax4b.axhline(30, color='red', lw=0.7, ls='--', alpha=0.4)
ax4b.axvline(np.log10(KAPPA_BASE), color='red', lw=0.7, ls='--', alpha=0.4)
plt.colorbar(im4b, ax=ax4b, label='x at t=200 (%)')
ax4b.set_xlabel('κ_learn (log scale)')
ax4b.set_ylabel('Initial open foraging x(0) (%)')
ax4b.set_title(f'x(t=200): x(0) × κ_learn\n(h={H_SENS}, intervention yr {INT_YR})', pad=10)
ax4b.set_xticks(np.log10([0.1, 0.3, 1.0, 3.0, 10.0]))
ax4b.set_xticklabels(['0.1', '0.3', '1.0', '3.0', '10.0'])
ax4b.legend(fontsize=8, loc='upper left')

plt.tight_layout()
plt.savefig('/mnt/user-data/outputs/sensitivity_x0_klearn_heatmap.png',
            dpi=150, bbox_inches='tight', facecolor=BG)
print("Saved: sensitivity_x0_klearn_heatmap.png")
