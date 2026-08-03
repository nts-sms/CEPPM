# -*- coding: utf-8 -*-
"""
CEPPM — h_erad Sensitivity to All Parameters (reconstructed + CORRECTED)
==========================================================================
Reconstruction of the "h_erad sensitivity — companion to
threshold_sensitivity_h_survival.png" figure. The original plotting code
wasn't available, so this rebuilds it from the compute_thresholds() logic
in sensitivity_thresholds_h_crit_h_erad.py, applying the same fix already
made in the other corrected files:

  h_erad = h_crit + beta*f(K_max),  f(K) = alpha_eff*K/(1+alpha_eff*H*K)

  alpha_eff = alpha_base * avg_vulnerability(x*_CEPPM)

  where x*_CEPPM is the EGT equilibrium under the DYNAMIC (sigmoid-
  penalized) payoffs evaluated at S=0 -- NOT the base-payoff EGT x*.
  This was hardcoded to alpha_base directly in the original (bug),
  making h_erad look independent of a,b,c,d,v_open,v_cover. It isn't:
  alpha_eff depends on x*_CEPPM (which depends on a,b,c,d) and on
  avg_vulnerability (which depends on v_open,v_cover). Those six panels
  are recomputed here rather than assumed flat/invariant.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUTPUT_DIR = '/home/claude/work/sensitivity_outputs'

# ── Baseline parameters (identical to sensitivity_thresholds_h_crit_h_erad.py)
r_base    = 0.05
alpha_b   = 0.044
beta_b    = 0.0175
delta_b   = 0.35
K_max     = 150
Hh        = 0.1
r_stoat_b = 0.6
S_max     = 8
a_b, b_b, c_b, d_b = 0.35, 0.55, 0.45, 0.20
v_open_b, v_cover_b = 1.3, 0.7

# Sigmoid/penalty params (fixed throughout -- not part of this sweep)
K_SIG, S_MID, OPEN_MAX, COV_MAX = 3.0, 1.0, 0.30, 0.15

x_fine = np.linspace(0.0, 1.0, 200001)


def f_holling(K, alpha_eff, H=Hh):
    return alpha_eff * K / (1 + alpha_eff * H * K)


def x_star_ceppm(a, b, c, d):
    """EGT equilibrium under DYNAMIC payoffs evaluated at S=0 (not the
    base-payoff EGT x*). Sigmoid penalty isn't exactly zero at S=0."""
    sig0   = 1 / (1 + np.exp(-K_SIG * (0 - S_MID)))
    a_dyn0 = a - OPEN_MAX * sig0
    b_dyn0 = b - OPEN_MAX * sig0
    c_dyn0 = c - COV_MAX  * sig0
    d_dyn0 = d - COV_MAX  * sig0
    denom  = (a_dyn0 - b_dyn0) + (d_dyn0 - c_dyn0)
    if abs(denom) < 1e-9:
        return None
    return (d_dyn0 - b_dyn0) / denom


def compute_thresholds(r=r_base, alpha=alpha_b, beta=beta_b, delta=delta_b,
                        r_stoat=r_stoat_b, a=a_b, b=b_b, c=c_b, d=d_b,
                        v_open=v_open_b, v_cover=v_cover_b):
    """Returns dict with h_crit, h_erad, h_survival and validity flags.
    h_erad now uses alpha_eff (CORRECTED), not alpha_base directly."""
    valid_lv = r_stoat > delta
    h_crit = r_stoat - delta

    # --- CORRECTED h_erad: alpha_eff at x*_CEPPM, not alpha_base ---
    xs_ceppm = x_star_ceppm(a, b, c, d)
    if xs_ceppm is None or not (0.0 < xs_ceppm < 1.0):
        h_erad = np.nan
        valid_lv_erad = False
    else:
        avg_vuln  = v_open * xs_ceppm + v_cover * (1 - xs_ceppm)
        alpha_eff = alpha * avg_vuln
        h_erad = h_crit + beta * f_holling(K_max, alpha_eff)
        valid_lv_erad = valid_lv

    denom = (a - b) + (d - c)
    valid_hs = True
    h_survival = np.nan
    if abs(denom) < 1e-9:
        valid_hs = False
    else:
        xs = (d - b) / denom
        if not (0.0 < xs < 1.0):
            valid_hs = False
        else:
            PI_BAR_ESS = (a - b - c + d) * xs**2 + (b + c - 2*d) * xs + d
            if PI_BAR_ESS <= 0:
                valid_hs = False
            else:
                pi_bar = (a - b - c + d) * x_fine**2 + (b + c - 2*d) * x_fine + d
                vuln   = v_cover + (v_open - v_cover) * x_fine
                if np.any(vuln <= 0):
                    valid_hs = False
                else:
                    R_max = np.max(pi_bar / vuln)
                    C_const = r / (alpha * PI_BAR_ESS)
                    S_crit  = C_const * R_max
                    h_survival = r_stoat * (1 - S_crit / S_max) - delta
                    valid_hs = (0 <= S_crit <= S_max) and (h_survival >= 0)

    return dict(h_crit=h_crit, h_erad=h_erad, h_survival=h_survival,
                valid_lv=valid_lv_erad, valid_hs=valid_hs)


baseline = compute_thresholds()
print(f"BASELINE (corrected): h_crit={baseline['h_crit']:.4f}  "
      f"h_erad={baseline['h_erad']:.4f}  h_survival={baseline['h_survival']:.4f}")

RANGES = {
    'r_stoat': (np.linspace(0.30, 0.90, 25),  'Wide stress-test', '#d62728', r_stoat_b),
    'delta':   (np.linspace(0.15, 0.55, 25),  'Wide stress-test', '#d62728', delta_b),
    'alpha':   (np.linspace(0.010, 0.100, 25), 'Wide stress-test', '#ff7f0e', alpha_b),
    'beta':    (np.linspace(0.005, 0.050, 25), 'Wide stress-test', '#ff7f0e', beta_b),
    'r':       (np.linspace(0.02, 0.10, 25),  'Wide stress-test (no lit. anchor for r)', '#ff7f0e', r_base),
    'a':       (np.linspace(0.245, 0.455, 25), '+/-30% baseline', '#1f77b4', a_b),
    'b':       (np.linspace(0.385, 0.715, 25), '+/-30% baseline', '#1f77b4', b_b),
    'c':       (np.linspace(0.315, 0.585, 25), '+/-30% baseline', '#1f77b4', c_b),
    'd':       (np.linspace(0.140, 0.260, 25), '+/-30% baseline', '#1f77b4', d_b),
    'v_open':  (np.linspace(1.0, 1.8, 25),    'Existing convention', '#2ca02c', v_open_b),
    'v_cover': (np.linspace(0.4, 0.9, 25),    'Existing convention', '#2ca02c', v_cover_b),
}

param_labels = {
    'r_stoat': r'$r_{stoat}$', 'delta': r'$\delta$', 'alpha': r'$\alpha$', 'beta': r'$\beta$',
    'r': 'r (kiwi growth)', 'a': 'a', 'b': 'b', 'c': 'c', 'd': 'd',
    'v_open': r'$v_{open}$', 'v_cover': r'$v_{cover}$',
}

results = {}
for key, (vals, source, col, pbase) in RANGES.items():
    rows = [compute_thresholds(**{key: v}) for v in vals]
    he  = np.array([row['h_erad'] for row in rows])
    vlv = np.array([row['valid_lv'] for row in rows])
    results[key] = dict(vals=vals, he=he, vlv=vlv, source=source, col=col, pbase=pbase)

# ── Print span table ────────────────────────────────────────────────────────
print(f"\n{'Param':<10} {'Range':<14} {'h_erad range':<20} {'span':>8}  status")
print("-"*70)
for key, res in results.items():
    vals = res['vals']
    he_v = res['he'][res['vlv']]
    vr = f"{vals[0]:.3f}-{vals[-1]:.3f}"
    if len(he_v) and np.nanmax(he_v)-np.nanmin(he_v) > 1e-6:
        span = np.nanmax(he_v) - np.nanmin(he_v)
        status = f"span={span:.4f}"
    else:
        span = 0.0
        status = "invariant"
    print(f"{key:<10} {vr:<14} {np.nanmin(he_v):.4f}-{np.nanmax(he_v):.4f}      {span:8.4f}  {status}")

# ── Plot: 3 rows x 4 cols, matching original layout ─────────────────────────
param_order = ['r_stoat', 'delta', 'alpha', 'beta',
               'r', 'a', 'b', 'c',
               'd', 'v_open', 'v_cover']

fig, axes = plt.subplots(3, 4, figsize=(18, 12))
axes_flat = axes.flatten()

for i, key in enumerate(param_order):
    ax = axes_flat[i]
    res = results[key]
    vals, he, vlv, col, pbase = res['vals'], res['he'], res['vlv'], res['col'], res['pbase']

    he_valid = he[vlv]
    span = (np.nanmax(he_valid) - np.nanmin(he_valid)) if len(he_valid) else np.nan
    invariant = span < 1e-4

    ax.plot(vals[vlv], he[vlv], color=col, lw=2)
    if (~vlv).any():
        ax.axvspan(vals[0], vals[~vlv].max(), alpha=0.15, color='gray')
        ax.text(0.02, 0.03, 'Wide stress-test', transform=ax.transAxes,
                fontsize=7, style='italic', color='#555')
    else:
        ax.text(0.02, 0.03, res['source'], transform=ax.transAxes,
                fontsize=7, style='italic', color='#555')

    base_out = compute_thresholds(**{key: pbase})['h_erad']
    ax.plot(pbase, base_out, marker='*', color='black', markersize=14, zorder=5)

    title_suffix = '(invariant)' if invariant else f'(span={span:.4f})'
    ax.set_title(f"{param_labels[key]}  {title_suffix}", fontsize=11)
    ax.set_xlabel(param_labels[key], fontsize=10)
    if i % 4 == 0:
        ax.set_ylabel('h_erad', fontsize=10)
    ax.set_ylim(0, max(0.65, np.nanmax(he_valid)*1.1 if len(he_valid) else 0.65))
    ax.grid(True, alpha=0.3)

# Legend box in the 12th slot
ax_leg = axes_flat[11]
ax_leg.axis('off')
legend_text = (
    "Star = baseline\n"
    "(r_stoat=0.6, delta=0.35, alpha=0.044,\n"
    "beta=0.0175, r=0.05,\n"
    "a=0.35,b=0.55,c=0.45,d=0.20,\n"
    "v_open=1.3, v_cover=0.7)\n\n"
    f"Baseline h_erad = {baseline['h_erad']:.4f}\n\n"
    "Grey shading = excluded\n"
    "(r_stoat <= delta)"
)
ax_leg.text(0.05, 0.95, legend_text, transform=ax_leg.transAxes,
            fontsize=10, va='top', ha='left',
            bbox=dict(boxstyle='round,pad=0.6', facecolor='white', edgecolor='gray'))

fig.suptitle(
    'h_erad sensitivity - companion to threshold_sensitivity_h_survival.png\n'
    'Red = stoat demography, orange = predation-bonus/growth, blue = payoff matrix, green = vulnerability weights',
    fontsize=12
)
plt.tight_layout(rect=[0, 0, 1, 0.94])
outpath = f'{OUTPUT_DIR}/h_erad_sensitivity_all_parameters_CORRECTED.png'
plt.savefig(outpath, dpi=150, bbox_inches='tight')
print(f"\nSaved: {outpath}")
