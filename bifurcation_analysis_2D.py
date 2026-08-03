{\rtf1\ansi\ansicpg1252\cocoartf2868
\cocoatextscaling0\cocoaplatform0{\fonttbl\f0\fswiss\fcharset0 Helvetica;}
{\colortbl;\red255\green255\blue255;}
{\*\expandedcolortbl;;}
\paperw11900\paperh16840\margl1440\margr1440\vieww11520\viewh8400\viewkind0
\pard\tx566\tx1133\tx1700\tx2267\tx2834\tx3401\tx3968\tx4535\tx5102\tx5669\tx6236\tx6803\pardirnatural\partightenfactor0

\f0\fs24 \cf0 # -*- coding: utf-8 -*-\
"""\
CEPPM \'97 Bifurcation Analysis vs Stoat Harvest Rate h\
======================================================\
Reviewer #5 response: analytical bifurcation analysis of the extended\
Lotka-Volterra kiwi-stoat system, using harvest rate h as the bifurcation\
parameter. Companion to kiwi_conservation_analysis_v7.py \'97 parameters below\
are copied from that script's run_complete_kiwi_analysis() baseline.\
\
NOTATION: Holling Type II handling time is written H throughout this script\
(matches paper convention). This is DELIBERATELY different from the main\
model script, which (confusingly) uses lowercase h for both the handling\
time AND the harvest rate. Here, h = harvest rate (bifurcation parameter),\
H = handling time (fixed constant = 0.1).\
\
SYSTEM\
------\
    dK/dt = r*K*(1 - K/K_max) - f(K)*S                                  (kiwi)\
    dS/dt = r_stoat*S*(1 - S/S_max) + beta*f(K)*S - delta*S - h*S       (stoat)\
    f(K)  = alpha*K / (1 + alpha*H*K)                        (Holling Type II)\
\
EQUILIBRIUM BRANCHES (derived analytically \'97 see session handoff v21 for\
full derivation)\
-----------------------------------------------------------------------\
  A. Total extinction:      (K,S) = (0, 0)                        \'97 always exists\
  B. Stoat-floor:            (K,S) = (0, S_floor(h))\
                             S_floor(h) = S_max*(1 - (delta+h)/r_stoat)\
                             physical only while S_floor(h) >= 0, i.e. h <= h_crit\
  C. Full suppression:       (K,S) = (K_max, 0)                   \'97 always exists\
  D. Coexistence:            solved numerically per h from the nullcline\
                             intersection (kiwi nullcline = stoat nullcline);\
                             no closed form because f(K) appears both inside\
                             a denominator and inside the stoat nullcline.\
\
JACOBIAN\
--------\
    J = [[ r*(1-2K/K_max) - f'(K)*S,   -f(K)                        ],\
         [ beta*f'(K)*S,                r_stoat*(1-2S/S_max) + beta*f(K) - delta - h ]]\
\
    f'(K) = alpha / (1 + alpha*H*K)^2\
\
THREE BIFURCATION THRESHOLDS (all confirmed transcritical; no Hopf found\
anywhere on Branch D \'97 see session handoff v21 for full eigenvalue sweep)\
--------------------------------------------------------------------------\
  h_fold = r_stoat*(1 - r/(alpha*S_max)) - delta\
           Branch B loses K-direction stability; Branch D (coexistence)\
           branches off. This is the "effective recovery threshold" (~0.165)\
           already shown in Figure 3 \'97 this script gives it a closed form.\
\
  h_crit = r_stoat - delta\
           Branch B's S_floor(h) hits exactly zero, merging with Branch A.\
           Matches the paper's existing h_crit = 0.25.\
\
  h_erad = h_crit + beta*f(K_max)\
           Branch D's S* declines to zero, meeting Branch C. Matches the\
           paper's existing h_erad ~= 0.32.\
\
  h_erad_theo = r_stoat\
           Theoretical upper bound (harvest exceeds total stoat intrinsic\
           growth capacity) \'97 PF2050 worst-case scenario, unchanged from\
           main model script.\
\
USAGE\
-----\
    python3 bifurcation_analysis_h.py\
\
Produces: bifurcation_diagram_2panel.png (K* and S* vs h, stacked panels,\
solid=stable / dashed=unstable, zone shading matches Figure 3 convention).\
No external dependencies beyond numpy + matplotlib (no scipy required \'97\
uses a manual bisection root-finder since scipy wasn't available in the\
sandbox this was developed in; swap in scipy.optimize.brentq if preferred).\
"""\
\
import numpy as np\
import matplotlib.pyplot as plt\
\
# ============================================================================\
# ROOT FINDER (bisection \'97 no scipy dependency)\
# ============================================================================\
\
def bisect(func, a, b, args=(), tol=1e-12, maxiter=200):\
    """Simple bisection root finder. Assumes func(a) and func(b) bracket a root."""\
    fa = func(a, *args)\
    for _ in range(maxiter):\
        m = 0.5 * (a + b)\
        fm = func(m, *args)\
        if abs(fm) < tol or (b - a) < tol:\
            return m\
        if np.sign(fm) == np.sign(fa):\
            a, fa = m, fm\
        else:\
            b = m\
    return 0.5 * (a + b)\
\
\
# ============================================================================\
# BASELINE PARAMETERS (from kiwi_conservation_analysis_v7.py,\
# run_complete_kiwi_analysis(): r, alpha, beta, delta = 0.05, 0.044, 0.0175, 0.35)\
# ============================================================================\
\
r        = 0.05    # kiwi intrinsic growth rate (DOC Recovery Plan 2018-2028)\
alpha    = 0.044   # predation attack rate\
beta     = 0.0175  # stoat conversion efficiency (kiwi predation bonus)\
delta    = 0.35    # stoat natural death rate\
K_max    = 150     # kiwi carrying capacity\
H        = 0.1     # Holling Type II handling time (renamed from lowercase h)\
r_stoat  = 0.6     # stoat intrinsic growth rate from non-kiwi prey\
S_max    = 8        # stoat carrying capacity from non-kiwi prey\
\
H_MAX_PLOT = 0.62   # upper bound of h shown on plots\
\
\
def f(K):\
    """Holling Type II functional response."""\
    return alpha * K / (1 + alpha * H * K)\
\
\
def fprime(K):\
    """d/dK of Holling Type II response."""\
    return alpha / (1 + alpha * H * K) ** 2\
\
\
def kiwi_nullcline_S(K):\
    """S as a function of K along dK/dt = 0."""\
    return r * K * (1 - K / K_max) / f(K)\
\
\
def stoat_nullcline_S(K, h):\
    """S as a function of K along dS/dt = 0, for a given harvest rate h."""\
    return S_max * (1 - (delta + h - beta * f(K)) / r_stoat)\
\
\
def gap(K, h):\
    """Difference between the two nullclines; root = coexistence equilibrium K*."""\
    return kiwi_nullcline_S(K) - stoat_nullcline_S(K, h)\
\
\
def jacobian(K, S, h):\
    """2x2 Jacobian of the (K,S) system evaluated at (K, S) for harvest rate h."""\
    fK, fpK = f(K), fprime(K)\
    J11 = r * (1 - 2 * K / K_max) - fpK * S\
    J12 = -fK\
    J21 = beta * fpK * S\
    J22 = r_stoat * (1 - 2 * S / S_max) + beta * fK - delta - h\
    return np.array([[J11, J12], [J21, J22]])\
\
\
# ============================================================================\
# CLOSED-FORM THRESHOLDS\
# ============================================================================\
\
h_crit      = r_stoat - delta\
h_erad      = h_crit + beta * f(K_max)\
h_fold      = r_stoat * (1 - r / (alpha * S_max)) - delta\
h_erad_theo = r_stoat\
\
\
def print_thresholds():\
    print(f"h_fold      = \{h_fold:.4f\}   (Branch B -> D transcritical; 'effective recovery threshold')")\
    print(f"h_crit      = \{h_crit:.4f\}   (Branch A/B collapse; non-kiwi stoat eq. hits zero)")\
    print(f"h_erad      = \{h_erad:.4f\}   (Branch D -> C transcritical; true eradication threshold)")\
    print(f"h_erad_theo = \{h_erad_theo:.4f\}   (theoretical upper bound = r_stoat)")\
\
\
# ============================================================================\
# BRANCH COMPUTATION\
# ============================================================================\
\
def compute_branches(h_axis):\
    """Returns dict of branch data: K, S, stable (bool array) for A, B, C, D."""\
    branches = \{\}\
\
    # --- Branch A: (0, 0), always exists ---\
    KA = np.zeros_like(h_axis)\
    SA = np.zeros_like(h_axis)\
    lamA1 = np.full_like(h_axis, r)          # always > 0 -> always unstable\
    lamA2 = h_crit - h_axis\
    stableA = (lamA1 < 0) & (lamA2 < 0)\
    branches['A'] = dict(K=KA, S=SA, stable=stableA, mask=None)\
\
    # --- Branch B: (0, S_floor(h)), physical while S_floor(h) >= 0 ---\
    S_floor = S_max * (1 - (delta + h_axis) / r_stoat)\
    maskB = S_floor >= -1e-9\
    KB = np.zeros_like(h_axis)\
    SB = np.clip(S_floor, 0, None)\
    lamB1 = r - alpha * np.clip(S_floor, 0, None)\
    lamB2 = h_axis - h_crit\
    stableB = (lamB1 < 0) & (lamB2 < 0)\
    branches['B'] = dict(K=KB, S=SB, stable=stableB, mask=maskB)\
\
    # --- Branch C: (K_max, 0), always exists ---\
    KC = np.full_like(h_axis, K_max)\
    SC = np.zeros_like(h_axis)\
    lamC1 = np.full_like(h_axis, -r)         # always < 0\
    lamC2 = h_erad - h_axis\
    stableC = (lamC1 < 0) & (lamC2 < 0)\
    branches['C'] = dict(K=KC, S=SC, stable=stableC, mask=None)\
\
    # --- Branch D: coexistence, solved numerically per h ---\
    Kgrid = np.linspace(0.01, K_max - 0.01, 6000)\
    KD = np.full_like(h_axis, np.nan)\
    SD = np.full_like(h_axis, np.nan)\
    stableD = np.zeros_like(h_axis, dtype=bool)\
    for idx, h in enumerate(h_axis):\
        if h <= h_fold or h >= h_erad:\
            continue\
        gvals = gap(Kgrid, h)\
        root = None\
        for i in range(len(Kgrid) - 1):\
            if np.sign(gvals[i]) != np.sign(gvals[i + 1]):\
                root = bisect(gap, Kgrid[i], Kgrid[i + 1], args=(h,))\
                break\
        if root is None:\
            continue\
        S = kiwi_nullcline_S(root)\
        eig = np.linalg.eigvals(jacobian(root, S, h))\
        KD[idx] = root\
        SD[idx] = S\
        stableD[idx] = np.all(eig.real < 0)\
    branches['D'] = dict(K=KD, S=SD, stable=stableD, mask=None)\
\
    return branches\
\
\
# ============================================================================\
# PLOTTING\
# ============================================================================\
\
ZONES = [\
    (0,           None,  '#f3d6d6', 'Failed\\nmanagement'),\
    (None,        None,  '#fbe6c8', 'Partial\\nrecovery'),\
    (None,        None,  '#f7f0b0', 'Kiwi bonus\\nzone'),\
    (None,        None,  '#c9e8d4', 'Full\\nsuppression'),\
    (None,        H_MAX_PLOT, '#c3ddef', 'PF2050 /\\neradication'),\
]\
\
\
def _zone_bounds():\
    return [\
        (0, h_fold),\
        (h_fold, h_crit),\
        (h_crit, h_erad),\
        (h_erad, h_erad_theo),\
        (h_erad_theo, H_MAX_PLOT),\
    ]\
\
\
def plot_branch(ax, h_axis, Y, stable, color, label, mask=None):\
    """Plot one branch, split into solid (stable) / dashed (unstable) segments."""\
    valid = ~np.isnan(Y)\
    if mask is not None:\
        valid = valid & mask\
    h_v, Y_v, s_v = h_axis[valid], Y[valid], stable[valid]\
    if len(h_v) == 0:\
        return\
    start = 0\
    first_solid, first_dashed = False, False\
    gap_tol = 5 * (H_MAX_PLOT / len(h_axis))\
    for i in range(1, len(h_v) + 1):\
        if i == len(h_v) or s_v[i] != s_v[start] or (h_v[i] - h_v[i - 1] > gap_tol):\
            seg_h, seg_Y = h_v[start:i], Y_v[start:i]\
            ls = '-' if s_v[start] else '--'\
            lw = 2.6 if s_v[start] else 1.8\
            lbl = None\
            if s_v[start] and not first_solid:\
                lbl = f'\{label\} (stable)'\
                first_solid = True\
            elif not s_v[start] and not first_dashed:\
                lbl = f'\{label\} (unstable)'\
                first_dashed = True\
            ax.plot(seg_h, seg_Y, ls, color=color, linewidth=lw, alpha=0.95,\
                     label=lbl, solid_capstyle='round')\
            start = i\
\
\
def make_two_panel_figure(save_path='bifurcation_diagram_2panel.png'):\
    h_axis = np.linspace(0.0, H_MAX_PLOT, 700)\
    br = compute_branches(h_axis)\
\
    zone_labels = ['Failed\\nmanagement', 'Partial\\nrecovery', 'Kiwi bonus\\nzone',\
                   'Full\\nsuppression', 'PF2050 /\\neradication']\
    zone_colors = ['#f3d6d6', '#fbe6c8', '#f7f0b0', '#c9e8d4', '#c3ddef']\
    bounds = _zone_bounds()\
\
    fig, (axK, axS) = plt.subplots(2, 1, figsize=(11, 10), sharex=True,\
                                    gridspec_kw=\{'height_ratios': [1.1, 1]\})\
\
    for ax in (axK, axS):\
        for (lo, hi), color in zip(bounds, zone_colors):\
            ax.axvspan(lo, hi, color=color, alpha=0.55, zorder=0)\
        for hv, c in [(h_fold, '#c0392b'), (h_crit, '#e67e22'),\
                      (h_erad, '#2980b9'), (h_erad_theo, '#7f8c8d')]:\
            ax.axvline(hv, color=c, linestyle=':', linewidth=1.2, alpha=0.8)\
\
    # --- Top panel: K* ---\
    colors = \{'A': '#7a4fa3', 'B': '#c0392b', 'C': '#2980b9', 'D': '#1e8449'\}\
    names  = \{'A': 'A: total extinction', 'B': 'B: stoat-floor',\
              'C': 'C: full suppression', 'D': 'D: coexistence'\}\
    for key in ['A', 'B', 'C', 'D']:\
        d = br[key]\
        plot_branch(axK, h_axis, d['K'], d['stable'], colors[key], names[key], mask=d['mask'])\
    for (lo, hi), lbl in zip(bounds, zone_labels):\
        axK.text((lo + hi) / 2, K_max * 1.1, lbl, ha='center', va='bottom',\
                  fontsize=8.5, color='#444')\
    axK.set_ylabel('Equilibrium kiwi\\npopulation K*', fontsize=11.5)\
    axK.set_ylim(-15, K_max * 1.22)\
    axK.set_title('Bifurcation diagram: kiwi (K*) and stoat (S*) equilibrium branches vs harvest rate h\\n'\
                  'solid = stable, dashed = unstable', fontsize=12.5)\
    axK.legend(loc='center left', bbox_to_anchor=(1.01, 0.5), fontsize=8.5, framealpha=0.95)\
    axK.grid(True, alpha=0.25)\
\
    # --- Bottom panel: S* ---\
    # NOTE: Branches A and C both sit at exactly S*=0 for every h (only K\
    # differs, shown in the top panel). Branch A is offset by a small fixed\
    # amount here PURELY for visibility -- true value is S*=0, identical to C.\
    OFFSET = -0.10\
    plot_branch(axS, h_axis, br['A']['S'] + OFFSET, br['A']['stable'], colors['A'], names['A'])\
    plot_branch(axS, h_axis, br['B']['S'], br['B']['stable'], colors['B'], names['B'], mask=br['B']['mask'])\
    plot_branch(axS, h_axis, br['C']['S'], br['C']['stable'], colors['C'], names['C'])\
    plot_branch(axS, h_axis, br['D']['S'], br['D']['stable'], colors['D'], names['D'])\
    axS.set_ylabel('Equilibrium stoat\\npopulation S*', fontsize=11.5)\
    axS.set_xlabel('Stoat harvest rate h (annual proportion removed)', fontsize=12)\
    axS.set_ylim(-0.55, S_max * 0.55)\
    axS.grid(True, alpha=0.25)\
    axS.text(0.01, OFFSET - 0.06,\
              'A offset slightly below 0 for visibility only \'97 true S*=0, identical to C',\
              fontsize=7.3, color=colors['A'], ha='left', va='top', style='italic')\
\
    label_y = S_max * 0.50\
    for hv, lbl, c, dx in [(h_fold, f'h_fold=\{h_fold:.3f\}', '#c0392b', -0.005),\
                            (h_crit, f'h_crit=\{h_crit:.3f\}', '#e67e22', 0.008),\
                            (h_erad, f'h_erad=\{h_erad:.3f\}', '#2980b9', 0.008),\
                            (h_erad_theo, f'h_erad_theo=\{h_erad_theo:.2f\}', '#7f8c8d', -0.005)]:\
        axS.text(hv + dx, label_y, lbl, rotation=90, ha='center', va='center',\
                  fontsize=8, color=c, backgroundcolor='white')\
\
    axS.set_xlim(0, H_MAX_PLOT)\
    plt.tight_layout()\
    plt.savefig(save_path, dpi=160, bbox_inches='tight')\
    print(f"Saved: \{save_path\}")\
    return fig\
\
\
# ============================================================================\
# MAIN\
# ============================================================================\
\
if __name__ == '__main__':\
    print_thresholds()\
    make_two_panel_figure()}