# -*- coding: utf-8 -*-
"""
CEPPM Pulsed-h Simulation
==========================
Branch of kiwi_conservation_analysis_v7.py for exploring pulsed (episodic)
stoat harvest versus continuous harvest at the same mean annual removal rate.

Motivation: DOC 1080 aerial operations are not continuous — they are short,
intensive pulses every 2-3 years (triggered by beech mast events). This
script asks whether the temporal concentration of control effort matters,
i.e. does pulsing at the same mean h produce the same long-run outcome?

DO NOT MODIFY kiwi_conservation_analysis_v7.py. This file is a standalone
branch. It duplicates only the parameters and classes it needs from v7.

Model version: v7 (a=0.35, b=0.55, c=0.45, d=0.20; kappa_learn=1.5/yr)
Author branch: pulsed-h exploration — not for publication

Pulsed-h design
---------------
Mean-equivalence: a pulse of height h_peak lasting pulse_dur years every
pulse_interval years delivers mean annual removal rate:

    h_mean = h_peak * pulse_dur / pulse_interval

So to match continuous h_mean = 0.20:
  - Annual pulse (interval=1):  h_peak = 0.20 * 1 / 0.1  = 2.0  for 0.1 yr
  - DOC-realistic (interval=2.5): h_peak = 0.20 * 2.5 / 0.1 = 5.0 for 0.1 yr

Three scenarios compared at mean h = 0.20, intervention year 20, T=200 yr:
  A) Continuous h = 0.20 (Scenario 2a baseline)
  B) Annual pulse: h_peak=2.0 for 5-6 weeks (0.1 yr) every year
  C) DOC-realistic pulse: h_peak=5.0 for 5-6 weeks (0.1 yr) every 2.5 years

Expected behaviour (pre-run hypothesis):
  - S(t) oscillates in pulsed cases; amplitude grows with interval
  - Between-pulse stoat rebounds may limit kiwi recovery relative to continuous
  - K(t=200) likely lower under pulsed, more so for longer intervals
  - x(t) should track S oscillations (more cover during rebounds)
  - Analytical thresholds (h_crit, h_erad) have no direct pulsed analogue;
    effective threshold will differ from continuous case
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# ============================================================================
# V7 PARAMETERS — copied from kiwi_conservation_analysis_v7.py
# Do not alter these; they are the locked v7 invariants.
# ============================================================================

# Initial conditions
X0   = 0.30   # Initial open-foraging proportion
K0   = 20.0   # Initial kiwi population (per 1,000 ha)
S0   = 3.0    # Initial stoat population (per 1,000 ha)

# L-V parameters
R_KIWI  = 0.05    # Kiwi intrinsic growth rate
ALPHA   = 0.044   # Predation rate
BETA    = 0.0175  # Stoat conversion efficiency (kiwi bonus)
DELTA   = 0.35    # Stoat natural death rate
K_MAX   = 150.0   # Kiwi carrying capacity
H_HOLL  = 0.1     # Holling Type II handling time (stoat saturation)
R_STOAT = 0.60    # Stoat intrinsic growth from non-kiwi prey
S_MAX   = 8.0     # Stoat non-kiwi carrying capacity
S_FLOOR = S_MAX * (1 - DELTA / R_STOAT)  # = 3.333

# EGT parameters
KAPPA_LEARN = 1.5   # Behavioural adaptation rate (yr^-1)
BASE_PAYOFFS = np.array([[0.35, 0.55],   # Open: a, b
                         [0.45, 0.20]])  # Cover: c, d

# Sigmoid penalty parameters
K_SIG   = 3.0   # Steepness
S_MID   = 1.0   # Inflection point (stoats/1,000 ha)
OPEN_MAX = 0.30  # Max open-foraging penalty
COV_MAX  = 0.15  # Max cover-foraging penalty

# Vulnerability parameters
OPEN_VUL  = 1.3
COVER_VUL = 0.7

# Derived EGT quantities (v7 invariants)
a, b = BASE_PAYOFFS[0]
c, d = BASE_PAYOFFS[1]
X_STAR     = (d - b) / ((a - b) + (d - c))   # = 7/9 = 0.7778
PI_BAR_ESS = X_STAR * (X_STAR * a + (1 - X_STAR) * b) + \
             (1 - X_STAR) * (X_STAR * c + (1 - X_STAR) * d)

# Simulation settings
T_MAX            = 200
INTERVENTION_YR  = 20
H_MEAN           = 0.25    # Mean annual harvest rate (all scenarios)
PULSE_DUR        = 0.10    # Pulse duration in years (~5-6 weeks)

# Mast event parameters — applied to ALL scenarios as environmental forcing.
# Beech mast events drive rodent irruptions, followed by stoat irruptions.
# Modelled as a temporary elevation of S_max (the stoat non-kiwi carrying
# capacity), repeating every MAST_CYCLE years, lasting MAST_DUR years.
# The irruption precedes the 1080 response by MAST_LAG years (scenario D only).
MAST_CYCLE    = 5.0    # Years between mast events
MAST_DUR      = 1.0    # Duration of elevated S_max (yr) — irruption window
MAST_LAG      = 0.5    # Lag from irruption onset to 1080 response (yr)
S_MAX_MAST    = 25.0   # Elevated stoat carrying capacity during mast
                       # (vs baseline S_MAX=8.0; reflects rodent prey explosion)
MAST_FIRST    = 5.0    # First mast event (pre-intervention; always present)

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def dynamic_payoffs(S):
    """Sigmoid predation penalty on base payoffs (identical to v7)."""
    sig      = 1.0 / (1.0 + np.exp(-K_SIG * (S - S_MID)))
    open_pen = OPEN_MAX * sig
    cov_pen  = COV_MAX  * sig
    pm = np.array([
        [max(BASE_PAYOFFS[0, 0] - open_pen, 0.01),
         max(BASE_PAYOFFS[0, 1] - open_pen, 0.01)],
        [max(BASE_PAYOFFS[1, 0] - cov_pen,  0.01),
         max(BASE_PAYOFFS[1, 1] - cov_pen,  0.01)]
    ])
    return pm


def strategy_params(x, pm_base=BASE_PAYOFFS):
    """Effective r and alpha from current strategy mix (identical to v7)."""
    avg_vul = x * OPEN_VUL + (1 - x) * COVER_VUL
    pi_bar_x = (x * (x * pm_base[0,0] + (1-x) * pm_base[0,1]) +
                (1-x) * (x * pm_base[1,0] + (1-x) * pm_base[1,1]))
    r_eff     = R_KIWI * (pi_bar_x / PI_BAR_ESS)
    alpha_eff = ALPHA * avg_vul
    return r_eff, alpha_eff


def predation_rate(alpha_eff, K):
    """Holling Type II functional response (identical to v7)."""
    return (alpha_eff * K) / (1.0 + alpha_eff * H_HOLL * K)


def pulsed_h(t, t_intervention, h_peak, pulse_dur, pulse_interval):
    """
    Time-dependent harvest rate for pulsed control.

    Returns h_peak during each pulse window, 0 otherwise.
    Pulses begin at t_intervention and repeat every pulse_interval years.

    Parameters
    ----------
    t               : current time (yr)
    t_intervention  : year control begins
    h_peak          : peak harvest rate during pulse
    pulse_dur       : duration of each pulse (yr)
    pulse_interval  : time between pulse starts (yr)

    Mean annual equivalent: h_mean = h_peak * pulse_dur / pulse_interval
    """
    if t < t_intervention:
        return 0.0
    t_since = (t - t_intervention) % pulse_interval
    return h_peak if t_since < pulse_dur else 0.0


def mast_S_max(t):
    """
    Time-dependent stoat carrying capacity reflecting beech mast irruptions.

    Applied to ALL scenarios — the mast is an environmental event independent
    of control strategy. Every MAST_CYCLE years (starting at MAST_FIRST),
    S_max rises to S_MAX_MAST for MAST_DUR years, then returns to S_MAX.

    This captures the rodent prey base explosion that precedes stoat irruptions:
    higher S_max allows stoats to grow faster and to a higher ceiling, stressing
    kiwi even if control is ongoing.
    """
    if t < MAST_FIRST:
        return S_MAX
    t_since = (t - MAST_FIRST) % MAST_CYCLE
    return S_MAX_MAST if t_since < MAST_DUR else S_MAX


# ============================================================================
# ODE SYSTEM
# ============================================================================

def ceppm_system(t, y, harvest_fn, s_max_fn=None):
    """
    Full CEPPM ODE system (EGT + extended L-V), identical to v7 hybrid_system
    except:
      - harvest_rate replaced by callable harvest_fn(t)
      - S_max replaced by callable s_max_fn(t) for mast irruption forcing

    Parameters
    ----------
    t          : time
    y          : [x, K, S]
    harvest_fn : callable returning instantaneous harvest rate at time t
    s_max_fn   : callable returning S_max at time t (default: constant S_MAX)
    """
    if s_max_fn is None:
        s_max_fn = lambda t: S_MAX
    x, K, S = y
    x = np.clip(x, 0.01, 0.99)
    K = max(K, 0.0)
    S = max(S, 0.0)

    # --- EGT replicator ---
    dp   = dynamic_payoffs(S)
    pi_A = x * dp[0, 0] + (1 - x) * dp[0, 1]
    pi_B = x * dp[1, 0] + (1 - x) * dp[1, 1]
    pi_avg = x * pi_A + (1 - x) * pi_B
    dxdt = KAPPA_LEARN * x * (pi_A - pi_avg) if 0.0 < x < 1.0 else 0.0

    # --- L-V population dynamics ---
    r_eff, alpha_eff = strategy_params(x)
    f_K  = predation_rate(alpha_eff, K)
    dKdt = r_eff * K * (1.0 - K / K_MAX) - f_K * S

    # Stoat: non-kiwi logistic + kiwi bonus - natural mortality - harvest
    # S_max is time-dependent to capture mast irruptions
    s_max_now = s_max_fn(t)
    s_floor_now = s_max_now * (1.0 - DELTA / R_STOAT)

    natural_dSdt = (R_STOAT * S * (1.0 - S / s_max_now)
                    + BETA * f_K * S
                    - DELTA * S)
    # Floor clamp: non-kiwi growth can't drive stoats below current S_floor
    non_kiwi_growth = R_STOAT * S * (1.0 - S / s_max_now) - DELTA * S
    if S <= s_floor_now and non_kiwi_growth < 0:
        natural_dSdt = max(natural_dSdt, 0.0)

    h_now = harvest_fn(t)
    dSdt  = natural_dSdt - h_now * S

    return [dxdt, dKdt, dSdt]


# ============================================================================
# SIMULATION RUNNER
# ============================================================================

def run_simulation(harvest_fn, label, s_max_fn=None, t_max=T_MAX,
                   rtol=1e-9, atol=1e-11):
    """
    Integrate the CEPPM system with a given harvest function and optional
    time-varying S_max (mast irruption forcing).
    """
    if s_max_fn is None:
        s_max_fn = mast_S_max   # All scenarios share the same mast forcing

    # Fine time grid so we don't miss short pulses or mast transitions
    t_eval = np.linspace(0, t_max, 5000)

    sol = solve_ivp(
        lambda t, y: ceppm_system(t, y, harvest_fn, s_max_fn),
        [0, t_max],
        [X0, K0, S0],
        t_eval=t_eval,
        method='RK45',
        rtol=rtol,
        atol=atol,
        max_step=PULSE_DUR / 4   # Force solver to step inside pulse windows
    )

    if not sol.success:
        print(f"  WARNING: solver did not converge for '{label}': {sol.message}")

    print(f"  {label:45s}  "
          f"x@200={sol.y[0,-1]*100:.1f}%  "
          f"K@200={sol.y[1,-1]:.1f}  "
          f"S@200={sol.y[2,-1]:.3f}")

    return sol.t, sol.y[0], sol.y[1], sol.y[2]


# ============================================================================
# MAIN ANALYSIS
# ============================================================================

def run_pulsed_comparison():

    print("=" * 70)
    print("CEPPM PULSED-h SIMULATION WITH MAST IRRUPTION FORCING")
    print(f"  Mean h = {H_MEAN:.2f} for all scenarios  |  "
          f"Intervention yr = {INTERVENTION_YR}  |  T = {T_MAX} yr")
    print(f"  Pulse duration = {PULSE_DUR} yr ({PULSE_DUR*52:.0f} weeks)")
    print(f"  Mast cycle: every {MAST_CYCLE} yr  |  "
          f"S_max baseline={S_MAX}  →  S_max mast={S_MAX_MAST}  |  "
          f"Duration={MAST_DUR} yr")
    print(f"  Mast irruption applied to ALL scenarios  |  "
          f"1080 lag (scenario D only) = {MAST_LAG} yr")
    print("=" * 70)

    # --- Define harvest functions ---

    # A) Continuous h
    def h_continuous(t):
        return H_MEAN if t >= INTERVENTION_YR else 0.0

    # B) Annual pulse — mean-equivalent
    H_PEAK_ANNUAL = H_MEAN * 1.0 / PULSE_DUR
    def h_annual(t):
        return pulsed_h(t, INTERVENTION_YR, H_PEAK_ANNUAL,
                        pulse_dur=PULSE_DUR, pulse_interval=1.0)

    # C) DOC-realistic pulse — every 2.5 years, NOT mast-synchronised
    DOC_INTERVAL  = 2.5
    H_PEAK_DOC    = H_MEAN * DOC_INTERVAL / PULSE_DUR
    def h_doc(t):
        return pulsed_h(t, INTERVENTION_YR, H_PEAK_DOC,
                        pulse_dur=PULSE_DUR, pulse_interval=DOC_INTERVAL)

    # D) Mast-synchronised pulse — fires MAST_LAG years after each irruption onset.
    #    The h pulse is tied to the mast cycle, not a fixed calendar interval.
    #    h_peak scaled to deliver same mean annual removal as other scenarios.
    H_PEAK_MAST = H_MEAN * MAST_CYCLE / PULSE_DUR
    def h_mast(t):
        if t < INTERVENTION_YR:
            return 0.0
        if t < MAST_FIRST:
            return 0.0
        # Fire pulse MAST_LAG years after each mast onset
        t_since = (t - MAST_FIRST) % MAST_CYCLE
        pulse_start = MAST_LAG
        pulse_end   = MAST_LAG + PULSE_DUR
        return H_PEAK_MAST if pulse_start <= t_since < pulse_end else 0.0

    scenarios = [
        (h_continuous, f"A) Continuous  h={H_MEAN:.2f}",                             'navy',      'solid'),
        (h_annual,     f"B) Annual pulse  h_peak={H_PEAK_ANNUAL:.1f}  (1 yr)",       'darkorange','dashed'),
        (h_doc,        f"C) DOC pulse     h_peak={H_PEAK_DOC:.1f}  (2.5 yr)",       'seagreen',  'dashdot'),
        (h_mast,       f"D) Mast-sync     h_peak={H_PEAK_MAST:.1f}  (5 yr + lag)",  'crimson',   'dotted'),
    ]

    print(f"\n  Mean-equivalence check (all = h_mean {H_MEAN:.2f}):")
    print(f"    A  h_mean = {H_MEAN:.2f}  (continuous)")
    print(f"    B  h_mean = {H_PEAK_ANNUAL:.1f} × {PULSE_DUR} / 1.0 = "
          f"{H_PEAK_ANNUAL * PULSE_DUR / 1.0:.2f}")
    print(f"    C  h_mean = {H_PEAK_DOC:.1f} × {PULSE_DUR} / {DOC_INTERVAL} = "
          f"{H_PEAK_DOC * PULSE_DUR / DOC_INTERVAL:.2f}")
    print(f"    D  h_mean = {H_PEAK_MAST:.1f} × {PULSE_DUR} / {MAST_CYCLE} = "
          f"{H_PEAK_MAST * PULSE_DUR / MAST_CYCLE:.2f}  (mast-synchronised)")
    print(f"\n  Running simulations (rtol=1e-9, max_step={PULSE_DUR/4:.3f} yr):\n")

    results = []
    for fn, label, colour, ls in scenarios:
        t, x, K, S = run_simulation(fn, label)
        results.append((t, x, K, S, label, colour, ls))

    # ---- Stoat rebound stats for pulsed scenarios -------------------------
    print("\n  Stoat rebound analysis (between-pulse peaks, post-intervention):")
    for i, (t, x, K, S, label, _, _) in enumerate(results[1:], start=1):
        mask = t >= INTERVENTION_YR
        S_post = S[mask]
        t_post = t[mask]
        print(f"    {label[:40]:<40s}  "
              f"S_min={S_post.min():.3f}  S_max={S_post.max():.3f}  "
              f"S@200={S_post[-1]:.3f}")

    # ====================================================================
    # PLOT
    # ====================================================================
    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
    ax_S, ax_K, ax_x = axes

    for t, x, K, S, label, colour, ls in results:
        lw = 2.2 if 'Continuous' in label else 1.8
        ax_S.plot(t, S, color=colour, linestyle=ls, linewidth=lw, label=label)
        ax_K.plot(t, K, color=colour, linestyle=ls, linewidth=lw, label=label)
        ax_x.plot(t, x * 100, color=colour, linestyle=ls, linewidth=lw, label=label)

    # Reference lines
    ax_S.axhline(S_FLOOR, color='grey', linestyle=':', linewidth=1.2,
                 label=f'S_floor = {S_FLOOR:.3f}')
    ax_K.axhline(K0, color='grey', linestyle=':', linewidth=1.2,
                 label=f'K₀ = {K0:.0f}')
    ax_x.axhline(X_STAR * 100, color='grey', linestyle=':', linewidth=1.2,
                 label=f'x* = {X_STAR*100:.1f}%')

    # Intervention marker + mast event shading
    mast_times = np.arange(MAST_FIRST, T_MAX, MAST_CYCLE)
    for ax in axes:
        ax.axvline(INTERVENTION_YR, color='black', linestyle='--',
                   linewidth=1.0, alpha=0.5)
        for mt in mast_times:
            ax.axvspan(mt, mt + MAST_DUR, color='gold', alpha=0.18,
                       label='Mast irruption' if mt == mast_times[0] else '')

    ax_S.set_ylabel('Stoat density S\n(per 1,000 ha)', fontsize=11)
    ax_K.set_ylabel('Kiwi population K\n(per 1,000 ha)', fontsize=11)
    ax_x.set_ylabel('Open foraging\nproportion x (%)', fontsize=11)
    ax_x.set_xlabel('Time (years)', fontsize=11)

    ax_S.set_title(
        f'CEPPM Pulsed-h + Mast Irruption  |  Mean h = {H_MEAN:.2f}  |  '
        f'Intervention yr {INTERVENTION_YR}  |  T = {T_MAX} yr\n'
        f'Gold bands = mast irruption (S_max {S_MAX}→{S_MAX_MAST}, every {MAST_CYCLE:.0f} yr)',
        fontsize=11, fontweight='bold')

    for ax in axes:
        ax.legend(fontsize=8.5, loc='upper right')
        ax.grid(True, alpha=0.3)
        ax.tick_params(labelsize=10)

    # Zoom inset on S(t) for post-intervention oscillations
    ax_S_zoom = fig.add_axes([0.13, 0.695, 0.25, 0.12])
    for t, x, K, S, label, colour, ls in results:
        mask = (t >= INTERVENTION_YR) & (t <= INTERVENTION_YR + 30)
        ax_S_zoom.plot(t[mask], S[mask], color=colour, linestyle=ls,
                       linewidth=1.4)
    ax_S_zoom.axhline(S_FLOOR, color='grey', linestyle=':', linewidth=0.9)
    ax_S_zoom.set_title('S(t) first 30 yr post-intervention', fontsize=7)
    ax_S_zoom.tick_params(labelsize=6)
    ax_S_zoom.grid(True, alpha=0.25)

    plt.tight_layout(rect=[0, 0, 1, 1])
    plt.savefig('/mnt/user-data/outputs/pulsed_h_comparison.png',
                dpi=150, bbox_inches='tight')
    plt.show()
    print("\n  Figure saved: pulsed_h_comparison.png")


# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == '__main__':
    run_pulsed_comparison()
