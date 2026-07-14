# -*- coding: utf-8 -*-
"""
CEPPM — Consolidated Threshold Sensitivity: h_crit, h_erad, h_survival
========================================================================
Reviewer #2 follow-up: extends the existing h_crit/h_erad sensitivity
(sensitivity_thresholds_h_crit_h_erad.py) to also cover h_survival, and
sweeps all three thresholds against a wider parameter set: r_stoat, delta,
alpha, beta, r (kiwi growth), and -- for h_survival specifically, since it
alone depends on them -- the payoff matrix (a,b,c,d) and vulnerability
weights (v_open, v_cover).

KEY STRUCTURAL FACTS (see session handoff v21 for full derivation):
  h_crit = r_stoat - delta
  h_erad = h_crit + beta * f(K_max),  f(K) = alpha*K/(1+alpha*H*K)
  h_survival = r_stoat*(1 - S_crit/S_max) - delta
      where S_crit = [r / (alpha * PI_BAR_ESS)] * R_max
            R_max = max_x [ pi_bar(x) / vulnerability(x) ]
            pi_bar(x), PI_BAR_ESS, x* all derived from payoff matrix a,b,c,d

  -> h_crit depends ONLY on r_stoat, delta.
  -> h_erad additionally depends on alpha, beta (via Holling term) but NOT
     on r, or the payoff matrix / vulnerability weights.
  -> h_survival additionally depends on r, alpha, and the full EGT layer
     (a,b,c,d,v_open,v_cover) but NOT on beta (confirmed below).

VALIDITY MASKING -- each threshold gets its OWN validity mask; they are not
pooled, since e.g. an r_stoat value can perfectly well leave h_crit/h_erad
valid while h_survival goes negative (or vice versa):
  - h_crit, h_erad: require r_stoat > delta (stated model precondition,
    main script L219). This is the ONLY thing that can invalidate them
    across every sweep here (never triggered except in the r_stoat sweep,
    since all other sweeps hold r_stoat=0.6 > delta's full tested range).
  - h_survival: requires the EGT layer well-defined (x* in (0,1),
    PI_BAR_ESS>0, vulnerability>0 everywhere) AND 0 <= S_crit <= S_max AND
    h_survival >= 0 (negative values mean kiwi already invade at h=0 --
    a real regime, just not a harvest threshold).
"""

import numpy as np

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

x_fine = np.linspace(0.0, 1.0, 200001)


def f_holling(K, alpha, H=Hh):
    return alpha * K / (1 + alpha * H * K)


def compute_thresholds(r=r_base, alpha=alpha_b, beta=beta_b, delta=delta_b,
                        r_stoat=r_stoat_b, a=a_b, b=b_b, c=c_b, d=d_b,
                        v_open=v_open_b, v_cover=v_cover_b):
    """Returns dict with h_crit, h_erad, h_survival and THREE separate
    validity flags (valid_hc, valid_he share one condition; valid_hs its own)."""
    valid_lv = r_stoat > delta  # governs h_crit AND h_erad
    h_crit = r_stoat - delta
    h_erad = h_crit + beta * f_holling(K_max, alpha)

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
                valid_lv=valid_lv, valid_hs=valid_hs)


base = compute_thresholds()
print(f"BASELINE: h_crit={base['h_crit']:.4f}  h_erad={base['h_erad']:.4f}  "
      f"h_survival={base['h_survival']:.4f}\n")

RANGES = {
    'r_stoat': (np.linspace(0.30, 0.90, 25),  'Wide stress-test'),
    'delta':   (np.linspace(0.15, 0.55, 25),  'Wide stress-test'),
    'alpha':   (np.linspace(0.010, 0.100, 25), 'Wide stress-test'),
    'beta':    (np.linspace(0.005, 0.050, 25), 'Wide stress-test'),
    'r':       (np.linspace(0.02, 0.10, 25),  'Wide stress-test (no lit. anchor for r)'),
    'a':       (np.linspace(0.245, 0.455, 25), 'Wide stress-test (+/-30% baseline)'),
    'b':       (np.linspace(0.385, 0.715, 25), 'Wide stress-test (+/-30% baseline)'),
    'c':       (np.linspace(0.315, 0.585, 25), 'Wide stress-test (+/-30% baseline)'),
    'd':       (np.linspace(0.140, 0.260, 25), 'Wide stress-test (+/-30% baseline)'),
    'v_open':  (np.linspace(1.0, 1.8, 25),    'Existing convention (penalty/vuln script)'),
    'v_cover': (np.linspace(0.4, 0.9, 25),    'Existing convention (penalty/vuln script)'),
}

results = {}
for key, (vals, source) in RANGES.items():
    rows = [compute_thresholds(**{key: v}) for v in vals]
    hc = np.array([row['h_crit'] for row in rows])
    he = np.array([row['h_erad'] for row in rows])
    hs = np.array([row['h_survival'] for row in rows])
    vlv = np.array([row['valid_lv'] for row in rows])
    vhs = np.array([row['valid_hs'] for row in rows])
    results[key] = dict(vals=vals, hc=hc, he=he, hs=hs, vlv=vlv, vhs=vhs, source=source)

print(f"{'Param':<8} {'Range swept':<14} {'h_crit range':<20} {'h_erad range':<18} {'h_survival range':<22} {'hs span':>8}  notes")
print("-"*118)
for key, res in results.items():
    vals = res['vals']
    vr = f"{vals[0]:.3f}-{vals[-1]:.3f}"

    hc_v = res['hc'][res['vlv']]
    hc_span = f"{hc_v.min():.3f}-{hc_v.max():.3f}" if len(hc_v) and hc_v.max()-hc_v.min() > 1e-9 else "invariant (0.250)"

    he_v = res['he'][res['vlv']]
    he_span = f"{he_v.min():.3f}-{he_v.max():.3f}" if len(he_v) and he_v.max()-he_v.min() > 1e-9 else "invariant (0.320)"

    hs_v = res['hs'][res['vhs']]
    if len(hs_v) > 0:
        hs_span_str = f"{hs_v.min():.4f}-{hs_v.max():.4f}"
        hs_span_val = hs_v.max() - hs_v.min()
    else:
        hs_span_str, hs_span_val = "ALL INVALID", np.nan

    n_lv_excl = (~res['vlv']).sum()
    n_hs_excl = (~res['vhs']).sum()
    note = []
    if n_lv_excl: note.append(f"{n_lv_excl} excl.(h_crit/h_erad)")
    if n_hs_excl: note.append(f"{n_hs_excl} excl.(h_survival)")
    print(f"{key:<8} {vr:<14} {hc_span:<20} {he_span:<18} {hs_span_str:<22} {hs_span_val:8.4f}  {'; '.join(note)}")

print(f"\nRange sources:")
for key, (vals, source) in RANGES.items():
    print(f"  {key:<8}: {source}")

print(f"\nCross-check vs existing appendix table (r_stoat/delta/alpha/beta, h_crit/h_erad only):")
print(f"  r_stoat -> h_crit 0.025-0.550, h_erad 0.095-0.620  (expect exact match)")
print(f"  delta   -> h_crit 0.050-0.450, h_erad 0.120-0.520  (expect exact match)")
print(f"  alpha   -> h_crit invariant,   h_erad 0.273-0.355  (expect exact match)")
print(f"  beta    -> h_crit invariant,   h_erad 0.270-0.449  (expect exact match)")