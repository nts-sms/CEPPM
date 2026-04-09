import numpy as np
from scipy.integrate import solve_ivp
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import warnings
warnings.filterwarnings('ignore')

# ── Core parameters ──────────────────────────────────────────────────────────
K0, S0, X0   = 20, 3, 0.3
R_STOAT      = 0.60
S_MAX        = 8
r            = 0.05
alpha        = 0.044
beta         = 0.0175
delta        = 0.35
K_MAX        = 150
h_handling   = 0.1
S_FLOOR      = S_MAX * (1 - delta / R_STOAT)   # 3.333
H_CRIT       = R_STOAT - delta                  # 0.25
H_ERAD       = R_STOAT                          # 0.60
BASE_PAYOFFS = np.array([[0.35, 0.55],
                         [0.50, 0.22]])

# ── EGT helpers ──────────────────────────────────────────────────────────────
def payoff_A(x, pm):
    return x * pm[0,0] + (1-x) * pm[0,1]

def payoff_B(x, pm):
    return x * pm[1,0] + (1-x) * pm[1,1]

def avg_payoff(x, pm):
    return x * payoff_A(x, pm) + (1-x) * payoff_B(x, pm)

def dynamic_payoffs(S):
    k, S_mid, open_max, cov_max = 3.0, 1.0, 0.40, 0.15
    sig      = 1 / (1 + np.exp(-k * (S - S_mid)))
    open_pen = open_max * sig
    cov_pen  = cov_max  * sig
    return np.array([
        [max(BASE_PAYOFFS[0,0] - open_pen, 0.01),
         max(BASE_PAYOFFS[0,1] - open_pen, 0.01)],
        [max(BASE_PAYOFFS[1,0] - cov_pen,  0.01),
         max(BASE_PAYOFFS[1,1] - cov_pen,  0.01)]
    ])

def strategy_params(x):
    avg_vuln = x * 1.3 + (1-x) * 0.7
    avg_repr = avg_payoff(x, BASE_PAYOFFS)
    return r * (avg_repr / 1.5), alpha * avg_vuln   # r_eff, alpha_eff

# ── Hybrid ODE system ────────────────────────────────────────────────────────
def hybrid_system(t, y, harvest_rate=0):
    x, K, S = y
    x = np.clip(x, 0.01, 0.99)
    K = max(K, 0)
    S = max(S, 0)

    dp     = dynamic_payoffs(S)
    dxdt   = x * (payoff_A(x, dp) - avg_payoff(x, dp)) if 0 < x < 1 else 0

    r_eff, alpha_eff = strategy_params(x)
    f_K    = (alpha_eff * K) / (1 + alpha_eff * h_handling * K)
    dKdt   = r_eff * K * (1 - K / K_MAX) - f_K * S

    nat    = R_STOAT * S * (1 - S / S_MAX) + beta * f_K * S - delta * S
    nkg    = R_STOAT * S * (1 - S / S_MAX) - delta * S
    if S <= S_FLOOR and nkg < 0:
        nat = max(nat, 0.0)
    dSdt   = nat - harvest_rate * S

    return [dxdt, dKdt, dSdt]

# ── Simulate one scenario ─────────────────────────────────────────────────────
def simulate(intervention_time, harvest_rate, t_max=200):
    y0     = [X0, K0, S0]
    t_eval = np.linspace(0, t_max, 2000)

    if intervention_time is None or intervention_time >= t_max:
        sol = solve_ivp(lambda t,y: hybrid_system(t, y, 0),
                        [0, t_max], y0, t_eval=t_eval, method='RK45',
                        rtol=1e-8, atol=1e-10)
        return sol.t, sol.y[0], sol.y[1], sol.y[2]

    t1 = np.linspace(0, intervention_time, max(400, int(intervention_time*20)))
    t2 = np.linspace(intervention_time, t_max, max(800, int((t_max-intervention_time)*20)))

    s1 = solve_ivp(lambda t,y: hybrid_system(t, y, 0),
                   [0, intervention_time], y0, t_eval=t1,
                   method='RK45', rtol=1e-8, atol=1e-10)
    s2 = solve_ivp(lambda t,y: hybrid_system(t, y, harvest_rate),
                   [intervention_time, t_max],
                   [s1.y[0][-1], s1.y[1][-1], s1.y[2][-1]],
                   t_eval=t2, method='RK45', rtol=1e-8, atol=1e-10)

    return (np.concatenate([s1.t, s2.t]),
            np.concatenate([s1.y[0], s2.y[0]]),
            np.concatenate([s1.y[1], s2.y[1]]),
            np.concatenate([s1.y[2], s2.y[2]]))

# ── 4 scenarios ───────────────────────────────────────────────────────────────
scenarios = [
    ('No control',                    None, 0.0),
    ('Intervene yr 20 | h=0.20',      20,   0.20),
    ('Intervene yr 10 | h=0.40',      10,   0.40),
    ('Intervene yr 2  | h=0.60',       2,   0.60),
]

SNAPSHOTS = [0, 10, 25, 50, 100, 200]

print("\nRunning 4 scenarios...")
results = {}
for label, it, hr in scenarios:
    t, x, K, S = simulate(it, hr, t_max=200)
    results[label] = {'t': t, 'x': x, 'K': K, 'S': S,
                      'it': it, 'hr': hr}
    print(f"  {label}: done  (final K={K[-1]:.1f}, S={S[-1]:.3f}, x={x[-1]:.3f})")

# ── Comparison table ─────────────────────────────────────────────────────────
def interp(arr_t, arr_v, t_query):
    return float(np.interp(t_query, arr_t, arr_v))

header_row = f"{'Scenario':<32} {'t':>5} {'x (open%)':>10} {'K (kiwi)':>10} {'S (stoats)':>10}"
sep        = "-" * len(header_row)

print("\n" + "="*len(header_row))
print("COMPARISON TABLE: Hybrid CEPPM Model")
print(f"S_floor={S_FLOOR:.3f}  h_crit={H_CRIT:.2f}  h_erad={H_ERAD:.2f}")
print("="*len(header_row))
print(header_row)
print(sep)

table_data = {}
for label, it, hr in scenarios:
    r_ = results[label]
    first = True
    row_vals = []
    for t_q in SNAPSHOTS:
        x_v = interp(r_['t'], r_['x'], t_q)
        K_v = interp(r_['t'], r_['K'], t_q)
        S_v = interp(r_['t'], r_['S'], t_q)
        row_vals.append((t_q, x_v, K_v, S_v))
        lbl = label if first else ""
        print(f"{lbl:<32} {t_q:>5}  {x_v*100:>8.1f}%  {K_v:>10.1f}  {S_v:>10.3f}")
        first = False
    print(sep)
    table_data[label] = row_vals

# ── Critical intervention time analysis ──────────────────────────────────────
print("\n" + "="*60)
print("CRITICAL INTERVENTION TIME ANALYSIS")
print(f"Testing harvest rates: h=0.20 (below h_crit), h=0.40, h=0.60")
print(f"Metric: kiwi population at t=200")
print("="*60)

crit_results = {}
for hr_test, hr_label in [(0.20, 'h=0.20 (< h_crit)'),
                           (0.40, 'h=0.40'),
                           (0.60, 'h=0.60 (= h_erad)')]:
    K_finals = []
    it_range = list(range(1, 101, 2))
    for it_test in it_range:
        t_, x_, K_, S_ = simulate(it_test, hr_test, t_max=200)
        K_finals.append(K_[-1])
    crit_results[hr_label] = (it_range, K_finals)

    # Find threshold: intervention time above which K_final < K0 (population lost)
    threshold = None
    for i, (it_t, kf) in enumerate(zip(it_range, K_finals)):
        if kf < K0 and i > 0:
            threshold = it_t
            break
    if threshold:
        print(f"  {hr_label}: kiwi fall below K0={K0} if intervention delayed beyond yr ~{threshold}")
    else:
        max_kf = max(K_finals)
        min_kf = min(K_finals)
        print(f"  {hr_label}: K_final range {min_kf:.1f}–{max_kf:.1f} (no recovery threshold found in range)")

print("\nDone — generating plots...")

# ── Plotting ──────────────────────────────────────────────────────────────────
colors  = ['#d62728', '#ff7f0e', '#2ca02c', '#1f77b4']
lstyles = ['-',       '--',      '--',      '--']

fig = plt.figure(figsize=(18, 14))
gs  = gridspec.GridSpec(3, 3, figure=fig, hspace=0.42, wspace=0.35)

t_end = 200

# ── Row 1: x, K, S time series ───────────────────────────────────────────────
ax_x = fig.add_subplot(gs[0, 0])
ax_K = fig.add_subplot(gs[0, 1])
ax_S = fig.add_subplot(gs[0, 2])

for (label, it, hr), col, ls in zip(scenarios, colors, lstyles):
    r_ = results[label]
    ax_x.plot(r_['t'], r_['x']*100, color=col, lw=2, ls=ls, label=label)
    ax_K.plot(r_['t'], r_['K'],     color=col, lw=2, ls=ls, label=label)
    ax_S.plot(r_['t'], r_['S'],     color=col, lw=2, ls=ls, label=label)

for ax in [ax_x, ax_K, ax_S]:
    ax.set_xlim(0, t_end)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=7.5, loc='upper right')
    ax.set_xlabel('Time (years)', fontsize=10)

ax_x.set_ylabel('Open foraging (%)', fontsize=10)
ax_x.set_title('Foraging strategy (x)', fontsize=11)
ax_x.set_ylim(0, 100)

ax_K.axhline(K_MAX,  color='green', ls=':', lw=1.2, alpha=0.5, label=f'K_max={K_MAX}')
ax_K.axhline(K0,     color='gray',  ls=':', lw=1.2, alpha=0.5, label=f'K0={K0}')
ax_K.set_ylabel('Kiwi population', fontsize=10)
ax_K.set_title('Kiwi population (K)', fontsize=11)

ax_S.axhline(S_FLOOR, color='darkred', ls=':', lw=1.5, alpha=0.7,
             label=f'S_floor={S_FLOOR:.2f}')
ax_S.set_ylabel('Stoat population', fontsize=10)
ax_S.set_title('Stoat population (S)', fontsize=11)

# ── Row 2: critical intervention time curves ──────────────────────────────────
ax_c0 = fig.add_subplot(gs[1, 0])
ax_c1 = fig.add_subplot(gs[1, 1])
ax_c2 = fig.add_subplot(gs[1, 2])

crit_axes  = [ax_c0, ax_c1, ax_c2]
crit_cols  = ['#ff7f0e', '#2ca02c', '#1f77b4']
crit_items = list(crit_results.items())

for ax, (hr_label, (it_range, K_finals)), col in zip(crit_axes, crit_items, crit_cols):
    ax.plot(it_range, K_finals, color=col, lw=2.5)
    ax.axhline(K0,    color='gray',  ls=':', lw=1.5, alpha=0.8,
               label=f'Initial K0={K0}')
    ax.axhline(K_MAX, color='green', ls=':', lw=1.2, alpha=0.5,
               label=f'K_max={K_MAX}')
    ax.axhline(0,     color='black', ls='-', lw=0.8, alpha=0.3)

    # Mark h_crit annotation on h=0.20 plot
    if '0.20' in hr_label:
        ax.text(50, K_finals[25], f'h < h_crit ({H_CRIT:.2f})\nInsufficient — kiwi lost\nregardless of timing',
                fontsize=8, color='#ff7f0e',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='wheat', alpha=0.7))

    ax.set_xlabel('Intervention year', fontsize=10)
    ax.set_ylabel(f'Kiwi at t=200', fontsize=10)
    ax.set_title(f'Critical timing: {hr_label}', fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(1, 100)
    ax.set_ylim(-5, K_MAX + 10)

# ── Row 3: phase portraits K vs S for each scenario ──────────────────────────
ax_ph = fig.add_subplot(gs[2, :])

for (label, it, hr), col, ls in zip(scenarios, colors, lstyles):
    r_ = results[label]
    ax_ph.plot(r_['K'], r_['S'], color=col, lw=2, ls=ls, label=label, alpha=0.85)
    ax_ph.plot(r_['K'][0], r_['S'][0], 'o', color=col, ms=7)
    ax_ph.plot(r_['K'][-1], r_['S'][-1], 's', color=col, ms=8)

ax_ph.axhline(S_FLOOR, color='darkred', ls='--', lw=1.5, alpha=0.7,
              label=f'S_floor={S_FLOOR:.2f} (stoat non-kiwi equilibrium)')
ax_ph.axhline(H_CRIT,  color='orange',  ls=':',  lw=1.2, alpha=0.6,
              label=f'h_crit={H_CRIT:.2f} reference')
ax_ph.set_xlabel('Kiwi population (K)', fontsize=11)
ax_ph.set_ylabel('Stoat population (S)', fontsize=11)
ax_ph.set_title('Phase portrait — all scenarios  (circle=start, square=end at t=200)',
                fontsize=11)
ax_ph.legend(fontsize=9)
ax_ph.grid(True, alpha=0.3)

fig.suptitle(
    'CEPPM Scenario Analysis  |  S_floor=3.33  h_crit=0.25  h_erad=0.60\n'
    'Rows: (1) time series  (2) critical intervention timing  (3) phase portrait',
    fontsize=12, fontweight='bold'
)

plt.savefig('/mnt/user-data/outputs/ceppm_scenario_analysis.png',
            dpi=150, bbox_inches='tight')
print("Plot saved.")
