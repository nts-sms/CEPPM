import numpy as np
from scipy.integrate import solve_ivp
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import warnings
warnings.filterwarnings('ignore')

# ── Fixed parameters ──────────────────────────────────────────────────────────
K0, S0, X0   = 20, 3, 0.3
alpha        = 0.044
beta         = 0.0175
K_MAX        = 150
h_handling   = 0.1
BASE_PAYOFFS = np.array([[0.35, 0.55],
                         [0.50, 0.22]])

HARVEST_RATE     = 0.40   # fixed at h_crit
INTERVENTION_YR  = 2
T_MAX            = 200

# ── Ranges to sweep ───────────────────────────────────────────────────────────
# Baseline: delta=0.35, r_stoat=0.60  →  h_crit = r_stoat - delta = 0.25
# We vary each independently to see how the *same* h=0.25 performs
DELTA_RANGE   = np.linspace(0.15, 0.55, 25)   # natural stoat mortality
RSTOAT_RANGE  = np.linspace(0.30, 0.90, 25)   # stoat intrinsic growth

# ── EGT helpers ───────────────────────────────────────────────────────────────
def avg_payoff(x, pm):
    pA = x*pm[0,0] + (1-x)*pm[0,1]
    pB = x*pm[1,0] + (1-x)*pm[1,1]
    return x*pA + (1-x)*pB

def dynamic_payoffs(S):
    k, S_mid, open_max, cov_max = 3.0, 1.0, 0.40, 0.15
    sig = 1 / (1 + np.exp(-k * (S - S_mid)))
    return np.array([
        [max(BASE_PAYOFFS[0,0] - open_max*sig, 0.01),
         max(BASE_PAYOFFS[0,1] - open_max*sig, 0.01)],
        [max(BASE_PAYOFFS[1,0] - cov_max*sig,  0.01),
         max(BASE_PAYOFFS[1,1] - cov_max*sig,  0.01)]
    ])

def strategy_params(x, r_base=0.05):
    avg_vuln = x*1.3 + (1-x)*0.7
    avg_repr = avg_payoff(x, BASE_PAYOFFS)
    return r_base * (avg_repr / 1.5), alpha * avg_vuln

# ── ODE system (parameterised) ────────────────────────────────────────────────
def make_system(delta, r_stoat):
    S_floor = 8 * (1 - delta/r_stoat) if r_stoat > delta else 0.0
    def system(t, y, hr=0):
        x, K, S = y
        x = np.clip(x, 0.01, 0.99)
        K = max(K, 0); S = max(S, 0)
        dp = dynamic_payoffs(S)
        pA = x*dp[0,0]+(1-x)*dp[0,1]
        pav= avg_payoff(x, dp)
        dxdt = x*(pA - pav) if 0 < x < 1 else 0
        r_eff, alpha_eff = strategy_params(x)
        f_K  = (alpha_eff*K)/(1+alpha_eff*h_handling*K)
        dKdt = r_eff*K*(1-K/K_MAX) - f_K*S
        nat  = r_stoat*S*(1-S/8) + beta*f_K*S - delta*S   # S_max=8 always
        nkg  = r_stoat*S*(1-S/8) - delta*S
        if S <= S_floor and nkg < 0:
            nat = max(nat, 0.0)
        dSdt = nat - hr*S
        return [dxdt, dKdt, dSdt]
    return system

def simulate(delta, r_stoat, hr=0.25, it=2, t_max=200):
    sys = make_system(delta, r_stoat)
    y0  = [X0, K0, S0]
    t1  = np.linspace(0, it, 200)
    t2  = np.linspace(it, t_max, 1800)
    try:
        s1 = solve_ivp(lambda t,y: sys(t,y,0),  [0,it],     y0,
                       t_eval=t1, method='RK45', rtol=1e-8, atol=1e-10)
        s2 = solve_ivp(lambda t,y: sys(t,y,hr), [it,t_max],
                       [s1.y[0][-1], s1.y[1][-1], s1.y[2][-1]],
                       t_eval=t2, method='RK45', rtol=1e-8, atol=1e-10)
        t_all = np.concatenate([s1.t, s2.t])
        x_all = np.concatenate([s1.y[0], s2.y[0]])
        K_all = np.concatenate([s1.y[1], s2.y[1]])
        S_all = np.concatenate([s1.y[2], s2.y[2]])
        x_f = float(np.interp(t_max, t_all, x_all))
        K_f = float(np.interp(t_max, t_all, K_all))
        S_f = float(np.interp(t_max, t_all, S_all))
        return x_f, K_f, S_f
    except:
        return np.nan, np.nan, np.nan

# ── 1D sweeps: vary delta, hold r_stoat=0.60 ─────────────────────────────────
print("Running 1D sweep: delta (r_stoat=0.60 fixed)...")
x_d, K_d, S_d, hcrit_d = [], [], [], []
for d in DELTA_RANGE:
    xf, Kf, Sf = simulate(d, 0.60)
    x_d.append(xf); K_d.append(Kf); S_d.append(Sf)
    hcrit_d.append(0.60 - d)   # h_crit = r_stoat - delta

print("Running 1D sweep: r_stoat (delta=0.35 fixed)...")
x_r, K_r, S_r, hcrit_r = [], [], [], []
for rs in RSTOAT_RANGE:
    xf, Kf, Sf = simulate(0.35, rs)
    x_r.append(xf); K_r.append(Kf); S_r.append(Sf)
    hcrit_r.append(rs - 0.35)

# ── 2D grid: both vary ────────────────────────────────────────────────────────
print("Running 2D grid (25×25)...")
X_grid = np.full((len(DELTA_RANGE), len(RSTOAT_RANGE)), np.nan)
K_grid = np.full_like(X_grid, np.nan)

for i, d in enumerate(DELTA_RANGE):
    for j, rs in enumerate(RSTOAT_RANGE):
        if rs <= d:          # stoats can't persist — skip
            X_grid[i,j] = np.nan
            K_grid[i,j] = np.nan
            continue
        xf, Kf, _ = simulate(d, rs)
        X_grid[i,j] = xf * 100   # as %
        K_grid[i,j] = Kf

print("Done. Building plots...")

# ── Print 1D table ────────────────────────────────────────────────────────────
print("\n" + "="*72)
print(f"1D SENSITIVITY: varying delta  (r_stoat=0.60, h=0.40, intervention yr=2, t=200)")
print(f"Baseline: delta=0.35 → h_crit=0.25 (h=h above h_crit)")
print("="*72)
print(f"{'delta':>8} {'h_crit':>8} {'h vs hcrit':>12} {'x% (open)':>12} {'K (kiwi)':>10} {'S (stoats)':>10}")
print("-"*72)
for d, hc, xf, Kf, Sf in zip(DELTA_RANGE, hcrit_d, x_d, K_d, S_d):
    rel = "= h_crit" if abs(hc - 0.40) < 0.01 else ("> h_crit" if hc < 0.40 else "< h_crit")
    print(f"  {d:6.3f}   {hc:6.3f}   {rel:>12}   {xf*100:8.1f}%   {Kf:8.1f}   {Sf:8.3f}")

print("\n" + "="*72)
print(f"1D SENSITIVITY: varying r_stoat  (delta=0.35, h=0.40, intervention yr=2, t=200)")
print(f"Baseline: r_stoat=0.60 → h_crit=0.25 (h=h above h_crit)")
print("="*72)
print(f"{'r_stoat':>8} {'h_crit':>8} {'h vs hcrit':>12} {'x% (open)':>12} {'K (kiwi)':>10} {'S (stoats)':>10}")
print("-"*72)
for rs, hc, xf, Kf, Sf in zip(RSTOAT_RANGE, hcrit_r, x_r, K_r, S_r):
    rel = "= h_crit" if abs(hc - 0.40) < 0.01 else ("> h_crit" if hc < 0.40 else "< h_crit")
    print(f"  {rs:6.3f}   {hc:6.3f}   {rel:>12}   {xf*100:8.1f}%   {Kf:8.1f}   {Sf:8.3f}")

# ── Plots ─────────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(18, 16))
gs  = gridspec.GridSpec(3, 3, figure=fig, hspace=0.45, wspace=0.38)

BASELINE_DELTA  = 0.35
BASELINE_RSTOAT = 0.60

# ─ Row 1: 1D delta sweep ─────────────────────────────────────────────────────
ax1 = fig.add_subplot(gs[0, 0])
ax2 = fig.add_subplot(gs[0, 1])
ax3 = fig.add_subplot(gs[0, 2])

# x vs delta
ax1.plot(DELTA_RANGE, [v*100 for v in x_d], 'b-o', lw=2, ms=4)
ax1.axvline(BASELINE_DELTA, color='red', ls='--', lw=1.5, label=f'Baseline δ={BASELINE_DELTA}')
ax1.axvline(0.25, color='orange', ls=':', lw=1.5, label='δ=0.25 (h_crit→0.35)')
ax1.set_xlabel('δ (stoat natural mortality)', fontsize=10)
ax1.set_ylabel('Open foraging % at t=200', fontsize=10)
ax1.set_title('Open foraging vs δ\n(r_stoat=0.60 fixed, h=0.40)', fontsize=10)
ax1.legend(fontsize=8); ax1.grid(True, alpha=0.3)

# K vs delta
ax2.plot(DELTA_RANGE, K_d, 'g-o', lw=2, ms=4)
ax2.axhline(K0,   color='gray',  ls=':',  lw=1.2, alpha=0.7, label=f'K0={K0}')
ax2.axhline(K_MAX,color='green', ls=':',  lw=1.2, alpha=0.5, label=f'K_max={K_MAX}')
ax2.axvline(BASELINE_DELTA, color='red', ls='--', lw=1.5, label=f'Baseline δ={BASELINE_DELTA}')
# Shade region where h < h_crit (delta < 0.35 → h_crit > 0.25)
ax2.axvspan(DELTA_RANGE[0], BASELINE_DELTA, alpha=0.08, color='red',
            label='h < h_crit (stoats win)')
ax2.axvspan(BASELINE_DELTA, DELTA_RANGE[-1], alpha=0.08, color='green',
            label='h > h_crit (kiwi recover)')
ax2.set_xlabel('δ (stoat natural mortality)', fontsize=10)
ax2.set_ylabel('Kiwi population at t=200', fontsize=10)
ax2.set_title('Kiwi outcome vs δ\n(r_stoat=0.60 fixed, h=0.40)', fontsize=10)
ax2.legend(fontsize=7.5); ax2.grid(True, alpha=0.3)

# h_crit vs delta
ax3.plot(DELTA_RANGE, hcrit_d, 'purple', lw=2.5)
ax3.axhline(0.40, color='orange', ls='--', lw=2, label='h=0.40 (our fixed harvest)')
ax3.fill_between(DELTA_RANGE, hcrit_d, 0.25,
                 where=[hc > 0.25 for hc in hcrit_d],
                 alpha=0.15, color='red',   label='h insufficient (kiwi lost)')
ax3.fill_between(DELTA_RANGE, hcrit_d, 0.25,
                 where=[hc <= 0.25 for hc in hcrit_d],
                 alpha=0.15, color='green', label='h sufficient (kiwi recover)')
ax3.axvline(BASELINE_DELTA, color='red', ls='--', lw=1.5, label=f'Baseline δ={BASELINE_DELTA}')
ax3.set_xlabel('δ (stoat natural mortality)', fontsize=10)
ax3.set_ylabel('h_crit = r_stoat − δ', fontsize=10)
ax3.set_title('h_crit vs δ\n(crossover shows where h=0.40 becomes sufficient)', fontsize=10)
ax3.legend(fontsize=7.5); ax3.grid(True, alpha=0.3)

# ─ Row 2: 1D r_stoat sweep ───────────────────────────────────────────────────
ax4 = fig.add_subplot(gs[1, 0])
ax5 = fig.add_subplot(gs[1, 1])
ax6 = fig.add_subplot(gs[1, 2])

ax4.plot(RSTOAT_RANGE, [v*100 for v in x_r], 'b-o', lw=2, ms=4)
ax4.axvline(BASELINE_RSTOAT, color='red', ls='--', lw=1.5, label=f'Baseline r_stoat={BASELINE_RSTOAT}')
ax4.set_xlabel('r_stoat (stoat growth rate)', fontsize=10)
ax4.set_ylabel('Open foraging % at t=200', fontsize=10)
ax4.set_title('Open foraging vs r_stoat\n(δ=0.35 fixed, h=0.40)', fontsize=10)
ax4.legend(fontsize=8); ax4.grid(True, alpha=0.3)

ax5.plot(RSTOAT_RANGE, K_r, 'g-o', lw=2, ms=4)
ax5.axhline(K0,   color='gray',  ls=':', lw=1.2, alpha=0.7, label=f'K0={K0}')
ax5.axhline(K_MAX,color='green', ls=':', lw=1.2, alpha=0.5, label=f'K_max={K_MAX}')
ax5.axvline(BASELINE_RSTOAT, color='red', ls='--', lw=1.5, label=f'Baseline={BASELINE_RSTOAT}')
ax5.axvspan(RSTOAT_RANGE[0], BASELINE_RSTOAT, alpha=0.08, color='green',
            label='h > h_crit (kiwi recover)')
ax5.axvspan(BASELINE_RSTOAT, RSTOAT_RANGE[-1], alpha=0.08, color='red',
            label='h < h_crit (stoats win)')
ax5.set_xlabel('r_stoat (stoat growth rate)', fontsize=10)
ax5.set_ylabel('Kiwi population at t=200', fontsize=10)
ax5.set_title('Kiwi outcome vs r_stoat\n(δ=0.35 fixed, h=0.40)', fontsize=10)
ax5.legend(fontsize=7.5); ax5.grid(True, alpha=0.3)

ax6.plot(RSTOAT_RANGE, hcrit_r, 'purple', lw=2.5)
ax6.axhline(0.40, color='orange', ls='--', lw=2, label='h=0.40 (our fixed harvest)')
ax6.fill_between(RSTOAT_RANGE, hcrit_r, 0.25,
                 where=[hc > 0.25 for hc in hcrit_r],
                 alpha=0.15, color='red',   label='h insufficient')
ax6.fill_between(RSTOAT_RANGE, hcrit_r, 0.25,
                 where=[hc <= 0.25 for hc in hcrit_r],
                 alpha=0.15, color='green', label='h sufficient')
ax6.axvline(BASELINE_RSTOAT, color='red', ls='--', lw=1.5, label=f'Baseline={BASELINE_RSTOAT}')
ax6.set_xlabel('r_stoat (stoat growth rate)', fontsize=10)
ax6.set_ylabel('h_crit = r_stoat − δ', fontsize=10)
ax6.set_title('h_crit vs r_stoat\n(crossover shows where h=0.40 flips)', fontsize=10)
ax6.legend(fontsize=7.5); ax6.grid(True, alpha=0.3)

# ─ Row 3: 2D heatmaps ────────────────────────────────────────────────────────
ax7 = fig.add_subplot(gs[2, 0])
ax8 = fig.add_subplot(gs[2, 1])
ax9 = fig.add_subplot(gs[2, 2])

D_mesh, R_mesh = np.meshgrid(DELTA_RANGE, RSTOAT_RANGE, indexing='ij')

# Open foraging heatmap
im7 = ax7.contourf(D_mesh, R_mesh, X_grid, levels=20, cmap='RdYlGn')
ax7.contour(D_mesh, R_mesh, X_grid, levels=[50], colors='white', linewidths=1.5,
            linestyles='--')
# h_crit = 0.25 line: r_stoat - delta = 0.25 → r_stoat = delta + 0.25
r_hcrit_line = DELTA_RANGE + 0.40
mask = (r_hcrit_line >= RSTOAT_RANGE[0]) & (r_hcrit_line <= RSTOAT_RANGE[-1])
ax7.plot(DELTA_RANGE[mask], r_hcrit_line[mask], 'k-', lw=2.5, label='h_crit=0.40 line')
ax7.plot(BASELINE_DELTA, BASELINE_RSTOAT, 'w*', ms=14, label='Baseline', zorder=5)
plt.colorbar(im7, ax=ax7, label='Open foraging % at t=200')
ax7.set_xlabel('δ (stoat natural mortality)', fontsize=10)
ax7.set_ylabel('r_stoat (stoat growth rate)', fontsize=10)
ax7.set_title('Open foraging % — 2D\n(black line = h_crit=0.40 boundary)', fontsize=10)
ax7.legend(fontsize=8)

# Kiwi heatmap
im8 = ax8.contourf(D_mesh, R_mesh, K_grid, levels=20, cmap='RdYlGn')
ax8.contour(D_mesh, R_mesh, K_grid, levels=[K0], colors='white', linewidths=1.5,
            linestyles='--')
ax8.plot(DELTA_RANGE[mask], r_hcrit_line[mask], 'k-', lw=2.5, label='h_crit=0.40 line')
ax8.plot(BASELINE_DELTA, BASELINE_RSTOAT, 'w*', ms=14, label='Baseline', zorder=5)
plt.colorbar(im8, ax=ax8, label='Kiwi population at t=200')
ax8.set_xlabel('δ (stoat natural mortality)', fontsize=10)
ax8.set_ylabel('r_stoat (stoat growth rate)', fontsize=10)
ax8.set_title(f'Kiwi population — 2D\n(white dashed = K={K0} threshold)', fontsize=10)
ax8.legend(fontsize=8)

# Regime map: above/below h_crit
regime = np.where(R_mesh - D_mesh > 0.40, 1.0, 0.0)   # 1=h sufficient, 0=not
regime[R_mesh <= D_mesh] = np.nan
im9 = ax9.contourf(D_mesh, R_mesh, regime, levels=[-0.5, 0.5, 1.5],
                   colors=['#d62728', '#2ca02c'], alpha=0.6)
ax9.plot(DELTA_RANGE[mask], r_hcrit_line[mask], 'k-', lw=3, label='h_crit=0.40 boundary')
ax9.plot(BASELINE_DELTA, BASELINE_RSTOAT, 'w*', ms=16, zorder=5, label='Baseline')
from matplotlib.patches import Patch
ax9.legend(handles=[
    Patch(facecolor='#d62728', alpha=0.6, label='h=0.40 insufficient (stoats win)'),
    Patch(facecolor='#2ca02c', alpha=0.6, label='h=0.40 sufficient (kiwi recover)'),
    plt.Line2D([0],[0], color='k', lw=2.5, label='h_crit=0.40 boundary'),
    plt.Line2D([0],[0], marker='*', color='w', markerfacecolor='white',
               ms=12, lw=0, label='Baseline (δ=0.35, r_stoat=0.60)'),
], fontsize=8, loc='upper left')
ax9.set_xlabel('δ (stoat natural mortality)', fontsize=10)
ax9.set_ylabel('r_stoat (stoat growth rate)', fontsize=10)
ax9.set_title('Regime map: h=0.40 sufficient?\n(green=kiwi recover, red=kiwi lost)', fontsize=10)
ax9.grid(True, alpha=0.2)

fig.suptitle(
    'Sensitivity at h=0.40 (above critical threshold), intervention yr=2, t=200\n'
    'Rows: (1) vary δ  (2) vary r_stoat  (3) joint 2D space',
    fontsize=12, fontweight='bold'
)

plt.savefig('/mnt/user-data/outputs/sensitivity_h04.png', dpi=150, bbox_inches='tight')
print("\nPlot saved.")
