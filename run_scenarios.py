import numpy as np
from scipy.integrate import solve_ivp, odeint
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import warnings
warnings.filterwarnings('ignore')

# ── Core parameters ───────────────────────────────────────────────────────────
K0, S0, X0   = 20, 3, 0.3
R_STOAT      = 0.60
S_MAX        = 8
r_base       = 0.05
alpha        = 0.044
beta         = 0.0175
delta        = 0.35
K_MAX        = 150
h_handling   = 0.1
S_FLOOR      = S_MAX * (1 - delta / R_STOAT)
H_CRIT       = R_STOAT - delta
H_ERAD       = R_STOAT
BASE_PAYOFFS = np.array([[0.35, 0.55],[0.50, 0.22]])

# ── Dynamically compute PI_BAR_ESS ───────────────────────────────────────────
_a, _b = BASE_PAYOFFS[0]; _c, _d = BASE_PAYOFFS[1]
x_star = float(np.clip((_d-_b)/((_a-_b)+(_d-_c)), 0.0, 1.0))
def avg_payoff_base(x):
    pA = x*BASE_PAYOFFS[0,0]+(1-x)*BASE_PAYOFFS[0,1]
    pB = x*BASE_PAYOFFS[1,0]+(1-x)*BASE_PAYOFFS[1,1]
    return x*pA+(1-x)*pB
PI_BAR_ESS = avg_payoff_base(x_star)

print("="*70)
print("KIWI CONSERVATION — CEPPM SCENARIO ANALYSIS")
print("Comparing: Standalone EGT | Standalone L-V | CEPPM (Hybrid)")
print("="*70)
print(f"  S_floor={S_FLOOR:.3f}  h_crit={H_CRIT:.2f}  h_erad={H_ERAD:.2f}")
print(f"  ESS x* = {x_star:.4f} ({x_star*100:.2f}%)  PI_BAR_ESS = {PI_BAR_ESS:.6f}")
print(f"  Effective kiwi recovery threshold ≈ h=0.155–0.160")

# ============================================================================
# PART 1: STANDALONE EGT MODEL
# Replicator dynamics with fixed base payoffs — no population feedback.
# Strategy evolves based on relative payoffs alone; stoat density is treated
# as a fixed external parameter (not updated by population dynamics).
# Intervention modelled as a payoff shift: open foraging becomes more
# rewarding when harvest reduces predation pressure.
# ============================================================================

def egt_payoff_A(x, pm=BASE_PAYOFFS):
    return x*pm[0,0] + (1-x)*pm[0,1]

def egt_avg_payoff(x, pm=BASE_PAYOFFS):
    pA = x*pm[0,0] + (1-x)*pm[0,1]
    pB = x*pm[1,0] + (1-x)*pm[1,1]
    return x*pA + (1-x)*pB

def egt_replicator(x, t, pm=BASE_PAYOFFS):
    if x <= 0 or x >= 1:
        return 0
    return x * (egt_payoff_A(x, pm) - egt_avg_payoff(x, pm))

# Intervention payoffs: reduced predation penalty when stoats are suppressed.
# Represents the behavioural shift that EGT alone would predict if told
# predation pressure has been removed — but without population dynamics.
INTERVENTION_PAYOFFS = BASE_PAYOFFS + np.array([[0.117, 0.275],[0.062, 0.02]])

def simulate_egt(it, hr, t_max=200):
    """
    Standalone EGT: replicator dynamics only.
    Pre-intervention: base payoffs (predation pressure present).
    Post-intervention: shifted payoffs (reduced predation pressure).
    Note: no population dynamics — strategy evolves in isolation.
    """
    t_span = np.linspace(0, t_max, 2000)
    t_pre  = t_span[t_span <= it]
    t_post = t_span[t_span >  it]

    sol_pre = odeint(egt_replicator, X0, t_pre,
                     args=(BASE_PAYOFFS,), tfirst=False)
    x_at_it = float(sol_pre[-1][0]) if len(sol_pre) > 0 else X0

    if hr > 0 and len(t_post) > 0:
        # Payoff shift scaled by harvest rate — larger h = bigger payoff improvement
        pm_post = BASE_PAYOFFS + np.array([[0.117, 0.275],[0.062, 0.02]]) * (hr / 0.4)
        sol_post = odeint(egt_replicator, x_at_it, t_post - it,
                          args=(pm_post,), tfirst=False)
        t_all = np.concatenate([t_pre, t_post])
        x_all = np.concatenate([sol_pre.flatten(), sol_post.flatten()])
    else:
        t_all = t_pre
        x_all = sol_pre.flatten()

    return t_all, x_all

# ============================================================================
# PART 2: STANDALONE L-V MODEL
# Extended Lotka-Volterra with strategy FIXED at X0 throughout.
# Population dynamics respond to harvest but there is no behavioural
# adaptation — open foraging proportion stays at X0=0.30 regardless.
# This isolates the demographic effect of predator control.
# ============================================================================

def lv_system(t, y, hr=0, x_fixed=X0):
    """
    Standalone L-V with fixed strategy x_fixed.
    No EGT coupling — strategy does not evolve.
    Effective parameters computed once from fixed x.
    """
    K, S = y; K=max(K,0); S=max(S,0)
    r_eff   = r_base * (avg_payoff_base(x_fixed) / PI_BAR_ESS)
    a_eff   = alpha * (x_fixed*1.3 + (1-x_fixed)*0.7)
    f_K     = (a_eff*K) / (1 + a_eff*h_handling*K)
    dKdt    = r_eff*K*(1-K/K_MAX) - f_K*S
    nat     = R_STOAT*S*(1-S/S_MAX) + beta*f_K*S - delta*S
    nkg     = R_STOAT*S*(1-S/S_MAX) - delta*S
    if S<=S_FLOOR and nkg<0: nat=max(nat,0.0)
    return [dKdt, nat-hr*S]

def simulate_lv(it, hr, t_max=200, x_fixed=X0):
    """
    Standalone L-V: population dynamics only, strategy fixed at x_fixed.
    Returns time, K, S. Note x is constant = x_fixed throughout.
    """
    y0=[K0, S0]
    t1=np.linspace(0, it, max(400, it*20))
    t2=np.linspace(it, t_max, max(800, (t_max-it)*20))
    s1=solve_ivp(lambda t,y: lv_system(t,y,0,x_fixed), [0,it], y0,
                 t_eval=t1, method='RK45', rtol=1e-8, atol=1e-10)
    s2=solve_ivp(lambda t,y: lv_system(t,y,hr,x_fixed), [it,t_max],
                 [s1.y[0][-1], s1.y[1][-1]],
                 t_eval=t2, method='RK45', rtol=1e-8, atol=1e-10)
    t_all=np.concatenate([s1.t,    s2.t])
    K_all=np.concatenate([s1.y[0], s2.y[0]])
    S_all=np.concatenate([s1.y[1], s2.y[1]])
    x_all=np.full_like(t_all, x_fixed)   # strategy is constant
    return t_all, x_all, K_all, S_all

# ============================================================================
# PART 3: CEPPM (HYBRID) MODEL
# Fully coupled EGT + L-V. Strategy evolves in response to stoat density
# via dynamic payoffs; population dynamics respond to current strategy.
# ============================================================================

def dynamic_payoffs(S):
    sig = 1/(1+np.exp(-3.0*(S-1.0)))
    return np.array([
        [max(BASE_PAYOFFS[0,0]-0.40*sig,0.01), max(BASE_PAYOFFS[0,1]-0.40*sig,0.01)],
        [max(BASE_PAYOFFS[1,0]-0.15*sig,0.01), max(BASE_PAYOFFS[1,1]-0.15*sig,0.01)]
    ])

def hybrid_system(t, y, hr=0):
    x,K,S = y; x=np.clip(x,0.01,0.99); K=max(K,0); S=max(S,0)
    dp  = dynamic_payoffs(S)
    pA  = x*dp[0,0]+(1-x)*dp[0,1]; pB=x*dp[1,0]+(1-x)*dp[1,1]
    pav = x*pA+(1-x)*pB
    dxdt = x*(pA-pav) if 0<x<1 else 0
    r_eff  = r_base*(avg_payoff_base(x)/PI_BAR_ESS)
    a_eff  = alpha*(x*1.3+(1-x)*0.7)
    f_K    = (a_eff*K)/(1+a_eff*h_handling*K)
    dKdt   = r_eff*K*(1-K/K_MAX)-f_K*S
    nat    = R_STOAT*S*(1-S/S_MAX)+beta*f_K*S-delta*S
    nkg    = R_STOAT*S*(1-S/S_MAX)-delta*S
    if S<=S_FLOOR and nkg<0: nat=max(nat,0.0)
    return [dxdt, dKdt, nat-hr*S]

def simulate_ceppm(it, hr, t_max=200):
    y0=[X0,K0,S0]
    t1=np.linspace(0,it,max(400,it*20))
    t2=np.linspace(it,t_max,max(800,(t_max-it)*20))
    s1=solve_ivp(lambda t,y:hybrid_system(t,y,0),[0,it],y0,
                 t_eval=t1,method='RK45',rtol=1e-8,atol=1e-10)
    s2=solve_ivp(lambda t,y:hybrid_system(t,y,hr),[it,t_max],
                 [s1.y[0][-1],s1.y[1][-1],s1.y[2][-1]],
                 t_eval=t2,method='RK45',rtol=1e-8,atol=1e-10)
    return (np.concatenate([s1.t,s2.t]),np.concatenate([s1.y[0],s2.y[0]]),
            np.concatenate([s1.y[1],s2.y[1]]),np.concatenate([s1.y[2],s2.y[2]]))

# ============================================================================
# SCENARIOS
# ============================================================================

scenarios = [
    ('Control',     'No intervention (h=0)',              199, 0.00),
    ('Scenario 1a', 'Intervene yr 20 | h=0.10 — extinct', 20,  0.10),
    ('Scenario 1b', 'Intervene yr 20 | h=0.20',           20,  0.20),
    ('Scenario 2',  'Intervene yr 10 | h=0.40',           10,  0.40),
    ('Scenario 3',  'Intervene yr 2  | h=0.60',            2,  0.60),
]
SNAPSHOTS = [0, 10, 25, 50, 100, 200]

print("\nRunning all three models for each scenario...")
results_egt   = {}
results_lv    = {}
results_ceppm = {}

for name, label, it, hr in scenarios:
    # EGT
    t_e, x_e = simulate_egt(it, hr)
    results_egt[name] = {'t':t_e,'x':x_e,'label':label}
    # L-V
    t_l, x_l, K_l, S_l = simulate_lv(it, hr)
    results_lv[name]  = {'t':t_l,'x':x_l,'K':K_l,'S':S_l,'label':label}
    # CEPPM
    t_c, x_c, K_c, S_c = simulate_ceppm(it, hr)
    results_ceppm[name] = {'t':t_c,'x':x_c,'K':K_c,'S':S_c,'label':label}
    print(f"  {label}")
    print(f"    EGT:   x_final={x_e[-1]*100:.1f}%")
    print(f"    L-V:   K_final={K_l[-1]:.1f}  S_final={S_l[-1]:.3f}  x=fixed {X0*100:.0f}%")
    print(f"    CEPPM: K_final={K_c[-1]:.1f}  S_final={S_c[-1]:.3f}  x_final={x_c[-1]*100:.1f}%")

# ============================================================================
# COMPARISON TABLES
# ============================================================================

def interp(at, av, tq): return float(np.interp(tq, at, av))

W = 90
print("\n" + "="*W)
print("THREE-MODEL COMPARISON TABLE")
print(f"S_floor={S_FLOOR:.3f}  h_crit={H_CRIT:.2f}  PI_BAR_ESS={PI_BAR_ESS:.4f}")
print("="*W)

for name, label, it, hr in scenarios:
    print(f"\n{'─'*W}")
    print(f"SCENARIO: {label}")
    print(f"{'─'*W}")
    print(f"{'t':>5} | {'EGT x%':>10} | {'LV x% (fixed)':>14} {'LV K':>8} {'LV S':>8} "
          f"| {'CEPPM x%':>10} {'CEPPM K':>9} {'CEPPM S':>9}")
    print(f"{'─'*W}")
    re = results_egt[name]
    rl = results_lv[name]
    rc = results_ceppm[name]
    for tq in SNAPSHOTS:
        xe = interp(re['t'],re['x'],tq)
        xl = interp(rl['t'],rl['x'],tq)
        Kl = interp(rl['t'],rl['K'],tq)
        Sl = interp(rl['t'],rl['S'],tq)
        xc = interp(rc['t'],rc['x'],tq)
        Kc = interp(rc['t'],rc['K'],tq)
        Sc = interp(rc['t'],rc['S'],tq)
        print(f"{tq:>5} | {xe*100:>9.1f}% | {xl*100:>13.1f}% {Kl:>8.1f} {Sl:>8.3f} "
              f"| {xc*100:>9.1f}% {Kc:>9.1f} {Sc:>9.3f}")

print(f"\n{'─'*W}")
print("Note: EGT x% = strategy from replicator dynamics with fixed payoffs (no pop. feedback)")
print("      LV x%  = strategy fixed at X0=30% throughout (no behavioural adaptation)")
print("      CEPPM  = fully coupled; strategy and population evolve together")

# ============================================================================
# PLOTS
# ============================================================================

colors  = {'Control':'#d62728','Scenario 1a':'#8c0000',
           'Scenario 1b':'#ff7f0e','Scenario 2':'#2ca02c','Scenario 3':'#1f77b4'}
lstyles = {'Control':'-','Scenario 1a':'-.','Scenario 1b':'--',
           'Scenario 2':'--','Scenario 3':'--'}

# ── Figure 1: CEPPM results (main) ──────────────────────────────────────────
fig=plt.figure(figsize=(17,13))
gs=gridspec.GridSpec(2,3,figure=fig,hspace=0.40,wspace=0.35)
ax_x=fig.add_subplot(gs[0,0]); ax_K=fig.add_subplot(gs[0,1])
ax_S=fig.add_subplot(gs[0,2]); ax_p=fig.add_subplot(gs[1,:])

for name,label,it,hr in scenarios:
    rc=results_ceppm[name]; col=colors[name]; ls=lstyles[name]
    ax_x.plot(rc['t'],rc['x']*100,color=col,lw=2,ls=ls,label=label)
    ax_K.plot(rc['t'],rc['K'],    color=col,lw=2,ls=ls,label=label)
    ax_S.plot(rc['t'],rc['S'],    color=col,lw=2,ls=ls,label=label)
    ax_p.plot(rc['K'],rc['S'],    color=col,lw=2,ls=ls,label=label,alpha=0.85)
    ax_p.plot(rc['K'][0], rc['S'][0], 'o',color=col,ms=7)
    ax_p.plot(rc['K'][-1],rc['S'][-1],'s',color=col,ms=8)

for ax in [ax_x,ax_K,ax_S]:
    ax.set_xlim(0,200); ax.grid(True,alpha=0.3)
    ax.legend(fontsize=7); ax.set_xlabel('Time (years)',fontsize=10)

ax_x.axhline(x_star*100,color='gray',ls=':',lw=1.2,alpha=0.6,label=f'ESS x*={x_star*100:.1f}%')
ax_x.axhline(66.28,color='purple',ls=':',lw=1.2,alpha=0.6,label='Simulated ESS 66.28%')
ax_x.set_ylabel('Open foraging (%)',fontsize=10); ax_x.set_title('CEPPM — Foraging strategy (x)',fontsize=11)
ax_x.set_ylim(0,100); ax_x.legend(fontsize=7)
ax_K.axhline(K_MAX,color='green',ls=':',lw=1.2,alpha=0.5); ax_K.axhline(K0,color='gray',ls=':',lw=1.2,alpha=0.5)
ax_K.axhspan(0,5,alpha=0.06,color='red',label='Extinction zone')
ax_K.set_ylabel('Kiwi population',fontsize=10); ax_K.set_title('CEPPM — Kiwi population (K)',fontsize=11); ax_K.legend(fontsize=7)
ax_S.axhline(S_FLOOR,color='darkred',ls='--',lw=1.5,alpha=0.7,label=f'S_floor={S_FLOOR:.2f}')
ax_S.set_ylabel('Stoat population',fontsize=10); ax_S.set_title('CEPPM — Stoat population (S)',fontsize=11); ax_S.legend(fontsize=7)
ax_p.axhline(S_FLOOR,color='darkred',ls='--',lw=1.5,alpha=0.6,label=f'S_floor={S_FLOOR:.2f}')
ax_p.set_xlabel('Kiwi population (K)',fontsize=11); ax_p.set_ylabel('Stoat population (S)',fontsize=11)
ax_p.set_title('CEPPM — Phase portrait  (circle=start, square=end t=200)',fontsize=11)
ax_p.legend(fontsize=9); ax_p.grid(True,alpha=0.3)
fig.suptitle('CEPPM (Hybrid) — 5 scenarios\n'
             f'S_floor={S_FLOOR:.3f}  h_crit={H_CRIT:.2f}  PI_BAR_ESS={PI_BAR_ESS:.4f}  '
             f'Effective recovery threshold h≈0.155–0.160',fontsize=11,fontweight='bold')
plt.savefig('/mnt/user-data/outputs/final_scenario_analysis.png',dpi=150,bbox_inches='tight')
plt.close()
print("\nSaved: final_scenario_analysis.png")

# ── Figure 2: Three-model comparison per scenario ───────────────────────────
for name,label,it,hr in scenarios:
    fig,axes=plt.subplots(1,3,figsize=(16,5))
    col=colors[name]
    re=results_egt[name]; rl=results_lv[name]; rc=results_ceppm[name]

    # Panel 1: Strategy (x)
    ax=axes[0]
    ax.plot(re['t'],re['x']*100,color=col,lw=2,ls='--',label='EGT only (fixed payoffs)')
    ax.plot(rl['t'],rl['x']*100,color='gray',lw=2,ls=':',label=f'L-V only (x fixed={X0*100:.0f}%)')
    ax.plot(rc['t'],rc['x']*100,color=col,lw=2.5,ls='-',label='CEPPM (coupled)')
    ax.axhline(x_star*100,color='black',ls=':',lw=1,alpha=0.4,label=f'ESS {x_star*100:.1f}%')
    ax.axhline(66.28,color='purple',ls=':',lw=1,alpha=0.4,label='Simulated ESS 66.28%')
    if it<200: ax.axvline(it,color='green',ls='--',lw=1.5,alpha=0.6,label=f'Intervention yr {it}')
    ax.set_xlim(0,200); ax.set_ylim(0,100)
    ax.set_xlabel('Time (years)',fontsize=11); ax.set_ylabel('Open foraging (%)',fontsize=11)
    ax.set_title('Foraging strategy (x)',fontsize=12); ax.legend(fontsize=8); ax.grid(True,alpha=0.3)

    # Panel 2: Kiwi (K)
    ax=axes[1]
    ax.plot(rl['t'],rl['K'],color='gray',lw=2,ls=':',label='L-V only (no behavioural adapt.)')
    ax.plot(rc['t'],rc['K'],color=col,lw=2.5,ls='-',label='CEPPM (coupled)')
    ax.axhline(K_MAX,color='green',ls=':',lw=1.2,alpha=0.5,label=f'K_max={K_MAX}')
    ax.axhline(K0,color='gray',ls=':',lw=1.2,alpha=0.5,label=f'K0={K0}')
    ax.axhspan(0,5,alpha=0.06,color='red',label='Extinction zone')
    if it<200: ax.axvline(it,color='green',ls='--',lw=1.5,alpha=0.6)
    ax.set_xlim(0,200)
    ax.set_xlabel('Time (years)',fontsize=11); ax.set_ylabel('Kiwi population',fontsize=11)
    ax.set_title('Kiwi population (K)',fontsize=12); ax.legend(fontsize=8); ax.grid(True,alpha=0.3)

    # Panel 3: Stoats (S)
    ax=axes[2]
    ax.plot(rl['t'],rl['S'],color='gray',lw=2,ls=':',label='L-V only')
    ax.plot(rc['t'],rc['S'],color=col,lw=2.5,ls='-',label='CEPPM (coupled)')
    ax.axhline(S_FLOOR,color='darkred',ls='--',lw=1.5,alpha=0.7,label=f'S_floor={S_FLOOR:.2f}')
    if it<200: ax.axvline(it,color='green',ls='--',lw=1.5,alpha=0.6,label=f'Intervention yr {it}')
    ax.set_xlim(0,200)
    ax.set_xlabel('Time (years)',fontsize=11); ax.set_ylabel('Stoat population',fontsize=11)
    ax.set_title('Stoat population (S)',fontsize=12); ax.legend(fontsize=8); ax.grid(True,alpha=0.3)

    slug=name.lower().replace(' ','_')
    fig.suptitle(f'Three-model comparison: {label}\n'
                 f'EGT only vs L-V only vs CEPPM (coupled)',fontsize=12,fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'/mnt/user-data/outputs/comparison_{slug}.png',dpi=150,bbox_inches='tight')
    plt.close()
    print(f"Saved: comparison_{slug}.png")

print("\nAll outputs saved.")
