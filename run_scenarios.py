import numpy as np
from scipy.integrate import solve_ivp
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
print("="*70)
print(f"  S_floor={S_FLOOR:.3f}  h_crit={H_CRIT:.2f}  h_erad={H_ERAD:.2f}")
print(f"  ESS x* = {x_star:.4f} ({x_star*100:.2f}%)  PI_BAR_ESS = {PI_BAR_ESS:.6f}")
print(f"  Effective kiwi recovery threshold ≈ h=0.155–0.160")

# ── EGT / ODE helpers ────────────────────────────────────────────────────────
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

def simulate(it, hr, t_max=200):
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

# ── Five scenarios ────────────────────────────────────────────────────────────
scenarios = [
    ('Control',     'No intervention (h=0)',              199, 0.00),
    ('Scenario 1a', 'Intervene yr 20 | h=0.10 — extinct', 20,  0.10),
    ('Scenario 1b', 'Intervene yr 20 | h=0.20',           20,  0.20),
    ('Scenario 2',  'Intervene yr 10 | h=0.40',           10,  0.40),
    ('Scenario 3',  'Intervene yr 2  | h=0.60',            2,  0.60),
]
SNAPSHOTS = [0, 10, 25, 50, 100, 200]

print("\nRunning scenarios...")
results={}
for name,label,it,hr in scenarios:
    t,x,K,S=simulate(it,hr)
    results[name]={'t':t,'x':x,'K':K,'S':S,'it':it,'hr':hr,'label':label}
    print(f"  {label}: K={K[-1]:.1f}  S={S[-1]:.3f}  x={x[-1]*100:.1f}%")

# ── Table ─────────────────────────────────────────────────────────────────────
def interp(at,av,tq): return float(np.interp(tq,at,av))
W=78
print("\n"+"="*W)
print("SCENARIO COMPARISON TABLE")
print(f"S_floor={S_FLOOR:.3f}  h_crit={H_CRIT:.2f}  h_erad={H_ERAD:.2f}  PI_BAR_ESS={PI_BAR_ESS:.4f}")
print("="*W)
print(f"{'Scenario':<34} {'t':>5} {'x (open%)':>10} {'K (kiwi)':>10} {'S (stoats)':>10}")
print("-"*W)
for name,label,it,hr in scenarios:
    r=results[name]; first=True
    for tq in SNAPSHOTS:
        xv=interp(r['t'],r['x'],tq); Kv=interp(r['t'],r['K'],tq); Sv=interp(r['t'],r['S'],tq)
        lbl=label if first else ""; first=False
        print(f"{lbl:<34} {tq:>5}  {xv*100:>8.1f}%  {Kv:>10.1f}  {Sv:>10.3f}")
    print("-"*W)

# ── Plots ─────────────────────────────────────────────────────────────────────
colors ={'Control':'#d62728','Scenario 1a':'#8c0000',
         'Scenario 1b':'#ff7f0e','Scenario 2':'#2ca02c','Scenario 3':'#1f77b4'}
lstyles={'Control':'-','Scenario 1a':'-.','Scenario 1b':'--','Scenario 2':'--','Scenario 3':'--'}

fig=plt.figure(figsize=(17,13))
gs=gridspec.GridSpec(2,3,figure=fig,hspace=0.40,wspace=0.35)
ax_x=fig.add_subplot(gs[0,0]); ax_K=fig.add_subplot(gs[0,1])
ax_S=fig.add_subplot(gs[0,2]); ax_p=fig.add_subplot(gs[1,:])

for name,label,it,hr in scenarios:
    r=results[name]; col=colors[name]; ls=lstyles[name]
    ax_x.plot(r['t'],r['x']*100,color=col,lw=2,ls=ls,label=label)
    ax_K.plot(r['t'],r['K'],    color=col,lw=2,ls=ls,label=label)
    ax_S.plot(r['t'],r['S'],    color=col,lw=2,ls=ls,label=label)
    ax_p.plot(r['K'],r['S'],    color=col,lw=2,ls=ls,label=label,alpha=0.85)
    ax_p.plot(r['K'][0], r['S'][0], 'o',color=col,ms=7)
    ax_p.plot(r['K'][-1],r['S'][-1],'s',color=col,ms=8)

for ax in [ax_x,ax_K,ax_S]:
    ax.set_xlim(0,200); ax.grid(True,alpha=0.3)
    ax.legend(fontsize=7); ax.set_xlabel('Time (years)',fontsize=10)

ax_x.axhline(x_star*100,color='gray',ls=':',lw=1.2,alpha=0.6,label=f'Analytical ESS {x_star*100:.1f}%')
ax_x.axhline(66.28,color='purple',ls=':',lw=1.2,alpha=0.6,label='Simulated ESS 66.28%')
ax_x.set_ylabel('Open foraging (%)',fontsize=10); ax_x.set_title('Foraging strategy (x)',fontsize=11)
ax_x.set_ylim(0,100); ax_x.legend(fontsize=7)

ax_K.axhline(K_MAX,color='green',ls=':',lw=1.2,alpha=0.5,label=f'K_max={K_MAX}')
ax_K.axhline(K0,  color='gray', ls=':',lw=1.2,alpha=0.5,label=f'K0={K0}')
ax_K.axhspan(0,5,alpha=0.06,color='red',label='Extinction zone')
ax_K.set_ylabel('Kiwi population',fontsize=10); ax_K.set_title('Kiwi population (K)',fontsize=11)
ax_K.legend(fontsize=7)

ax_S.axhline(S_FLOOR,color='darkred',ls='--',lw=1.5,alpha=0.7,label=f'S_floor={S_FLOOR:.2f}')
ax_S.set_ylabel('Stoat population',fontsize=10); ax_S.set_title('Stoat population (S)',fontsize=11)
ax_S.legend(fontsize=7)

ax_p.axhline(S_FLOOR,color='darkred',ls='--',lw=1.5,alpha=0.6,label=f'S_floor={S_FLOOR:.2f}')
ax_p.set_xlabel('Kiwi population (K)',fontsize=11); ax_p.set_ylabel('Stoat population (S)',fontsize=11)
ax_p.set_title('Phase portrait — all scenarios  (circle=start, square=end t=200)',fontsize=11)
ax_p.legend(fontsize=9); ax_p.grid(True,alpha=0.3)

fig.suptitle(
    'CEPPM Scenario Analysis — 5 scenarios\n'
    f'S_floor={S_FLOOR:.3f}  h_crit={H_CRIT:.2f}  h_erad={H_ERAD:.2f}  '
    f'PI_BAR_ESS={PI_BAR_ESS:.4f}  Effective recovery threshold h≈0.155–0.160',
    fontsize=11,fontweight='bold')

plt.savefig('/mnt/user-data/outputs/final_scenario_analysis.png',dpi=150,bbox_inches='tight')
print("\nPlot saved: final_scenario_analysis.png")
