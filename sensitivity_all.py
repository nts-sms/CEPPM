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
BASE_PAYOFFS = np.array([[0.35, 0.55],[0.50, 0.22]])

# Dynamic PI_BAR_ESS
a, b = BASE_PAYOFFS[0]; c, d = BASE_PAYOFFS[1]
x_star = float(np.clip((d-b)/((a-b)+(d-c)), 0, 1))
def avg_payoff_base(x):
    pA = x*BASE_PAYOFFS[0,0]+(1-x)*BASE_PAYOFFS[0,1]
    pB = x*BASE_PAYOFFS[1,0]+(1-x)*BASE_PAYOFFS[1,1]
    return x*pA+(1-x)*pB
PI_BAR_ESS = avg_payoff_base(x_star)

DELTA_RANGE  = np.linspace(0.15, 0.55, 25)
RSTOAT_RANGE = np.linspace(0.30, 0.90, 25)
INTERVENTION_YR = 2
T_MAX = 200

# ── ODE ───────────────────────────────────────────────────────────────────────
def make_system(delta_p, r_stoat_p):
    S_floor_p = 8*(1 - delta_p/r_stoat_p) if r_stoat_p > delta_p else 0.0
    def system(t, y, hr=0):
        x,K,S = y; x=np.clip(x,0.01,0.99); K=max(K,0); S=max(S,0)
        sig = 1/(1+np.exp(-3.0*(S-1.0)))
        dp = np.array([[max(BASE_PAYOFFS[0,0]-0.40*sig,0.01),max(BASE_PAYOFFS[0,1]-0.40*sig,0.01)],
                       [max(BASE_PAYOFFS[1,0]-0.15*sig,0.01),max(BASE_PAYOFFS[1,1]-0.15*sig,0.01)]])
        pA=x*dp[0,0]+(1-x)*dp[0,1]; pB=x*dp[1,0]+(1-x)*dp[1,1]
        pav=x*pA+(1-x)*pB; dxdt=x*(pA-pav) if 0<x<1 else 0
        r_eff=r_base*(avg_payoff_base(x)/PI_BAR_ESS)
        a_eff=alpha*(x*1.3+(1-x)*0.7)
        f_K=(a_eff*K)/(1+a_eff*h_handling*K)
        dKdt=r_eff*K*(1-K/K_MAX)-f_K*S
        nat=r_stoat_p*S*(1-S/8)+beta*f_K*S-delta_p*S
        nkg=r_stoat_p*S*(1-S/8)-delta_p*S
        if S<=S_floor_p and nkg<0: nat=max(nat,0.0)
        return [dxdt,dKdt,nat-hr*S]
    return system

def simulate_sens(delta_p, r_stoat_p, hr, it=INTERVENTION_YR, t_max=T_MAX):
    sys = make_system(delta_p, r_stoat_p)
    y0 = [X0,K0,S0]
    t1 = np.linspace(0,it,max(200,it*20))
    t2 = np.linspace(it,t_max,max(800,(t_max-it)*10))
    try:
        s1=solve_ivp(lambda t,y:sys(t,y,0),[0,it],y0,t_eval=t1,method='RK45',rtol=1e-8,atol=1e-10)
        s2=solve_ivp(lambda t,y:sys(t,y,hr),[it,t_max],[s1.y[0][-1],s1.y[1][-1],s1.y[2][-1]],
                     t_eval=t2,method='RK45',rtol=1e-8,atol=1e-10)
        return float(s2.y[0][-1]), float(s2.y[1][-1]), float(s2.y[2][-1])
    except:
        return np.nan, np.nan, np.nan

# ── Run all three harvest rates ───────────────────────────────────────────────
harvest_runs = [
    (0.15, 'h=0.15 (failed management zone)'),
    (0.25, 'h=0.25 (= h_crit)'),
    (0.40, 'h=0.40 (above h_crit)'),
]

all_results = {}
for hr_fixed, hr_label in harvest_runs:
    print(f"\nRunning sensitivity: {hr_label}")
    x_d=[]; K_d=[]; hcrit_d=[]
    for dv in DELTA_RANGE:
        xf,Kf,_=simulate_sens(dv, R_STOAT, hr_fixed)
        x_d.append(xf); K_d.append(Kf); hcrit_d.append(R_STOAT-dv)

    x_r=[]; K_r=[]; hcrit_r=[]
    for rs in RSTOAT_RANGE:
        xf,Kf,_=simulate_sens(delta, rs, hr_fixed)
        x_r.append(xf); K_r.append(Kf); hcrit_r.append(rs-delta)

    X_grid=np.full((len(DELTA_RANGE),len(RSTOAT_RANGE)),np.nan)
    K_grid=np.full_like(X_grid,np.nan)
    for i,dv in enumerate(DELTA_RANGE):
        for j,rs in enumerate(RSTOAT_RANGE):
            if rs<=dv: continue
            xf,Kf,_=simulate_sens(dv,rs,hr_fixed)
            X_grid[i,j]=xf*100; K_grid[i,j]=Kf

    all_results[hr_label]={'x_d':x_d,'K_d':K_d,'hcrit_d':hcrit_d,
                            'x_r':x_r,'K_r':K_r,'hcrit_r':hcrit_r,
                            'X_grid':X_grid,'K_grid':K_grid,'hr':hr_fixed}
    print(f"  Done.")

# ── Print tables ──────────────────────────────────────────────────────────────
for hr_label, res in all_results.items():
    hr_fixed = res['hr']
    print(f"\n{'='*72}")
    print(f"1D SENSITIVITY — delta sweep  ({hr_label}, intervention yr={INTERVENTION_YR}, t={T_MAX})")
    print(f"Baseline: delta={delta}, r_stoat={R_STOAT}  →  h_crit={H_CRIT:.2f}")
    print(f"{'='*72}")
    print(f"{'delta':>8} {'h_crit':>8} {'h vs hcrit':>12} {'x% (open)':>12} {'K (kiwi)':>10}")
    print("-"*55)
    for dv,hc,xf,Kf in zip(DELTA_RANGE,res['hcrit_d'],res['x_d'],res['K_d']):
        rel="= h_crit" if abs(hc-hr_fixed)<0.01 else ("> h_crit" if hc<hr_fixed else "< h_crit")
        print(f"  {dv:6.3f}   {hc:6.3f}   {rel:>12}   {xf*100 if not np.isnan(xf) else float('nan'):8.1f}%   {Kf:8.1f}")

    print(f"\n{'='*72}")
    print(f"1D SENSITIVITY — r_stoat sweep  ({hr_label}, intervention yr={INTERVENTION_YR}, t={T_MAX})")
    print(f"{'='*72}")
    print(f"{'r_stoat':>8} {'h_crit':>8} {'h vs hcrit':>12} {'x% (open)':>12} {'K (kiwi)':>10}")
    print("-"*55)
    for rs,hc,xf,Kf in zip(RSTOAT_RANGE,res['hcrit_r'],res['x_r'],res['K_r']):
        rel="= h_crit" if abs(hc-hr_fixed)<0.01 else ("> h_crit" if hc<hr_fixed else "< h_crit")
        print(f"  {rs:6.3f}   {hc:6.3f}   {rel:>12}   {xf*100 if not np.isnan(xf) else float('nan'):8.1f}%   {Kf:8.1f}")

# ── Plots — one figure per harvest rate ───────────────────────────────────────
for hr_label, res in all_results.items():
    hr_fixed = res['hr']
    fig=plt.figure(figsize=(18,16))
    gs=gridspec.GridSpec(3,3,figure=fig,hspace=0.45,wspace=0.38)

    # Row 1: delta sweep
    ax1=fig.add_subplot(gs[0,0]); ax2=fig.add_subplot(gs[0,1]); ax3=fig.add_subplot(gs[0,2])
    ax1.plot(DELTA_RANGE,[v*100 if not np.isnan(v) else np.nan for v in res['x_d']],'b-o',lw=2,ms=4)
    ax1.axvline(delta,color='red',ls='--',lw=1.5,label=f'Baseline δ={delta}')
    ax1.set_xlabel('δ (stoat natural mortality)',fontsize=10); ax1.set_ylabel('Open foraging % at t=200',fontsize=10)
    ax1.set_title(f'Open foraging vs δ\n(r_stoat={R_STOAT} fixed, {hr_label})',fontsize=10)
    ax1.legend(fontsize=8); ax1.grid(True,alpha=0.3)

    ax2.plot(DELTA_RANGE,[v if not np.isnan(v) else np.nan for v in res['K_d']],'g-o',lw=2,ms=4)
    ax2.axhline(K0,color='gray',ls=':',lw=1.2,alpha=0.7,label=f'K0={K0}')
    ax2.axhline(K_MAX,color='green',ls=':',lw=1.2,alpha=0.5,label=f'K_max={K_MAX}')
    ax2.axvline(delta,color='red',ls='--',lw=1.5,label=f'Baseline δ={delta}')
    bdelta=[dv for dv,hc in zip(DELTA_RANGE,res['hcrit_d']) if hc>hr_fixed]
    if bdelta: ax2.axvspan(min(DELTA_RANGE),max(bdelta),alpha=0.08,color='red',label='h < h_crit')
    gdelta=[dv for dv,hc in zip(DELTA_RANGE,res['hcrit_d']) if hc<=hr_fixed]
    if gdelta: ax2.axvspan(min(gdelta),max(DELTA_RANGE),alpha=0.08,color='green',label='h ≥ h_crit')
    ax2.set_xlabel('δ (stoat natural mortality)',fontsize=10); ax2.set_ylabel('Kiwi at t=200',fontsize=10)
    ax2.set_title(f'Kiwi outcome vs δ\n(r_stoat={R_STOAT} fixed, {hr_label})',fontsize=10)
    ax2.legend(fontsize=7.5); ax2.grid(True,alpha=0.3)

    ax3.plot(DELTA_RANGE,res['hcrit_d'],'purple',lw=2.5)
    ax3.axhline(hr_fixed,color='orange',ls='--',lw=2,label=f'h={hr_fixed:.2f} (fixed harvest)')
    ax3.fill_between(DELTA_RANGE,res['hcrit_d'],hr_fixed,
                     where=[hc>hr_fixed for hc in res['hcrit_d']],alpha=0.15,color='red',label='h insufficient')
    ax3.fill_between(DELTA_RANGE,res['hcrit_d'],hr_fixed,
                     where=[hc<=hr_fixed for hc in res['hcrit_d']],alpha=0.15,color='green',label='h sufficient')
    ax3.axvline(delta,color='red',ls='--',lw=1.5,label=f'Baseline δ={delta}')
    ax3.set_xlabel('δ',fontsize=10); ax3.set_ylabel('h_crit = r_stoat − δ',fontsize=10)
    ax3.set_title(f'h_crit vs δ\n({hr_label})',fontsize=10)
    ax3.legend(fontsize=7.5); ax3.grid(True,alpha=0.3)

    # Row 2: r_stoat sweep
    ax4=fig.add_subplot(gs[1,0]); ax5=fig.add_subplot(gs[1,1]); ax6=fig.add_subplot(gs[1,2])
    ax4.plot(RSTOAT_RANGE,[v*100 if not np.isnan(v) else np.nan for v in res['x_r']],'b-o',lw=2,ms=4)
    ax4.axvline(R_STOAT,color='red',ls='--',lw=1.5,label=f'Baseline r_stoat={R_STOAT}')
    ax4.set_xlabel('r_stoat',fontsize=10); ax4.set_ylabel('Open foraging % at t=200',fontsize=10)
    ax4.set_title(f'Open foraging vs r_stoat\n(δ={delta} fixed, {hr_label})',fontsize=10)
    ax4.legend(fontsize=8); ax4.grid(True,alpha=0.3)

    ax5.plot(RSTOAT_RANGE,[v if not np.isnan(v) else np.nan for v in res['K_r']],'g-o',lw=2,ms=4)
    ax5.axhline(K0,color='gray',ls=':',lw=1.2,alpha=0.7,label=f'K0={K0}')
    ax5.axhline(K_MAX,color='green',ls=':',lw=1.2,alpha=0.5,label=f'K_max={K_MAX}')
    ax5.axvline(R_STOAT,color='red',ls='--',lw=1.5,label=f'Baseline={R_STOAT}')
    brs=[rs for rs,hc in zip(RSTOAT_RANGE,res['hcrit_r']) if hc>hr_fixed]
    if brs: ax5.axvspan(max(brs),max(RSTOAT_RANGE),alpha=0.08,color='red',label='h < h_crit')
    grs=[rs for rs,hc in zip(RSTOAT_RANGE,res['hcrit_r']) if hc<=hr_fixed]
    if grs: ax5.axvspan(min(RSTOAT_RANGE),max(grs),alpha=0.08,color='green',label='h ≥ h_crit')
    ax5.set_xlabel('r_stoat',fontsize=10); ax5.set_ylabel('Kiwi at t=200',fontsize=10)
    ax5.set_title(f'Kiwi outcome vs r_stoat\n(δ={delta} fixed, {hr_label})',fontsize=10)
    ax5.legend(fontsize=7.5); ax5.grid(True,alpha=0.3)

    ax6.plot(RSTOAT_RANGE,res['hcrit_r'],'purple',lw=2.5)
    ax6.axhline(hr_fixed,color='orange',ls='--',lw=2,label=f'h={hr_fixed:.2f} (fixed harvest)')
    ax6.fill_between(RSTOAT_RANGE,res['hcrit_r'],hr_fixed,
                     where=[hc>hr_fixed for hc in res['hcrit_r']],alpha=0.15,color='red',label='h insufficient')
    ax6.fill_between(RSTOAT_RANGE,res['hcrit_r'],hr_fixed,
                     where=[hc<=hr_fixed for hc in res['hcrit_r']],alpha=0.15,color='green',label='h sufficient')
    ax6.axvline(R_STOAT,color='red',ls='--',lw=1.5,label=f'Baseline={R_STOAT}')
    ax6.set_xlabel('r_stoat',fontsize=10); ax6.set_ylabel('h_crit = r_stoat − δ',fontsize=10)
    ax6.set_title(f'h_crit vs r_stoat\n({hr_label})',fontsize=10)
    ax6.legend(fontsize=7.5); ax6.grid(True,alpha=0.3)

    # Row 3: 2D heatmaps
    ax7=fig.add_subplot(gs[2,0]); ax8=fig.add_subplot(gs[2,1]); ax9=fig.add_subplot(gs[2,2])
    D_mesh,R_mesh=np.meshgrid(DELTA_RANGE,RSTOAT_RANGE,indexing='ij')
    im7=ax7.contourf(D_mesh,R_mesh,res['X_grid'],levels=20,cmap='RdYlGn')
    r_hcrit=DELTA_RANGE+hr_fixed
    mask=(r_hcrit>=RSTOAT_RANGE[0])&(r_hcrit<=RSTOAT_RANGE[-1])
    ax7.plot(DELTA_RANGE[mask],r_hcrit[mask],'k-',lw=2.5,label='h_crit boundary')
    ax7.plot(delta,R_STOAT,'w*',ms=14,label='Baseline',zorder=5)
    plt.colorbar(im7,ax=ax7,label='Open foraging % at t=200')
    ax7.set_xlabel('δ',fontsize=10); ax7.set_ylabel('r_stoat',fontsize=10)
    ax7.set_title(f'Open foraging % — 2D\n({hr_label})',fontsize=10); ax7.legend(fontsize=8)

    im8=ax8.contourf(D_mesh,R_mesh,res['K_grid'],levels=20,cmap='RdYlGn')
    ax8.contour(D_mesh,R_mesh,res['K_grid'],levels=[K0],colors='white',linewidths=1.5,linestyles='--')
    ax8.plot(DELTA_RANGE[mask],r_hcrit[mask],'k-',lw=2.5,label='h_crit boundary')
    ax8.plot(delta,R_STOAT,'w*',ms=14,label='Baseline',zorder=5)
    plt.colorbar(im8,ax=ax8,label='Kiwi at t=200')
    ax8.set_xlabel('δ',fontsize=10); ax8.set_ylabel('r_stoat',fontsize=10)
    ax8.set_title(f'Kiwi population — 2D\n({hr_label})',fontsize=10); ax8.legend(fontsize=8)

    regime=np.where(R_mesh-D_mesh>hr_fixed,1.0,0.0); regime[R_mesh<=D_mesh]=np.nan
    ax9.contourf(D_mesh,R_mesh,regime,levels=[-0.5,0.5,1.5],colors=['#d62728','#2ca02c'],alpha=0.6)
    ax9.plot(DELTA_RANGE[mask],r_hcrit[mask],'k-',lw=3,label=f'h_crit={hr_fixed:.2f} boundary')
    ax9.plot(delta,R_STOAT,'w*',ms=16,zorder=5,label='Baseline')
    from matplotlib.patches import Patch
    ax9.legend(handles=[
        Patch(facecolor='#d62728',alpha=0.6,label=f'h={hr_fixed:.2f} insufficient'),
        Patch(facecolor='#2ca02c',alpha=0.6,label=f'h={hr_fixed:.2f} sufficient'),
        plt.Line2D([0],[0],color='k',lw=2.5,label='h_crit boundary'),
        plt.Line2D([0],[0],marker='*',color='w',markerfacecolor='white',ms=12,lw=0,label='Baseline'),
    ],fontsize=8,loc='upper left')
    ax9.set_xlabel('δ',fontsize=10); ax9.set_ylabel('r_stoat',fontsize=10)
    ax9.set_title(f'Regime map\n({hr_label})',fontsize=10); ax9.grid(True,alpha=0.2)

    slug = hr_label.replace(' ','_').replace('=','').replace('(','').replace(')','').replace('.','')
    fname = f'/mnt/user-data/outputs/sensitivity_{slug}.png'
    fig.suptitle(f'Sensitivity analysis: {hr_label}\nintervention yr={INTERVENTION_YR}, t={T_MAX}, PI_BAR_ESS={PI_BAR_ESS:.4f}',
                 fontsize=12,fontweight='bold')
    plt.savefig(fname,dpi=150,bbox_inches='tight')
    plt.close()
    print(f"Saved: {fname}")

print("\nAll sensitivity plots saved.")

# ============================================================================
# PART 2: ALPHA AND BETA SENSITIVITY
# Varies predation rate (alpha) and conversion efficiency (beta) independently,
# with delta and r_stoat held at baseline. Run at three harvest rates to show
# how parameter sensitivity changes across management regimes.
# ============================================================================

ALPHA_RANGE  = np.linspace(0.010, 0.100, 25)
BETA_RANGE   = np.linspace(0.005, 0.050, 25)
ALPHA_BASE   = alpha    # 0.044
BETA_BASE    = beta     # 0.0175

def make_system_ab(alpha_p, beta_p):
    """ODE system with variable alpha and beta, delta and r_stoat at baseline."""
    S_floor_p = 8*(1 - delta/R_STOAT)
    def system(t, y, hr=0):
        x,K,S = y; x=np.clip(x,0.01,0.99); K=max(K,0); S=max(S,0)
        sig = 1/(1+np.exp(-3.0*(S-1.0)))
        dp = np.array([
            [max(BASE_PAYOFFS[0,0]-0.40*sig,0.01), max(BASE_PAYOFFS[0,1]-0.40*sig,0.01)],
            [max(BASE_PAYOFFS[1,0]-0.15*sig,0.01), max(BASE_PAYOFFS[1,1]-0.15*sig,0.01)]
        ])
        pA=x*dp[0,0]+(1-x)*dp[0,1]; pB=x*dp[1,0]+(1-x)*dp[1,1]
        pav=x*pA+(1-x)*pB; dxdt=x*(pA-pav) if 0<x<1 else 0
        r_eff = r_base*(avg_payoff_base(x)/PI_BAR_ESS)
        a_eff = alpha_p*(x*1.3+(1-x)*0.7)
        f_K   = (a_eff*K)/(1+a_eff*h_handling*K)
        dKdt  = r_eff*K*(1-K/K_MAX)-f_K*S
        nat   = R_STOAT*S*(1-S/S_MAX)+beta_p*f_K*S-delta*S
        nkg   = R_STOAT*S*(1-S/S_MAX)-delta*S
        if S<=S_floor_p and nkg<0: nat=max(nat,0.0)
        return [dxdt, dKdt, nat-hr*S]
    return system

def simulate_ab(alpha_p, beta_p, hr, it=INTERVENTION_YR, t_max=T_MAX):
    sys = make_system_ab(alpha_p, beta_p)
    y0  = [X0, K0, S0]
    t1  = np.linspace(0, it, max(200, it*20))
    t2  = np.linspace(it, t_max, max(800, (t_max-it)*10))
    try:
        s1=solve_ivp(lambda t,y:sys(t,y,0),[0,it],y0,
                     t_eval=t1,method='RK45',rtol=1e-8,atol=1e-10)
        s2=solve_ivp(lambda t,y:sys(t,y,hr),[it,t_max],
                     [s1.y[0][-1],s1.y[1][-1],s1.y[2][-1]],
                     t_eval=t2,method='RK45',rtol=1e-8,atol=1e-10)
        return float(s2.y[0][-1]), float(s2.y[1][-1]), float(s2.y[2][-1])
    except:
        return np.nan, np.nan, np.nan

# ── Run sweeps ────────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("PART 2: Alpha and Beta Sensitivity")
print("="*60)

ab_results = {}
for hr_fixed, hr_label in harvest_runs:
    print(f"\nRunning α/β sensitivity: {hr_label}")

    # 1D alpha sweep (beta fixed)
    x_a=[]; K_a=[]
    for av in ALPHA_RANGE:
        xf,Kf,_ = simulate_ab(av, BETA_BASE, hr_fixed)
        x_a.append(xf); K_a.append(Kf)

    # 1D beta sweep (alpha fixed)
    x_b=[]; K_b=[]
    for bv in BETA_RANGE:
        xf,Kf,_ = simulate_ab(ALPHA_BASE, bv, hr_fixed)
        x_b.append(xf); K_b.append(Kf)

    # 2D grid
    X_grid_ab = np.full((len(ALPHA_RANGE), len(BETA_RANGE)), np.nan)
    K_grid_ab = np.full_like(X_grid_ab, np.nan)
    for i,av in enumerate(ALPHA_RANGE):
        for j,bv in enumerate(BETA_RANGE):
            xf,Kf,_ = simulate_ab(av, bv, hr_fixed)
            X_grid_ab[i,j] = xf*100; K_grid_ab[i,j] = Kf

    ab_results[hr_label] = {'x_a':x_a,'K_a':K_a,'x_b':x_b,'K_b':K_b,
                             'X_grid':X_grid_ab,'K_grid':K_grid_ab,'hr':hr_fixed}
    print(f"  Done.")

# ── Print tables ──────────────────────────────────────────────────────────────
for hr_label, res in ab_results.items():
    hr_fixed = res['hr']
    print(f"\n{'='*65}")
    print(f"α sweep  ({hr_label}, β={BETA_BASE}, t={T_MAX})")
    print(f"Baseline α={ALPHA_BASE}")
    print(f"{'='*65}")
    print(f"{'alpha':>8} {'x% open':>10} {'K (kiwi)':>10}  note")
    print("-"*50)
    for av,xf,Kf in zip(ALPHA_RANGE, res['x_a'], res['K_a']):
        note = '<-- baseline' if abs(av-ALPHA_BASE)<0.002 else ''
        ext  = 'EXTINCT' if (not np.isnan(Kf) and Kf<1) else ''
        print(f"  {av:.4f}   {xf*100 if not np.isnan(xf) else float('nan'):8.1f}%"
              f"   {Kf if not np.isnan(Kf) else float('nan'):8.1f}  {ext}{note}")

    print(f"\n{'='*65}")
    print(f"β sweep  ({hr_label}, α={ALPHA_BASE}, t={T_MAX})")
    print(f"Baseline β={BETA_BASE}")
    print(f"{'='*65}")
    print(f"{'beta':>8} {'x% open':>10} {'K (kiwi)':>10}  note")
    print("-"*50)
    for bv,xf,Kf in zip(BETA_RANGE, res['x_b'], res['K_b']):
        note = '<-- baseline' if abs(bv-BETA_BASE)<0.001 else ''
        ext  = 'EXTINCT' if (not np.isnan(Kf) and Kf<1) else ''
        print(f"  {bv:.4f}   {xf*100 if not np.isnan(xf) else float('nan'):8.1f}%"
              f"   {Kf if not np.isnan(Kf) else float('nan'):8.1f}  {ext}{note}")

# ── Plots ─────────────────────────────────────────────────────────────────────
for hr_label, res in ab_results.items():
    hr_fixed = res['hr']
    fig = plt.figure(figsize=(18, 16))
    gs  = gridspec.GridSpec(3, 3, figure=fig, hspace=0.45, wspace=0.38)
    from matplotlib.patches import Patch

    # Row 1: alpha sweep
    ax1=fig.add_subplot(gs[0,0]); ax2=fig.add_subplot(gs[0,1]); ax3=fig.add_subplot(gs[0,2])

    ax1.plot(ALPHA_RANGE, [v*100 if not np.isnan(v) else np.nan for v in res['x_a']],
             'b-o', lw=2, ms=4)
    ax1.axvline(ALPHA_BASE, color='red', ls='--', lw=1.5, label=f'Baseline α={ALPHA_BASE}')
    ax1.set_xlabel('α (predation rate)', fontsize=10)
    ax1.set_ylabel('Open foraging % at t=200', fontsize=10)
    ax1.set_title(f'Open foraging vs α\n(β={BETA_BASE} fixed, {hr_label})', fontsize=10)
    ax1.legend(fontsize=8); ax1.grid(True, alpha=0.3)

    ax2.plot(ALPHA_RANGE, [v if not np.isnan(v) else np.nan for v in res['K_a']],
             'g-o', lw=2, ms=4)
    ax2.axhline(K0,   color='gray',  ls=':', lw=1.2, alpha=0.7, label=f'K0={K0}')
    ax2.axhline(K_MAX,color='green', ls=':', lw=1.2, alpha=0.5, label=f'K_max={K_MAX}')
    ax2.axvline(ALPHA_BASE, color='red', ls='--', lw=1.5, label=f'Baseline α={ALPHA_BASE}')
    ax2.axhspan(0, 5, alpha=0.06, color='red', label='Extinction zone')
    ax2.set_xlabel('α (predation rate)', fontsize=10)
    ax2.set_ylabel('Kiwi at t=200', fontsize=10)
    ax2.set_title(f'Kiwi outcome vs α\n(β={BETA_BASE} fixed, {hr_label})', fontsize=10)
    ax2.legend(fontsize=7.5); ax2.grid(True, alpha=0.3)

    # Elasticity panel for alpha
    bi = np.argmin(np.abs(ALPHA_RANGE - ALPHA_BASE))
    Ka = res['K_a']
    if (0 < bi < len(Ka)-1 and
        not any(np.isnan(v) for v in [Ka[bi-1], Ka[bi], Ka[bi+1]]) and
        Ka[bi] > 1):
        s = ((Ka[bi+1]-Ka[bi-1])/Ka[bi]) / ((ALPHA_RANGE[bi+1]-ALPHA_RANGE[bi-1])/ALPHA_BASE)
        ax3.text(0.5, 0.5, f'Elasticity of K to α\nat baseline:\n\n{s:.3f}\n\n'
                 f'(1% ↑ α → {s:.2f}% change in K)',
                 ha='center', va='center', fontsize=13, transform=ax3.transAxes,
                 bbox=dict(boxstyle='round,pad=0.6', facecolor='#E1F5EE', edgecolor='#0F6E56'))
    else:
        ax3.text(0.5, 0.5, 'Baseline at or near\nextinction boundary\n— elasticity undefined\nor K≈0',
                 ha='center', va='center', fontsize=12, transform=ax3.transAxes,
                 bbox=dict(boxstyle='round,pad=0.6', facecolor='#FAEEDA', edgecolor='#854F0B'))
    ax3.axis('off')
    ax3.set_title(f'α sensitivity coefficient\n({hr_label})', fontsize=10)

    # Row 2: beta sweep
    ax4=fig.add_subplot(gs[1,0]); ax5=fig.add_subplot(gs[1,1]); ax6=fig.add_subplot(gs[1,2])

    ax4.plot(BETA_RANGE, [v*100 if not np.isnan(v) else np.nan for v in res['x_b']],
             'b-o', lw=2, ms=4)
    ax4.axvline(BETA_BASE, color='red', ls='--', lw=1.5, label=f'Baseline β={BETA_BASE}')
    ax4.set_xlabel('β (conversion efficiency)', fontsize=10)
    ax4.set_ylabel('Open foraging % at t=200', fontsize=10)
    ax4.set_title(f'Open foraging vs β\n(α={ALPHA_BASE} fixed, {hr_label})', fontsize=10)
    ax4.legend(fontsize=8); ax4.grid(True, alpha=0.3)

    ax5.plot(BETA_RANGE, [v if not np.isnan(v) else np.nan for v in res['K_b']],
             'g-o', lw=2, ms=4)
    ax5.axhline(K0,   color='gray',  ls=':', lw=1.2, alpha=0.7, label=f'K0={K0}')
    ax5.axhline(K_MAX,color='green', ls=':', lw=1.2, alpha=0.5, label=f'K_max={K_MAX}')
    ax5.axvline(BETA_BASE, color='red', ls='--', lw=1.5, label=f'Baseline β={BETA_BASE}')
    ax5.axhspan(0, 5, alpha=0.06, color='red', label='Extinction zone')
    ax5.set_xlabel('β (conversion efficiency)', fontsize=10)
    ax5.set_ylabel('Kiwi at t=200', fontsize=10)
    ax5.set_title(f'Kiwi outcome vs β\n(α={ALPHA_BASE} fixed, {hr_label})', fontsize=10)
    ax5.legend(fontsize=7.5); ax5.grid(True, alpha=0.3)

    # Elasticity panel for beta
    bi = np.argmin(np.abs(BETA_RANGE - BETA_BASE))
    Kb = res['K_b']
    if (0 < bi < len(Kb)-1 and
        not any(np.isnan(v) for v in [Kb[bi-1], Kb[bi], Kb[bi+1]]) and
        Kb[bi] > 1):
        s = ((Kb[bi+1]-Kb[bi-1])/Kb[bi]) / ((BETA_RANGE[bi+1]-BETA_RANGE[bi-1])/BETA_BASE)
        ax6.text(0.5, 0.5, f'Elasticity of K to β\nat baseline:\n\n{s:.3f}\n\n'
                 f'(1% ↑ β → {s:.2f}% change in K)',
                 ha='center', va='center', fontsize=13, transform=ax6.transAxes,
                 bbox=dict(boxstyle='round,pad=0.6', facecolor='#E1F5EE', edgecolor='#0F6E56'))
    else:
        ax6.text(0.5, 0.5, 'β insensitive across\nfull range — flat\nresponse throughout',
                 ha='center', va='center', fontsize=12, transform=ax6.transAxes,
                 bbox=dict(boxstyle='round,pad=0.6', facecolor='#FAEEDA', edgecolor='#854F0B'))
    ax6.axis('off')
    ax6.set_title(f'β sensitivity coefficient\n({hr_label})', fontsize=10)

    # Row 3: 2D heatmaps in (alpha, beta) space
    ax7=fig.add_subplot(gs[2,0]); ax8=fig.add_subplot(gs[2,1]); ax9=fig.add_subplot(gs[2,2])
    A_mesh, B_mesh = np.meshgrid(ALPHA_RANGE, BETA_RANGE, indexing='ij')

    im7 = ax7.contourf(A_mesh, B_mesh, res['X_grid'], levels=20, cmap='RdYlGn')
    ax7.axvline(ALPHA_BASE, color='white', ls='--', lw=1.5, label=f'Baseline α')
    ax7.axhline(BETA_BASE,  color='white', ls=':',  lw=1.5, label=f'Baseline β')
    ax7.plot(ALPHA_BASE, BETA_BASE, 'w*', ms=14, label='Baseline', zorder=5)
    plt.colorbar(im7, ax=ax7, label='Open foraging % at t=200')
    ax7.set_xlabel('α (predation rate)', fontsize=10)
    ax7.set_ylabel('β (conversion efficiency)', fontsize=10)
    ax7.set_title(f'Open foraging % — 2D (α, β)\n({hr_label})', fontsize=10)
    ax7.legend(fontsize=8)

    im8 = ax8.contourf(A_mesh, B_mesh, res['K_grid'], levels=20, cmap='RdYlGn')
    ax8.contour(A_mesh, B_mesh, res['K_grid'], levels=[K0],
                colors='white', linewidths=1.5, linestyles='--')
    ax8.axvline(ALPHA_BASE, color='white', ls='--', lw=1.5)
    ax8.axhline(BETA_BASE,  color='white', ls=':',  lw=1.5)
    ax8.plot(ALPHA_BASE, BETA_BASE, 'w*', ms=14, label='Baseline', zorder=5)
    plt.colorbar(im8, ax=ax8, label='Kiwi at t=200')
    ax8.set_xlabel('α (predation rate)', fontsize=10)
    ax8.set_ylabel('β (conversion efficiency)', fontsize=10)
    ax8.set_title(f'Kiwi population — 2D (α, β)\n(white dashed = K={K0} boundary)', fontsize=10)
    ax8.legend(fontsize=8)

    # Recovery/extinction regime map
    regime_ab = np.where(res['K_grid'] >= K0, 1.0, 0.0)
    regime_ab[np.isnan(res['K_grid'])] = np.nan
    ax9.contourf(A_mesh, B_mesh, regime_ab, levels=[-0.5, 0.5, 1.5],
                 colors=['#d62728', '#2ca02c'], alpha=0.6)
    ax9.contour(A_mesh, B_mesh, res['K_grid'], levels=[K0],
                colors='black', linewidths=2)
    ax9.axvline(ALPHA_BASE, color='white', ls='--', lw=1.5)
    ax9.axhline(BETA_BASE,  color='white', ls=':',  lw=1.5)
    ax9.plot(ALPHA_BASE, BETA_BASE, 'w*', ms=16, zorder=5, label='Baseline')
    ax9.legend(handles=[
        Patch(facecolor='#d62728', alpha=0.6, label='Kiwi extinct (K<K0)'),
        Patch(facecolor='#2ca02c', alpha=0.6, label='Kiwi recover (K≥K0)'),
        plt.Line2D([0],[0], color='black', lw=2, label='K=K0 boundary'),
        plt.Line2D([0],[0], marker='*', color='w', markerfacecolor='white',
                   ms=12, lw=0, label='Baseline'),
    ], fontsize=8, loc='upper right')
    ax9.set_xlabel('α (predation rate)', fontsize=10)
    ax9.set_ylabel('β (conversion efficiency)', fontsize=10)
    ax9.set_title(f'Recovery regime map (α, β)\n({hr_label})', fontsize=10)
    ax9.grid(True, alpha=0.2)

    slug = hr_label.replace(' ','_').replace('=','').replace('(','').replace(')','').replace('.','')
    fname = f'/mnt/user-data/outputs/sensitivity_alpha_beta_{slug}.png'
    fig.suptitle(f'Sensitivity to α and β: {hr_label}\n'
                 f'intervention yr={INTERVENTION_YR}, t={T_MAX}, '
                 f'δ={delta}, r_stoat={R_STOAT}, PI_BAR_ESS={PI_BAR_ESS:.4f}',
                 fontsize=12, fontweight='bold')
    plt.savefig(fname, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {fname}")

print("\nAll sensitivity plots saved (δ/r_stoat and α/β).")

# ============================================================================
# PART 3: L-V PARAMETER SENSITIVITY (r, alpha, beta, delta)
# Varies each of the four core L-V parameters across its plausible empirical
# range using CEPPM simulation at three harvest rates (h=0.15, 0.25, 0.40)
# with intervention at year 20. Elasticity computed via finite difference
# at the baseline parameter value.
# Replaces earlier nullcline-based analysis which was derived under S_MAX=5
# and is inconsistent with the current S_MAX=8 parameterisation.
# ============================================================================

print("\n" + "="*60)
print("PART 3: L-V Parameter Sensitivity (r, α, β, δ)")
print("="*60)

LV_PARAMS = [
    ('r',     np.linspace(0.02, 0.20, 30), r,          'r (kiwi growth rate)',       'Kiwi growth rate r',        '#2ca02c'),
    ('alpha', np.linspace(0.01, 0.10, 30), alpha,       'α (predation rate)',          'Predation rate α',          '#d62728'),
    ('beta',  np.linspace(0.005,0.05, 30), beta,        'β (conversion efficiency)',   'Conversion efficiency β',   '#ff7f0e'),
    ('delta', np.linspace(0.15, 0.55, 30), delta,       'δ (stoat natural mortality)', 'Stoat natural mortality δ', '#1f77b4'),
]
LV_HARVEST = [
    (0.15, 'h=0.15 (failed zone)', '#d62728'),
    (0.25, 'h=0.25 (= h_crit)',    '#ff7f0e'),
    (0.40, 'h=0.40 (above h_crit)','#2ca02c'),
]
LV_IT   = 20
LV_TMAX = T_MAX

def simulate_lv_param(r_p, alpha_p, beta_p, delta_p, hr, it, t_max):
    """Simulate CEPPM with variable L-V parameters and return K at t_max."""
    S_floor_p = S_MAX*(1 - delta_p/R_STOAT) if R_STOAT > delta_p else 0.0
    def system(t, y):
        x, K, S = y
        x = np.clip(x, 0.01, 0.99); K = max(K, 0); S = max(S, 0)
        sig = 1/(1 + np.exp(-3.0*(S - 1.0)))
        dp = np.array([
            [max(BASE_PAYOFFS[0,0]-0.40*sig,0.01), max(BASE_PAYOFFS[0,1]-0.40*sig,0.01)],
            [max(BASE_PAYOFFS[1,0]-0.15*sig,0.01), max(BASE_PAYOFFS[1,1]-0.15*sig,0.01)]
        ])
        pA  = x*dp[0,0]+(1-x)*dp[0,1]; pB=x*dp[1,0]+(1-x)*dp[1,1]
        pav = x*pA+(1-x)*pB
        dxdt = x*(pA-pav) if 0<x<1 else 0
        r_eff  = r_p*(avg_payoff_base(x)/PI_BAR_ESS)
        a_eff  = alpha_p*(x*1.3+(1-x)*0.7)
        f_K    = (a_eff*K)/(1+a_eff*h_handling*K)
        dKdt   = r_eff*K*(1-K/K_MAX)-f_K*S
        nat    = R_STOAT*S*(1-S/S_MAX)+beta_p*f_K*S-delta_p*S
        nkg    = R_STOAT*S*(1-S/S_MAX)-delta_p*S
        if S <= S_floor_p and nkg < 0: nat = max(nat, 0.0)
        dSdt   = nat - hr*S
        return [dxdt, dKdt, dSdt]

    y0 = [X0, K0, S0]
    t1 = np.linspace(0, it, max(200, it*20))
    t2 = np.linspace(it, t_max, max(800, (t_max-it)*10))
    try:
        s1 = solve_ivp(lambda t,y: system(t,y), [0,it], y0,
                       t_eval=t1, method='RK45', rtol=1e-8, atol=1e-10)
        # Switch to harvest rate after intervention
        def sys_hr(t, y):
            x,K,S = y; x=np.clip(x,0.01,0.99); K=max(K,0); S=max(S,0)
            sig=1/(1+np.exp(-3.0*(S-1.0)))
            dp=np.array([[max(BASE_PAYOFFS[0,0]-0.40*sig,0.01),max(BASE_PAYOFFS[0,1]-0.40*sig,0.01)],
                         [max(BASE_PAYOFFS[1,0]-0.15*sig,0.01),max(BASE_PAYOFFS[1,1]-0.15*sig,0.01)]])
            pA=x*dp[0,0]+(1-x)*dp[0,1]; pB=x*dp[1,0]+(1-x)*dp[1,1]
            pav=x*pA+(1-x)*pB; dxdt=x*(pA-pav) if 0<x<1 else 0
            r_eff=r_p*(avg_payoff_base(x)/PI_BAR_ESS); a_eff=alpha_p*(x*1.3+(1-x)*0.7)
            f_K=(a_eff*K)/(1+a_eff*h_handling*K); dKdt=r_eff*K*(1-K/K_MAX)-f_K*S
            nat=R_STOAT*S*(1-S/S_MAX)+beta_p*f_K*S-delta_p*S
            nkg=R_STOAT*S*(1-S/S_MAX)-delta_p*S
            if S<=S_floor_p and nkg<0: nat=max(nat,0.0)
            return [dxdt,dKdt,nat-hr*S]
        s2 = solve_ivp(sys_hr, [it, t_max],
                       [s1.y[0][-1], s1.y[1][-1], s1.y[2][-1]],
                       t_eval=t2, method='RK45', rtol=1e-8, atol=1e-10)
        return float(s2.y[1][-1])
    except:
        return np.nan

# ── Run sweeps ────────────────────────────────────────────────────────────────
lv_results      = {}
lv_elasticities = {}

for hr, hrlabel, hrcol in LV_HARVEST:
    lv_results[hr]      = {}
    lv_elasticities[hr] = {}
    for pname, prange, pbase, xlabel, title, col in LV_PARAMS:
        K_vals = []
        for v in prange:
            rp = v     if pname=='r'     else r
            ap = v     if pname=='alpha' else alpha
            bp = v     if pname=='beta'  else beta
            dp = v     if pname=='delta' else delta
            K_vals.append(simulate_lv_param(rp, ap, bp, dp, hr, LV_IT, LV_TMAX))
        lv_results[hr][pname] = K_vals

        bi = np.argmin(np.abs(prange - pbase))
        if 0 < bi < len(K_vals)-1:
            k_lo, k_hi, k_c = K_vals[bi-1], K_vals[bi+1], K_vals[bi]
            if not any(np.isnan(v) for v in [k_lo,k_hi,k_c]) and k_c and k_c>0:
                e=((k_hi-k_lo)/k_c)/((prange[bi+1]-prange[bi-1])/pbase)
                lv_elasticities[hr][pname] = e
            else:
                lv_elasticities[hr][pname] = None
        else:
            lv_elasticities[hr][pname] = None
    print(f"  L-V sweep h={hr}: done")

# ── Print updated table ────────────────────────────────────────────────────────
print(f"\n{'='*75}")
print(f"L-V PARAMETER SENSITIVITY TABLE — intervention yr={LV_IT}, t={LV_TMAX}")
print(f"{'='*75}")
print(f"{'Param':<8} {'Range':<14} {'h=0.15':>10} {'h=0.25':>10} {'h=0.40':>10}")
print("-"*55)
ranges_lv = {'r':'0.02–0.20','alpha':'0.01–0.10','beta':'0.005–0.050','delta':'0.15–0.55'}
for pname,_,_,_,_,_ in LV_PARAMS:
    row = f"  {pname:<6}  {ranges_lv[pname]:<14}"
    for hr,_,_ in LV_HARVEST:
        e = lv_elasticities[hr][pname]
        row += f"  {e:>+8.3f}" if e is not None else f"  {'undef.':>8}"
    print(row)

# ── Summary plot: 3 rows (harvest rates) × 4 cols (parameters) ───────────────
fig, axes = plt.subplots(3, 4, figsize=(20, 14))
fig.suptitle(
    f'L-V Parameter Sensitivity — Kiwi population K at t={LV_TMAX}\n'
    f'CEPPM simulation, intervention yr={LV_IT}  |  '
    f'Rows = harvest rate  |  Columns = parameter varied',
    fontsize=12, fontweight='bold'
)

for row, (hr, hrlabel, hrcol) in enumerate(LV_HARVEST):
    for col, (pname, prange, pbase, xlabel, title, pcol) in enumerate(LV_PARAMS):
        ax      = axes[row, col]
        K_vals  = lv_results[hr][pname]

        ax.plot(prange, K_vals, color=hrcol, lw=2.5, zorder=3)
        ax.axvline(pbase, color='black', ls='--', lw=1.5, alpha=0.8,
                   label=f'Baseline={pbase}')
        ax.axhline(K_MAX, color='green', ls=':', lw=1.2, alpha=0.5,
                   label=f'K_max={K_MAX}')
        ax.axhline(K0, color='gray', ls=':', lw=1.2, alpha=0.5,
                   label=f'K0={K0}')
        ax.axhspan(0, 5, alpha=0.06, color='red')

        e    = lv_elasticities[hr][pname]
        estr = f'ε={e:+.2f}' if e is not None else 'ε=undef.'
        ax.text(0.97, 0.97, estr, transform=ax.transAxes, fontsize=9,
                ha='right', va='top',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                          edgecolor='gray', alpha=0.85))

        if col == 0:
            ax.set_ylabel(f'{hrlabel}\nKiwi K at t={LV_TMAX}', fontsize=9)
        if row == 0:
            ax.set_title(title, fontsize=10)
        ax.set_xlabel(xlabel, fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(-5, K_MAX+10)
        if row == 0 and col == 0:
            ax.legend(fontsize=7)

plt.tight_layout()
lv_fname = '/mnt/user-data/outputs/sensitivity_lv_parameters.png'
plt.savefig(lv_fname, dpi=150, bbox_inches='tight')
plt.close()
print(f"\nSaved: {lv_fname}")
print("\nAll sensitivity plots saved (δ/r_stoat, α/β, and L-V parameters).")

# ============================================================================
# PART 4: INTERVENTION TIMING SENSITIVITY
# Tests how sensitive kiwi outcomes are to the year in which predator control
# begins, at two harvest rates:
#   h=0.16 — approximately the effective kiwi recovery threshold
#   h=0.20 — Scenario 1b (above effective threshold, below h_crit)
# Sweep covers intervention years 0–30 at t_max=200.
# Key finding: at h=0.16 timing is irrelevant (harvest rate insufficient
# for meaningful recovery regardless of when it starts); at h=0.20 timing
# matters — earlier is better — with K dropping below K0 if intervention
# is delayed beyond approximately year 22.
# ============================================================================

print("\n" + "="*60)
print("PART 4: Intervention Timing Sensitivity")
print("="*60)

IT_RANGE  = list(range(0, 31))
SNAPSHOTS_T = [50, 100, 200]
TIMING_CONFIGS = [
    (0.16, 'h=0.16 (≈ effective recovery threshold)', '#9467bd'),
    (0.20, 'h=0.20 (Scenario 1b)',                    '#ff7f0e'),
]

def simulate_timing(it, hr, t_max=T_MAX):
    """Simulate CEPPM with given intervention year and harvest rate."""
    y0 = [X0, K0, S0]
    if it == 0:
        t_eval = np.linspace(0, t_max, 2000)
        sol = solve_ivp(lambda t,y: hybrid_system_timing(t, y, hr),
                        [0, t_max], y0, t_eval=t_eval,
                        method='RK45', rtol=1e-8, atol=1e-10)
        return sol.t, sol.y[0], sol.y[1], sol.y[2]
    t1 = np.linspace(0, it, max(400, it*20))
    t2 = np.linspace(it, t_max, max(800, (t_max-it)*10))
    s1 = solve_ivp(lambda t,y: hybrid_system_timing(t, y, 0),
                   [0, it], y0, t_eval=t1,
                   method='RK45', rtol=1e-8, atol=1e-10)
    s2 = solve_ivp(lambda t,y: hybrid_system_timing(t, y, hr),
                   [it, t_max],
                   [s1.y[0][-1], s1.y[1][-1], s1.y[2][-1]],
                   t_eval=t2, method='RK45', rtol=1e-8, atol=1e-10)
    return (np.concatenate([s1.t,    s2.t]),
            np.concatenate([s1.y[0], s2.y[0]]),
            np.concatenate([s1.y[1], s2.y[1]]),
            np.concatenate([s1.y[2], s2.y[2]]))

def hybrid_system_timing(t, y, hr=0):
    """Identical to main hybrid system — separate function for clarity."""
    x, K, S = y
    x = np.clip(x, 0.01, 0.99); K = max(K, 0); S = max(S, 0)
    sig = 1 / (1 + np.exp(-3.0*(S - 1.0)))
    dp = np.array([
        [max(BASE_PAYOFFS[0,0]-0.40*sig, 0.01), max(BASE_PAYOFFS[0,1]-0.40*sig, 0.01)],
        [max(BASE_PAYOFFS[1,0]-0.15*sig, 0.01), max(BASE_PAYOFFS[1,1]-0.15*sig, 0.01)]
    ])
    pA  = x*dp[0,0]+(1-x)*dp[0,1]; pB=x*dp[1,0]+(1-x)*dp[1,1]
    pav = x*pA+(1-x)*pB
    dxdt = x*(pA-pav) if 0 < x < 1 else 0
    r_eff  = r*(avg_payoff_base(x)/PI_BAR_ESS)
    a_eff  = alpha*(x*1.3+(1-x)*0.7)
    f_K    = (a_eff*K)/(1+a_eff*h_handling*K)
    dKdt   = r_eff*K*(1-K/K_MAX)-f_K*S
    nat    = R_STOAT*S*(1-S/S_MAX)+beta*f_K*S-delta*S
    nkg    = R_STOAT*S*(1-S/S_MAX)-delta*S
    S_floor_t = S_MAX*(1-delta/R_STOAT)
    if S <= S_floor_t and nkg < 0: nat = max(nat, 0.0)
    dSdt   = nat - hr*S
    return [dxdt, dKdt, dSdt]

# ── Run sweeps ────────────────────────────────────────────────────────────────
timing_results = {}
for hr, hrlabel, hrcol in TIMING_CONFIGS:
    it_K = {t: [] for t in SNAPSHOTS_T}
    K_finals = []; S_finals = []; x_finals = []
    for it in IT_RANGE:
        t_arr, x_arr, K_arr, S_arr = simulate_timing(it, hr)
        for tq in SNAPSHOTS_T:
            it_K[tq].append(float(np.interp(tq, t_arr, K_arr)))
        K_finals.append(float(K_arr[-1]))
        S_finals.append(float(S_arr[-1]))
        x_finals.append(float(x_arr[-1])*100)
    timing_results[hr] = {
        'K': it_K, 'K_final': K_finals,
        'S_final': S_finals, 'x_final': x_finals,
        'label': hrlabel, 'color': hrcol
    }
    print(f"  {hrlabel}: done")

# ── Print table ───────────────────────────────────────────────────────────────
print(f"\n{'='*80}")
print(f"INTERVENTION TIMING SENSITIVITY TABLE — K at t=50, 100, 200")
print(f"{'='*80}")
for hr, hrlabel, hrcol in TIMING_CONFIGS:
    res = timing_results[hr]
    print(f"\n{hrlabel}")
    print(f"{'Int. yr':>8} {'K@t=50':>10} {'K@t=100':>10} {'K@t=200':>10} "
          f"{'S@t=200':>10} {'x%@t=200':>10}")
    print("-"*62)
    for i, it in enumerate(IT_RANGE):
        print(f"  {it:>6}   {res['K'][50][i]:>10.1f}  {res['K'][100][i]:>10.1f}  "
              f"{res['K_final'][i]:>10.1f}  {res['S_final'][i]:>10.3f}  "
              f"{res['x_final'][i]:>9.1f}%")

# ── Plots ─────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 3, figsize=(17, 11))
fig.suptitle(
    'Sensitivity to Intervention Timing\n'
    'h=0.16 (≈ effective recovery threshold) vs h=0.20 (Scenario 1b)',
    fontsize=12, fontweight='bold'
)

snapshot_cols = ['#1f77b4', '#2ca02c', '#d62728']

# Row 0: K vs intervention year at t=50, 100, 200
for col, (tq, scol) in enumerate(zip(SNAPSHOTS_T, snapshot_cols)):
    ax = axes[0, col]
    for hr, hrlabel, hrcol in TIMING_CONFIGS:
        res = timing_results[hr]
        ax.plot(IT_RANGE, res['K'][tq], color=hrcol, lw=2.5, label=hrlabel)
    ax.axhline(K_MAX, color='green', ls=':', lw=1.2, alpha=0.5, label=f'K_max={K_MAX}')
    ax.axhline(K0,    color='gray',  ls=':', lw=1.2, alpha=0.5, label=f'K0={K0}')
    ax.axhspan(0, 5, alpha=0.06, color='red', label='Extinction zone')
    ax.set_xlabel('Intervention year', fontsize=11)
    ax.set_ylabel(f'Kiwi K at t={tq}', fontsize=11)
    ax.set_title(f'Kiwi population at t={tq}', fontsize=12)
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 30); ax.set_ylim(-5, K_MAX+10)

# Row 1, col 0: Stoat at t=200
ax = axes[1, 0]
for hr, hrlabel, hrcol in TIMING_CONFIGS:
    res = timing_results[hr]
    ax.plot(IT_RANGE, res['S_final'], color=hrcol, lw=2.5, label=hrlabel)
S_nk = S_MAX*(1-delta/R_STOAT)
ax.axhline(S_nk, color='darkred', ls='--', lw=1.5, alpha=0.7, label=f'S_nk={S_nk:.2f}')
ax.set_xlabel('Intervention year', fontsize=11)
ax.set_ylabel('Stoat S at t=200', fontsize=11)
ax.set_title('Stoat population at t=200', fontsize=12)
ax.legend(fontsize=8); ax.grid(True, alpha=0.3); ax.set_xlim(0, 30)

# Row 1, col 1: Open foraging at t=200
ax = axes[1, 1]
for hr, hrlabel, hrcol in TIMING_CONFIGS:
    res = timing_results[hr]
    ax.plot(IT_RANGE, res['x_final'], color=hrcol, lw=2.5, label=hrlabel)
ax.axhline(66.3, color='purple', ls=':', lw=1.2, alpha=0.6, label='Simulated ESS 66.3%')
ax.axhline(19.1, color='gray',   ls=':', lw=1.2, alpha=0.6, label='Unmanaged 19.1%')
ax.set_xlabel('Intervention year', fontsize=11)
ax.set_ylabel('Open foraging % at t=200', fontsize=11)
ax.set_title('Open foraging proportion at t=200', fontsize=12)
ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
ax.set_xlim(0, 30); ax.set_ylim(0, 100)

# Row 1, col 2: Critical timing — K@200 vs intervention year with K0 crossing
ax = axes[1, 2]
for hr, hrlabel, hrcol in TIMING_CONFIGS:
    res = timing_results[hr]
    ax.plot(IT_RANGE, res['K_final'], color=hrcol, lw=2.5, label=hrlabel)
ax.axhline(K_MAX, color='green', ls=':', lw=1.2, alpha=0.5, label=f'K_max={K_MAX}')
ax.axhline(K0,    color='gray',  ls='--', lw=1.5, alpha=0.7, label=f'K0={K0}')
ax.axhspan(0, K0, alpha=0.06, color='red', label='Net population loss zone')
for hr, hrlabel, hrcol in TIMING_CONFIGS:
    res = timing_results[hr]
    for i in range(len(IT_RANGE)-1):
        if res['K_final'][i] >= K0 > res['K_final'][i+1]:
            ax.axvline(IT_RANGE[i], color=hrcol, ls='--', lw=1.5, alpha=0.7,
                       label=f'Critical yr ≈ {IT_RANGE[i]} ({hrlabel[:6]})')
            break
ax.set_xlabel('Intervention year', fontsize=11)
ax.set_ylabel('Kiwi K at t=200', fontsize=11)
ax.set_title('Critical intervention timing\n(below K0 = net population loss)', fontsize=11)
ax.legend(fontsize=7.5); ax.grid(True, alpha=0.3)
ax.set_xlim(0, 30); ax.set_ylim(-5, K_MAX+10)

plt.tight_layout()
timing_fname = '/mnt/user-data/outputs/sensitivity_intervention_timing.png'
plt.savefig(timing_fname, dpi=150, bbox_inches='tight')
plt.close()
print(f"\nSaved: {timing_fname}")
print("\nAll sensitivity plots saved (δ/r_stoat, α/β, L-V parameters, intervention timing).")
