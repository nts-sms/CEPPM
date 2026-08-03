#=
CEPPM 3D Bifurcation Cross-Check — Julia / BifurcationKit
===========================================================

Direct port of `hybrid_system()` from kiwi_conservation_analysis_v7.py.
State vector u = [x, K, S] (open-foraging proportion, kiwi pop, stoat pop).
Continuation parameter: h (harvest rate).

Purpose (per session handoff v26): independent cross-check of the Python
3D self-consistent bifurcation results — automatic bifurcation detection
(transcritical/fold/Hopf) rather than pre-specifying h_dynamic/h_erad, and
check whether Branch B and Branch D connect at a bifurcation point.

KNOWN NON-SMOOTHNESS: the stoat natural growth term has a floor clamp
(`if S <= S_floor and non_kiwi_growth < 0: natural_dSdt = max(natural_dSdt, 0)`).
This introduces a kink in F at S = S_floor = S_max*(1-delta/r_stoat) = 3.333.
Continuation should be well-behaved on branches that stay clear of this
value, but if Newton convergence gets flaky or step sizes collapse near
S≈3.333, that's the likely cause — not a setup bug.

CONFIRMED FINDINGS (this session):
1. Branch B / Branch D meet at h ≈ 0.16853 (transcritical). Confirmed three
   ways: (a) bp detected on the original branch, (b) branch switching lands
   on a second bp at the same h to 5 sig figs, (c) the switched branch has
   K≡0 identically (floating-point noise only) at every sampled point —
   i.e. it IS the K=0 invariant subspace (Branch B). Stability flips from
   stable (n_unstable=0) to unstable (n_unstable=1) at exactly this h.
2. Branch D / Branch C: NOT cleanly caught by automatic detection. S
   declines smoothly and monotonically as h increases past ~0.19, reaching
   S≈8e-5 at h≈0.3258 — a genuine crossing, just not flagged as a `bp`.
   The very last continuation step then jumps discontinuously to h=0 with
   S slightly negative — a Newton-corrector artifact from the Jacobian
   becoming singular right at S=0, not a real branch point. Treat h≈0.326
   as the D/C crossing location (extrapolated from the smooth approach),
   not the algorithm's own (unreliable) final point.
   NOTE: this is HIGHER than the h_erad≈0.3196 used in the Python scripts
   and appendix — that figure was computed with alpha_base=0.044 rather
   than alpha_effective evaluated at the coexistence branch's true x*≈0.762
   (avg_vuln-weighted). Using alpha_eff(x*) gives h_erad≈0.3258, matching
   this run. The Python-side h_erad computation needs the same fix before
   trusting either number in the writeup — see session handoff for scope.
=#

using Accessors         # provides @optic (newer BifurcationKit expects this,
                         # not Setfield's @lens)
using NonlinearSolve
using BifurcationKit
using ForwardDiff
using Plots

const BK = BifurcationKit

# ----------------------------------------------------------------------
# 1. Parameters (fixed values from session_handoff_v26 §1; h is the
#    continuation parameter, given a starting value here)
#
#    Using a plain NamedTuple rather than a typed struct: automatic branch
#    switching perturbs h with a ForwardDiff.Dual internally to compute the
#    normal form, then reconstructs the parameter object with that Dual in
#    place of h. A struct with `h::Float64` can't hold a Dual there and
#    errors. A NamedTuple tracks each field's type independently in its
#    own type signature, so only the h slot's type changes — no conflict.
# ----------------------------------------------------------------------
default_params = (
    a = 0.35,     b = 0.55,     c = 0.45,     d = 0.20,     # base payoffs
    kappa_learn = 1.5,                                       # EGT adapt rate, /yr
    r_base = 0.05, alpha_base = 0.044, beta = 0.0175, delta = 0.35,
    r_stoat = 0.60, S_max = 8.0, K_max = 150.0, H = 0.1,     # Holling handling time
    h = 0.20,                                                # CONTINUATION PARAMETER
)

# ----------------------------------------------------------------------
# 2. F(u, p) — exact port of hybrid_system()
# ----------------------------------------------------------------------
function F(u, p)
    x, K, S = u
    (; a, b, c, d, kappa_learn, r_base, alpha_base, beta,
       delta, r_stoat, S_max, K_max, H, h) = p

    # NOTE: the Python hybrid_system() applies np.clip(x,0.01,0.99) and
    # max(K,0), max(S,0) here as ODE-integration safety rails. Deliberately
    # OMITTED for continuation: (1) they break Jacobian smoothness — a trial
    # Newton point landing outside the clamp gives a zero row/column and a
    # singular Jacobian, even though the converged branch never sits there;
    # (2) we specifically want continuation to detect S -> 0 as a genuine
    # bifurcation (the h_erad transition), not have it silently clamped away.

    # --- dynamic (sigmoid-penalized) payoffs, feed the EGT replicator ---
    k_sig    = 3.0
    S_mid    = 1.0
    open_max = 0.30
    cov_max  = 0.15
    sig      = 1 / (1 + exp(-k_sig * (S - S_mid)))
    open_pen = open_max * sig
    cov_pen  = cov_max  * sig

    a_dyn = max(a - open_pen, 0.01)
    b_dyn = max(b - open_pen, 0.01)
    c_dyn = max(c - cov_pen,  0.01)
    d_dyn = max(d - cov_pen,  0.01)

    piA_dyn  = x * a_dyn + (1 - x) * b_dyn
    piB_dyn  = x * c_dyn + (1 - x) * d_dyn
    pi_avg   = x * piA_dyn + (1 - x) * piB_dyn
    dxdt     = kappa_learn * x * (piA_dyn - pi_avg)

    # --- strategy-dependent L-V parameters, use BASE payoffs (not dynamic) ---
    open_vuln  = 1.3
    cover_vuln = 0.7
    avg_vuln   = x * open_vuln + (1 - x) * cover_vuln

    piA_base(xx) = xx * a + (1 - xx) * b
    piB_base(xx) = xx * c + (1 - xx) * d
    avgpay_base(xx) = xx * piA_base(xx) + (1 - xx) * piB_base(xx)

    denom  = (a - b) + (d - c)
    x_star = abs(denom) > 1e-10 ? (d - b) / denom : 0.5
    x_star = clamp(x_star, 0.0, 1.0)

    PI_BAR_ESS = avgpay_base(x_star)
    avg_repro  = avgpay_base(x)

    r_eff     = r_base * (avg_repro / PI_BAR_ESS)
    alpha_eff = alpha_base * avg_vuln

    # --- Holling Type II predation ---
    fK   = (alpha_eff * K) / (1 + alpha_eff * H * K)
    dKdt = r_eff * K * (1 - K / K_max) - fK * S

    # --- generalist stoat dynamics + kiwi-predation bonus − harvest ---
    natural_dSdt     = r_stoat * S * (1 - S / S_max) + beta * fK * S - delta * S
    S_floor          = S_max * (1 - delta / r_stoat)
    non_kiwi_growth  = r_stoat * S * (1 - S / S_max) - delta * S
    if S <= S_floor && non_kiwi_growth < 0
        natural_dSdt = max(natural_dSdt, 0.0)   # NON-SMOOTH — see header note
    end
    harvest_dSdt = -h * S
    dSdt = natural_dSdt + harvest_dSdt

    return [dxdt, dKdt, dSdt]
end

# ----------------------------------------------------------------------
# 3. Find a converged Branch D (coexistence) equilibrium at a starting h
#    to seed continuation. Adjust u0_guess if this fails to converge —
#    x near x*=7/9≈0.778, K moderate (tens), S low-but-nonzero is the
#    right neighbourhood for a managed/coexistence starting point.
# ----------------------------------------------------------------------
p0 = merge(default_params, (h = 0.20,))
u0_guess = [0.75, 80.0, 1.0]

ss_prob = NonlinearProblem((u, p) -> F(u, p), u0_guess, p0)
ss_sol  = solve(ss_prob, NewtonRaphson())
u0 = ss_sol.u

println("Seed equilibrium at h = $(p0.h):")
println("  x* = $(u0[1]),  K* = $(u0[2]),  S* = $(u0[3])")
println("  residual |F(u0)| = $(maximum(abs.(F(u0, p0))))")

# ----------------------------------------------------------------------
# 4. Set up the bifurcation problem and continue over h
# ----------------------------------------------------------------------
prob = BifurcationProblem(
    F, u0, p0, (@optic _.h);
    J = (u, p) -> ForwardDiff.jacobian(z -> F(z, p), u),
)

opts = ContinuationPar(
    dsmax = 0.01, dsmin = 1e-6, ds = 0.001,
    p_min = 0.0,  p_max = 0.6,
    max_steps = 8000,
    detect_bifurcation = 3,   # auto-detect fold / transcritical / Hopf
    n_inversion = 6,
    nev = 3,                 # track all 3 eigenvalues (state dim = 3)
)

br = continuation(prob, PALC(), opts; bothside = true)

println("\nDetected bifurcation points:")
println(br)

plot(br, vars = (:param, :x))

# ----------------------------------------------------------------------
# 5. Branch switching at the detected transcritical point (h ≈ 0.16853)
#    — confirms Branch B / Branch D connect there (see header §1).
# ----------------------------------------------------------------------
println("\n--- Branch switching at bp ---")

bp_indices = findall(sp -> sp.type == :bp, br.specialpoint)
println("bp special-point indices found: ", bp_indices)

for idx in bp_indices
    println("\nSwitching branch at bp index $idx (h = $(br.specialpoint[idx].param))...")
    try
        br2 = continuation(br, idx, opts; bothside = true)
        println(br2)
        plot(br2, vars = (:param, :x))
    catch e
        println("Branch switching failed at index $idx: ", e)
    end
end

# ----------------------------------------------------------------------
# 6. Endpoint diagnostics — check whether continuation endpoints are
#    genuine feasibility boundaries or just step-count/Newton artifacts,
#    by probing just past each one with a fresh, independent Newton solve.
# ----------------------------------------------------------------------
println("\n--- Endpoint diagnostics ---")

u_low, p_low   = br.sol[1].x,   br.sol[1].p
u_high, p_high = br.sol[end].x, br.sol[end].p

println("Near LOW endpoint  (h ≈ $p_low):  x=$(u_low[1]), K=$(u_low[2]), S=$(u_low[3])")
println("Near HIGH endpoint (h ≈ $p_high): x=$(u_high[1]), K=$(u_high[2]), S=$(u_high[3])")

function probe_past_endpoint(u_guess, p_at, h_step; label = "")
    p_test    = merge(p0, (h = p_at + h_step,))
    prob_test = NonlinearProblem((u, p) -> F(u, p), u_guess, p_test)
    sol_test  = solve(prob_test, NewtonRaphson())
    resid     = maximum(abs.(F(sol_test.u, p_test)))
    println("$label h=$(p_test.h): converged to x=$(sol_test.u[1]), " *
            "K=$(sol_test.u[2]), S=$(sol_test.u[3]), residual=$resid")
    return sol_test.u
end

println("\nProbing just past LOW endpoint (h decreasing further):")
probe_past_endpoint(u_low, p_low, -0.01; label = "  ")

println("\nProbing just past HIGH endpoint (h increasing further):")
probe_past_endpoint(u_high, p_high, 0.01; label = "  ")
