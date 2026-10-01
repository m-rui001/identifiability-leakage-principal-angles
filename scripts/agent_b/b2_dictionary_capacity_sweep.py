"""
B Agent / D2: Dictionary capacity sweep for Duffing oscillator
Question: Does model-form error (I_lin(K*) - I_emp) shrink with richer dictionaries?
If yes: S2 certifies the PRIMARY error after dictionary upgrade.
If no: model-form error is intrinsic to linearization; S3 correction is essential.

System: Duffing double-well: dx = v dt, dv = (x - x^3 - delta*v + gamma*cos(omega*t)) dt
       + process noise sigma_p
"""
import numpy as np

print("="*80)
print("[B-D2] Dictionary capacity sweep: Duffing double-well")
print("="*80)

rng = np.random.default_rng(2026)

# Duffing parameters
delta = 0.1; gamma = 0.3; omega = 1.0; sigma_p = 1e-2
dt = 0.01; T_total = 2000.0; burn_in = 200.0

def duffing_step(x, v, t):
    a = x - x**3 - delta*v + gamma*np.cos(omega*t)
    return v, a

def simulate(n_steps, x0, v0, t0=0.0):
    xs = np.zeros(n_steps); vs = np.zeros(n_steps)
    xs[0], vs[0] = x0, v0
    for i in range(1, n_steps):
        t = t0 + i*dt
        x, v = xs[i-1], vs[i-1]
        k1v, k1a = duffing_step(x, v, t)
        k2v, k2a = duffing_step(x+0.5*dt*k1v, v+0.5*dt*k1a, t+0.5*dt)
        k3v, k3a = duffing_step(x+0.5*dt*k2v, v+0.5*dt*k2a, t+0.5*dt)
        k4v, k4a = duffing_step(x+dt*k3v, v+dt*k3a, t+dt)
        xs[i] = x + dt/6*(k1v+2*k2v+2*k3v+k4v) + sigma_p*rng.normal()
        vs[i] = v + dt/6*(k1a+2*k2a+2*k3a+k4a) + sigma_p*rng.normal()
    return xs, vs

def make_dict(xs, vs, t, level):
    if level == 1:
        return np.column_stack([xs, vs])
    elif level == 2:
        return np.column_stack([xs, vs, xs**2, vs**2, xs*vs])
    elif level == 3:
        return np.column_stack([xs, vs, xs**2, vs**2, xs*vs, xs**3, vs**3, xs**2*vs, xs*vs**2])
    elif level == 4:
        return np.column_stack([xs, vs, xs**2, vs**2, xs*vs, xs**3, vs**3,
                               xs**2*vs, xs*vs**2, np.sin(t*omega), np.cos(t*omega),
                               xs*np.sin(t*omega), xs*np.cos(t*omega)])
    else:
        raise ValueError(f"level {level} not defined")

def edmd_koopman(Phi, n_steps, dt):
    X = Phi[:-1].T
    Y = Phi[1:].T
    K = Y @ np.linalg.pinv(X)
    return K

def MI_linear(K, Sigma, n):
    d = K.shape[0]
    Kn = np.linalg.matrix_power(K, n)
    M = sum(np.linalg.matrix_power(K,i) @ Sigma @ np.linalg.matrix_power(K,i).T for i in range(n))
    P = Kn @ Sigma @ Kn.T
    A_mat = np.eye(d) + np.linalg.solve(M, P)
    return 0.5 * np.linalg.slogdet(A_mat)[1]

def MI_empirical_from_data(Phi_past, Phi_fut):
    d = Phi_past.shape[1]
    C_past = np.cov(Phi_past, rowvar=False)
    C_fut = np.cov(Phi_fut, rowvar=False)
    C_cross = Phi_past.T @ Phi_fut / len(Phi_past)
    try:
        invCp = np.linalg.inv(C_past)
        invCf = np.linalg.inv(C_fut)
        A_mat = np.eye(d) - C_cross.T @ invCf @ C_cross @ invCp
        ev = np.linalg.eigvalsh(A_mat)
        ev = np.clip(ev, 1e-10, None)
        return -0.5 * np.sum(np.log(ev))
    except:
        return np.nan

print()
print("Simulating Duffing oscillator (long trajectory)...")
n_total = int(T_total / dt)
xs, vs = simulate(n_total, 0.5, 0.0)
t_arr = np.arange(n_total) * dt
xs = xs[int(burn_in/dt):]
vs = vs[int(burn_in/dt):]
t_arr = t_arr[int(burn_in/dt):]
print(f"  Burned in {burn_in}s, remaining {len(xs)} steps")

n_pred = 20
results = []

for level in [1, 2, 3, 4]:
    Phi = make_dict(xs, vs, t_arr, level)
    # Standardize columns
    Phi_mean = Phi.mean(axis=0)
    Phi_std = Phi.std(axis=0) + 1e-10
    Phi_z = (Phi - Phi_mean) / Phi_std

    K = edmd_koopman(Phi_z, len(Phi_z), dt)
    eigvals = np.linalg.eigvals(K)
    spec_radius = np.max(np.abs(eigvals))

    d = Phi_z.shape[1]
    Sigma = np.cov(np.diff(Phi_z, axis=0).T) * 0.5  # approx noise covariance

    # I_lin(K) - the "linearized information"
    I_lin = MI_linear(K, np.eye(d) * 1e-4, n_pred)

    # I_emp - the "true empirical information"
    Phi_past = Phi_z[:-n_pred]
    Phi_fut = Phi_z[n_pred:]
    I_emp = MI_empirical_from_data(Phi_past, Phi_fut)

    # Model-form error
    model_err = abs(I_lin - I_emp) if not np.isnan(I_emp) else np.nan

    # Estimation error: split trajectory, fit on each half
    half = len(Phi_z) // 2
    K1 = edmd_koopman(Phi_z[:half], half, dt)
    K2 = edmd_koopman(Phi_z[half:], len(Phi_z)-half, dt)
    I1 = MI_linear(K1, np.eye(d)*1e-4, n_pred)
    I2 = MI_linear(K2, np.eye(d)*1e-4, n_pred)
    est_err = abs(I1 - I2) / 2

    ratio = model_err / max(est_err, 1e-10)
    results.append((level, d, spec_radius, I_emp, I_lin, model_err, est_err, ratio))
    print(f"  Level {level} (dim={d:2d}): spec_rad={spec_radius:.4f}  "
          f"I_emp={I_emp:.4f}  I_lin={I_lin:.4f}  "
          f"model_err={model_err:.4f}  est_err={est_err:.4f}  ratio={ratio:.1f}x")

print()
print("="*80)
print("VERDICT:")
if len(results) >= 2:
    r1 = results[0][-1]
    r_last = results[-1][-1]
    print(f"  Ratio at dim=2 (minimal): {r1:.1f}x")
    print(f"  Ratio at dim=12 (rich):   {r_last:.1f}x")
    print(f"  Trend: {'DECREASING' if r_last < r1 else 'NOT DECREASING'}")
    print()
    if r_last < r1 * 0.5:
        print("  => Dictionary upgrade HELPS. S2 recovers as PRIMARY error certificate.")
        print("     The 'model-form error >> estimation error' finding is fixable.")
    elif r_last < r1:
        print("  => Partial improvement. S2 is useful but still bounded by form error.")
        print("     The combined S2+S3 narrative is justified.")
    else:
        print("  => Dictionary DOES NOT help. Linearization itself is the bottleneck.")
        print("     S3 (correction) is ESSENTIAL. S2 alone only certifies the minor term.")
print("="*80)
