# Units: normalized so q = m = 1. B0 = 1 sets omega_c = 1, T_c = 2*pi.

import numpy as np
import matplotlib.pyplot as plt

def rk4_step(f, state, t, dt):
    k1 = f(state, t)
    k2 = f(state + 0.5 * dt * k1, t + 0.5 * dt)
    k3 = f(state + 0.5 * dt * k2, t + 0.5 * dt)
    k4 = f(state + dt * k3, t + dt)
    return state + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

def rk4_integrate(f, state0, t0, tf, dt):
    n_steps = int(round((tf - t0) / dt))
    states = np.zeros((n_steps + 1, len(state0)))
    times = np.zeros(n_steps + 1)
    states[0] = state0
    times[0] = t0
    s, t = state0.copy(), t0
    for i in range(1, n_steps + 1):
        s = rk4_step(f, s, t, dt)
        t += dt
        states[i] = s
        times[i] = t
    return times, states

def make_derivatives(E_field, B_field, q=1.0, m=1.0):
    """
    E_field, B_field: functions of (x, y, z) -> np.array([Ex,Ey,Ez]) / [Bx,By,Bz]
    Returns a derivatives(state, t) function for use with rk4_integrate.
    """
    def derivatives(state, t):
        x, y, z, vx, vy, vz = state
        v = np.array([vx, vy, vz])
        E = E_field(x, y, z)
        B = B_field(x, y, z)
        a = (q / m) * (E + np.cross(v, B))
        return np.array([vx, vy, vz, a[0], a[1], a[2]])
    return derivatives


def part1_uniform_B():
    print("PART 1: Uniform B field")

    q, m, B0 = 1.0, 1.0, 1.0
    omega_c = q * B0 / m
    T_c = 2 * np.pi / abs(omega_c)

    B_field = lambda x, y, z: np.array([0.0, 0.0, B0])
    E_field = lambda x, y, z: np.array([0.0, 0.0, 0.0])
    deriv = make_derivatives(E_field, B_field, q, m)

    # v_perp = 1.0, v_parallel = 0.5 -> helix
    state0 = np.array([1.0, 0.0, 0.0, 0.0, 1.0, 0.5])
    dt = T_c / 200
    t_final = 5 * T_c
    times, states = rk4_integrate(deriv, state0, 0.0, t_final, dt)

    x, y, z = states[:, 0], states[:, 1], states[:, 2]
    vx, vy, vz = states[:, 3], states[:, 4], states[:, 5]
    speed = np.sqrt(vx**2 + vy**2 + vz**2)

    # Check 1: speed (KE) conservation — Lorentz force does no work
    speed_drift = (speed.max() - speed.min()) / speed[0]
    print(f"Speed conservation: |v| ranges {speed.min():.6f} to {speed.max():.6f} "
          f"(relative drift {speed_drift:.2e})")

    # Check 2: gyroradius
    v_perp0 = np.hypot(state0[3], state0[4])
    r_L_theory = v_perp0 / omega_c
    r_numeric = np.sqrt((x - x.mean())**2 + (y - y.mean())**2).mean()
    r_from_extent = (x.max() - x.min()) / 2
    print(f"Gyroradius: theory r_L = v_perp/omega_c = {r_L_theory:.6f}, "
          f"numeric (from x-extent) = {r_from_extent:.6f}")

    # Check 3: period, via zero-crossings of vy (or FFT)
    sign_changes = np.where(np.diff(np.sign(vy)) > 0)[0]
    if len(sign_changes) >= 2:
        # linear-interpolate crossing times
        crossing_times = []
        for i in sign_changes:
            t1, t2 = times[i], times[i + 1]
            y1, y2 = vy[i], vy[i + 1]
            t_cross = t1 - y1 * (t2 - t1) / (y2 - y1)
            crossing_times.append(t_cross)
        periods = np.diff(crossing_times)
        T_numeric = np.mean(periods)
        print(f"Cyclotron period: theory T_c = 2*pi*m/(qB0) = {T_c:.6f}, "
              f"numeric (from vy zero-crossings) = {T_numeric:.6f}, "
              f"rel. error = {abs(T_numeric - T_c)/T_c:.2e}")

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    axes[0].plot(x, y)
    axes[0].set_xlabel("x"); axes[0].set_ylabel("y")
    axes[0].set_title("x-y projection")
    axes[0].set_aspect("equal")

    axes[1].plot(times, z)
    axes[1].set_xlabel("t"); axes[1].set_ylabel("z")
    axes[1].set_title("z(t) (slope = v_z0)")

    axes[2].plot(times, speed)
    axes[2].set_xlabel("t"); axes[2].set_ylabel("|v|")
    axes[2].set_title("Speed vs t")
    axes[2].set_ylim(speed[0] - 0.01, speed[0] + 0.01)

    plt.tight_layout()
    plt.show()
    return omega_c, T_c


def part2_ExB_drift():
    print("PART 2: Uniform E + B fields -> E x B drift")

    q, m, B0, E0 = 1.0, 1.0, 1.0, 0.2
    omega_c = q * B0 / m
    T_c = 2 * np.pi / abs(omega_c)

    B_field = lambda x, y, z: np.array([0.0, 0.0, B0])
    E_field = lambda x, y, z: np.array([0.0, E0, 0.0])
    deriv = make_derivatives(E_field, B_field, q, m)

    # Analytic drift velocity: v_drift = (E x B) / B^2
    E_vec, B_vec = np.array([0, E0, 0]), np.array([0, 0, B0])
    v_drift_theory = np.cross(E_vec, B_vec) / np.dot(B_vec, B_vec)
    print(f"Theory drift velocity (E x B)/B^2 = {v_drift_theory}")

    state0 = np.array([0.0, 0.0, 0.0, 0.0, 1.0, 0.0])
    dt = T_c / 200
    n_periods = 20
    t_final = n_periods * T_c
    times, states = rk4_integrate(deriv, state0, 0.0, t_final, dt)
    x, y, z = states[:, 0], states[:, 1], states[:, 2]
    vx, vy, vz = states[:, 3], states[:, 4], states[:, 5]

    # Numeric drift: average velocity over an integer number of gyro-periods
    v_drift_numeric = np.array([vx.mean(), vy.mean(), vz.mean()])
    print(f"Numeric drift velocity (time-averaged v) = {v_drift_numeric}")
    err = np.linalg.norm(v_drift_numeric - v_drift_theory)
    print(f"Absolute error: {err:.2e}")

    # Guiding center should move in a straight line along x at v_drift
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    axes[0].plot(x, y)
    axes[0].set_xlabel("x"); axes[0].set_ylabel("y")
    axes[0].set_title("Trapezoidal drift path (cycloid, E x B along x)")
    axes[0].set_aspect("equal")

    axes[1].plot(times, vx, label="vx(t)")
    axes[1].axhline(v_drift_theory[0], color="k", ls="--",
                     label=f"theory drift vx = {v_drift_theory[0]:.3f}")
    axes[1].set_xlabel("t"); axes[1].set_ylabel("vx")
    axes[1].set_title("vx oscillates about the drift velocity")
    axes[1].legend()

    plt.tight_layout()
    plt.show()


def mirror_B_field(x, y, z, B0=1.0, Lz=5.0):
    """
    Simple magnetic bottle: Bz increases away from z=0 (mirror points),
    Br generated self-consistently from div(B)=0 to lowest order in r.
    Bz(r=0, z) = B0 * (1 + (z/Lz)^2)
    """
    r = np.hypot(x, y)
    Bz = B0 * (1 + (z / Lz) ** 2)
    dBz_dz = B0 * 2 * z / Lz ** 2
    Br = -0.5 * r * dBz_dz
    if r > 1e-12:
        Bx = Br * x / r
        By = Br * y / r
    else:
        Bx = By = 0.0
    return np.array([Bx, By, Bz])


def part3_magnetic_mirror():
    print("PART 3: Magnetic mirror / magnetic bottle")

    q, m, B0, Lz = 1.0, 1.0, 1.0, 5.0
    omega_c0 = q * B0 / m
    T_c0 = 2 * np.pi / abs(omega_c0)

    E_field = lambda x, y, z: np.array([0.0, 0.0, 0.0])
    B_field = lambda x, y, z: mirror_B_field(x, y, z, B0, Lz)
    deriv = make_derivatives(E_field, B_field, q, m)

    # z=0 (weakest field point) with mixed perpendicular/parallel v
    v_perp0, v_par0 = 1.0, 0.3
    state0 = np.array([1.0, 0.0, 0.0, 0.0, v_perp0, v_par0])

    dt = T_c0 / 200
    t_final = 60 * T_c0
    times, states = rk4_integrate(deriv, state0, 0.0, t_final, dt)
    x, y, z = states[:, 0], states[:, 1], states[:, 2]
    vx, vy, vz = states[:, 3], states[:, 4], states[:, 5]

    v_perp = np.sqrt(vx**2 + vy**2 + vz**2 - vz**2)  # placeholder, fixed below
    speed2 = vx**2 + vy**2 + vz**2
    v_perp2 = speed2 - vz**2
    v_perp = np.sqrt(np.maximum(v_perp2, 0))

    # local |B| at each point (for mu = v_perp^2 / (2B), m=1)
    B_mag = np.array([np.linalg.norm(mirror_B_field(x[i], y[i], z[i], B0, Lz))
                       for i in range(len(x))])
    mu = v_perp**2 / (2 * B_mag)

    print(f"Total speed conservation: min={np.sqrt(speed2).min():.6f}, "
          f"max={np.sqrt(speed2).max():.6f}")
    print(f"Adiabatic invariant mu = v_perp^2/(2B): "
          f"mean={mu.mean():.6f}, std={mu.std():.6f}, "
          f"relative fluctuation={mu.std()/mu.mean():.2e}")

    # predicted turning point (where v_parallel -> 0): B(z_turn) = B0_start * v0^2/v_perp0^2
    v0 = np.hypot(v_perp0, v_par0)
    B_start = np.linalg.norm(mirror_B_field(*state0[:3], B0, Lz))
    B_turn = B_start * (v0 / v_perp0) ** 2
    # invert Bz(0,z) = B0*(1+(z/Lz)^2) = B_turn
    z_turn_theory = Lz * np.sqrt(max(B_turn / B0 - 1, 0))
    z_max_numeric = np.abs(z).max()
    print(f"Predicted mirror (turning) point: z_turn = {z_turn_theory:.4f}")
    print(f"Numeric max |z| reached: {z_max_numeric:.4f}")

    reflected = np.any(np.diff(np.sign(vz)) != 0)
    print(f"Particle reflected (vz changed sign): {reflected}")

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    axes[0].plot(times, z)
    axes[0].axhline(z_turn_theory, color="k", ls="--", label="theory turning pt")
    axes[0].axhline(-z_turn_theory, color="k", ls="--")
    axes[0].set_xlabel("t"); axes[0].set_ylabel("z")
    axes[0].set_title("z(t) — bounce between mirror points")
    axes[0].legend()

    axes[1].plot(times, mu)
    axes[1].set_xlabel("t"); axes[1].set_ylabel("mu = v_perp^2 / (2B)")
    axes[1].set_title("Adiabatic invariant")

    r = np.hypot(x, y)
    axes[2].plot(z, r)
    axes[2].set_xlabel("z"); axes[2].set_ylabel("r = sqrt(x^2+y^2)")
    axes[2].set_title("Guiding-center path in (z, r)")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    part1_uniform_B()
    part2_ExB_drift()
    part3_magnetic_mirror()
