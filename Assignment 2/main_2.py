from visualization_2 import *
from orbital_mechanics_2 import *

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from scipy.integrate import solve_ivp

data_log = {
    "time": [],
    "theta": [],
    "R_pqw_i": [],
    "r_i": [],
    "v_i": [],
    "a_i": [],
    "w_i_io": [],
    "w_i_io_dot": [],
}
mu = 398600.4418  # Gravitational parameter (km^3/s^2)
earth_radius = 6378  # Earth radius (km)


def satellite_dynamic_loop(t, state, params, t_0=0):
    """
    Compute the dynamics of the satellite, updating the true anomaly and logging data.

    :param t: Current time (s)
    :param state: Current state vector [theta]
    :param params: Dictionary of orbital parameters
    :param t_0: Initial time (s)
    :return: State derivative vector and a data log entry
    """
    # Extracting the state vector
    theta = state[0]

    # Orbital parameters
    ra = params["ra"]  # Apogee radius (km)
    rp = params["rp"]  # Perigee radius (km)
    omega = params["omega"]  # Argument of perigee (rad)
    Omega = params["Omega"]  # Right ascension of ascending node (rad)
    i = params["i"]  # Inclination (rad)

    # Calulate semi-major axis, eccentricity and mean motion
    a = calculate_semimajor_axis(ra, rp)
    e = calculate_eccentricity(ra, rp)
    n = calculate_mean_motion(a, mu)

    # Calculate derivatives
    theta_dot = calculate_true_anomaly_derivative(e, theta, n)

    # Rotation matrix from PQW to inertial frame
    R_pqw_i = calculate_rotation_matrix_from_inertial_to_pqw(omega, Omega, i)

    # Calculate position in PQW frame
    E = calculate_eccentric_anomaly(n * (t - t_0), e)
    r_pqw = calculate_radius_vector_in_pqw(a, e, E)
    r_i = calculate_radius_vector_in_inertial(r_pqw, R_pqw_i)
    r_mag = np.linalg.norm(r_i)
    v_pqw = calculate_velocity_vector_in_pqw(a, e, n, r_mag, E)
    a_pqw = calculate_acceleration_vector_in_pqw(a, e, n, r_mag, E)

    # Convert to inertial frame
    v_i = calculate_velocity_vector_in_inertial(v_pqw, R_pqw_i)
    a_i = calculate_acceleration_vector_in_inertial(a_pqw, R_pqw_i)

    # Calculate angular velocity and acceleration of orbit relative to inertial frame
    w_i_io = (
        calculate_angular_velocity_of_orbit_relative_to_inertial_referenced_in_inertial(
            r_i, v_i
        )
    )
    w_i_io_dot = calculate_angular_acceleration_of_orbit_relative_to_inertial_referenced_in_inertial(
        r_i, v_i, a_i
    )

    # Logging data
    data_entry = {
        "time": t,
        "theta": theta,
        "R_pqw_i": R_pqw_i,
        "r_i": r_i,
        "v_i": v_i,
        "a_i": a_i,
        "w_i_io": w_i_io,
        "w_i_io_dot": w_i_io_dot,
    }

    # Populate the state vector derivative
    state_dot = np.zeros_like(state)
    state_dot[0] = theta_dot

    return state_dot, data_entry


def satellite_dynamic_loop_wrapper(t, state, params):
    """
    Wrapper function for satellite dynamics loop.

    :param t: Current time (s)
    :param state: Current state vector
    :param params: Orbital parameters
    :return: State derivative vector
    """
    state_dot, _ = satellite_dynamic_loop(t, state, params)
    return state_dot


# Orbital parameters for testing
params = {
    "ra": earth_radius + 10000,  # Apogee radius (km)
    "rp": earth_radius + 400,  # Perigee radius (km)
    "omega": np.radians(0),  # Argument of perigee (rad)
    "Omega": np.radians(0),  # Right ascension of ascending node (rad)
    "i": np.radians(90),  # Inclination (rad)
}

# Initial conditions
initial_state = np.array([0.0])  # Initial true anomaly (rad)
t_span = [0, 25000]  # Time span for one day (s)
n_steps = 1000  # Number of steps

# Solve the system
result = solve_ivp(
    satellite_dynamic_loop_wrapper,
    t_span,
    initial_state,
    method="RK45",
    args=(params,),
    t_eval=np.linspace(t_span[0], t_span[1], n_steps),
)

# Extract the results
t = result.t
state_vectors = result.y

for i in range(len(t)):
    time = t[i]
    state = state_vectors[:, i]
    _, log_entry = satellite_dynamic_loop(time, state, params)

    data_log["time"].append(log_entry["time"])
    data_log["theta"].append(log_entry["theta"])
    data_log["R_pqw_i"].append(log_entry["R_pqw_i"])
    data_log["r_i"].append(log_entry["r_i"])
    data_log["v_i"].append(log_entry["v_i"])
    data_log["a_i"].append(log_entry["a_i"])
    data_log["w_i_io"].append(log_entry["w_i_io"])
    data_log["w_i_io_dot"].append(log_entry["w_i_io_dot"])

# Convert data log to arrays
r_i = np.array(data_log["r_i"])
v_i = np.array(data_log["v_i"])
time = np.array(data_log["time"])

# Extract x, y, z components for position and velocity
x, y, z = r_i[:, 0], r_i[:, 1], r_i[:, 2]
vx, vy, vz = v_i[:, 0], v_i[:, 1], v_i[:, 2]

# Plot the orbit
plt.plot(t, state_vectors[0, :], label=r"$\theta$")
plt.xlabel(r"Time ($s$)")
plt.ylabel(r"True Anomaly ($rad$)")
plt.title("True Anomaly vs Time")
plt.grid("on")
plt.legend()
plt.show()

# Plot the position vectors (x, y, z components)
plt.figure(figsize=(10, 6))
plt.plot(t, x, label=r"Position $x$")
plt.plot(t, y, label=r"Position $y$")
plt.plot(t, z, label=r"Position $z$")
plt.xlabel(r"Time ($s$)")
plt.ylabel(r"Position ($km$)")
plt.title("Position Vector Components Over Time")
plt.grid(True)
plt.legend()
plt.show()

# Plot the velocity vectors (vx, vy, vz components)
plt.figure(figsize=(10, 6))
plt.plot(t, vx, label=r"Velocity $x$")
plt.plot(t, vy, label=r"Velocity $y$")
plt.plot(t, vz, label=r"Velocity $z$")
plt.xlabel(r"Time ($s$)")
plt.ylabel(r"Velocity ($km/s$)")
plt.title("Velocity Vector Components Over Time")
plt.grid(True)
plt.legend()
plt.show()

# Create a 3D figure
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection="3d")
ax.plot(x, y, z, label="Orbit")
ax.scatter(0, 0, 0, color="orange", s=100, label="Earth center")
ax.set_xlabel(r"$X$ (km)")
ax.set_ylabel(r"$Y$ (km)")
ax.set_zlabel(r"$Z$ (km)")
ax.set_box_aspect([1, 1, 1])
ax.legend()
ax.set_title("Orbital Trajectory in 3D")
plt.show()

# Animation of the satellite orbit
animate_satellite(t, data_log)
