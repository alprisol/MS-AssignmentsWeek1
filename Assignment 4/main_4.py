from visualization_4 import *
from orbital_mechanics_4 import *
from attitude_dynamics_4 import *
from controller_4 import *

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from scipy.integrate import solve_ivp

# Set global NumPy print options
np.set_printoptions(
    precision=3,  # Limit the precision to 3 decimal places
    suppress=True,  # Avoid scientific notation for small numbers
)

# Data log
data_log = {
    "time": [],  # Time (s)
    "theta": [],  # True anomaly (rad)
    "q_ob": [],  # Quaternion of body wrt orbit
    "w_ob_b": [],  # Angular velocity of body wrt orbit
    "R_pqw_i": [],  # Rotation matrix from inertial to PQW frame
    "q_ib": [],  # Quaternion of body wrt inertial
    "q_io": [],  # Quaternion of orbit wrt inertial
    "q_oi": [],  # Quaternion of inertial wrt orbit
    "R_i_o_normal": [],  # Rotation matrix to inertial from orbit
    "R_i_o_quaternion": [],  # Rotation matrix to inertial from orbit (quaternion)
    "r_i": [],  # Radius vector in inertial frame
    "v_i": [],  # Velocity vector in inertial frame
    "a_i": [],  # Acceleration vector in inertial frame
    "w_io_i": [],  # Angular velocity of orbit relative to inertial frame
    "THETA_ib": [],  # Angles between body and inertial frame
    "THETA_io": [],  # Angles between orbit and inertial frame
}

# Constants
mu = 398600.4418  # Gravitational parameter (km^3/s^2)
earth_radius = 6378  # Earth radius (km)


# Functions
def satellite_dynamic_loop(t, state, params):
    """
    Compute the dynamics of the satellite, updating the true anomaly
    and attitude states, returning both the state derivative and
    a dictionary of data for logging/analysis.

    :param t: Current time (s)
    :param state: Current state vector [theta, q_ob(0..3), w_ob_b(0..2)]
    :param params: Dictionary of parameters:
                   - ra, rp: apogee/perigee (km)
                   - omega, OMEGA, i: orbit angles (rad)
                   - mu: grav param (km^3/s^2)
                   - J:  inertia matrix (3x3)
                   - tau_p_b: disturbance torque (3,)
                   - kp, kd: gains for PD controller
                   - q_od: desired attitude quaternion of body wrt orbit
                   - w_od_d: desired angular velocity of body wrt orbit
    :return: (state_dot, data_entry)
             state_dot: time derivative of the state
             data_entry: dict logging interesting values at time t
    """
    # 1 CURRENT STATE
    theta = state[0]
    q_ob = np.array([state[1], state[2], state[3], state[4]])  # [eta, eps1..3]
    w_ob_b = np.array([state[5], state[6], state[7]])  # [wx, wy, wz]

    # 2 PARAMETERS
    # 2a) Orbital-related
    t_0 = params["t_0"]  # Initial time (s)
    ra = params["ra"]  # Apogee radius (km)
    rp = params["rp"]  # Perigee radius (km)
    omega = params["omega"]  # Arg. of perigee (rad)
    OMEGA = params["OMEGA"]  # Right ascension of ascending node (rad)
    i = params["i"]  # Inclination (rad)
    mu = params["mu"]  # Grav param (km^3/s^2)
    # 2b) Inertial-related
    J = params["J"]  # Inertia matrix (3x3) in body frame
    tau_p_b = params["tau_p_b"]  # Disturbance torque (3,)
    # 2c) Controller-related
    kp = params.get("kp", 0.0)  # Proportional gain
    kd = params.get("kd", 0.0)  # Derivative gain
    # 2d) Desired attitude
    q_od = params.get("q_od", np.array([1.0, 0.0, 0.0, 0.0]))  # Desired quaternion
    w_od_d = params.get("w_od_d", np.zeros(3))  # Desired angular velocity (body)

    # 3 ORBITAL DYNAMICS
    a = calculate_semimajor_axis(ra, rp)
    e = calculate_eccentricity(ra, rp)
    n = calculate_mean_motion(a, mu)

    # 4 TRUE ANOMALY
    theta_dot = calculate_true_anomaly_derivative(e, theta, n)

    # 5 POSITION, VELOCITY, ACCELERATION
    M = n * (t - t_0)  # mean anomaly, approx
    E = calculate_eccentric_anomaly(M, e)
    R_pqw_i = calculate_rotation_matrix_from_inertial_to_pqw(omega, OMEGA, i)
    r_pqw = calculate_radius_vector_in_pqw(a, e, E)
    r_i = calculate_radius_vector_in_inertial(r_pqw, R_pqw_i)
    r_mag = np.linalg.norm(r_i)
    v_pqw = calculate_velocity_vector_in_pqw(a, e, n, r_mag, E)
    a_pqw = calculate_acceleration_vector_in_pqw(a, e, n, r_mag, E)

    v_i = calculate_velocity_vector_in_inertial(v_pqw, R_pqw_i)
    a_i = calculate_acceleration_vector_in_inertial(a_pqw, R_pqw_i)

    # 6 GET ORBITAL ANGULAR VELOCITY
    w_io_i = (
        calculate_angular_velocity_of_orbit_relative_to_inertial_referenced_in_inertial(
            r_i, v_i
        )
    )
    w_io_i_dot = calculate_angular_acceleration_of_orbit_relative_to_inertial_referenced_in_inertial(
        r_i, v_i, a_i
    )

    # 7 COMPUTE ATTITUDE DYNAMICS
    # 7a) Compute the torque
    R_i_o = calculate_rotation_matrix_from_inertial_to_orbit(omega, theta, OMEGA, i).T
    tau_d_b = pd_attitude_controller(q_ob, w_ob_b, q_od, w_od_d, kp, kd)
    # 7b) Quaternion derivative
    q_ob_b_dot = quaternion_kinematics(q_ob, w_ob_b)
    # 7c) Body-rate derivative
    w_ob_b_dot = attitude_dyanamics(
        J, q_ob, w_ob_b, w_io_i, w_io_i_dot, R_i_o, tau_d_b, tau_p_b
    )

    # 8 COMPUTE q_io AND R_i_o_quaternion
    q_io = calculate_quaternion_from_orbital_parameters(omega, OMEGA, i, theta)
    R_i_o_quaternion = calculate_rotation_matrix_from_quaternion(q_io)

    # 9 CALCULATE q_ib AND q_oi
    q_ib = calculate_quaternion_product(q_io, q_ob)
    q_oi = calculate_inverse_quaternion(q_io)

    # 10 CALCULATE ANGLES BETWEEN FRAMES
    THETA_ib = calculate_euler_angles_from_quaternion(q_ib)
    THETA_io = calculate_euler_angles_from_quaternion(q_io)

    # 9 UPDATE THE LOGG ENTRY
    data_entry = {
        "time": t,
        "theta": theta,
        "q_ob": q_ob,
        "w_ob_b": w_ob_b,
        "R_pqw_i": R_pqw_i,
        "q_ib": q_ib,
        "q_io": q_io,
        "q_oi": q_oi,
        "R_i_o_normal": R_i_o,
        "R_i_o_quaternion": R_i_o_quaternion,
        "r_i": r_i,
        "v_i": v_i,
        "a_i": a_i,
        "w_io_i": w_io_i,
        "THETA_ib": THETA_ib,
        "THETA_io": THETA_io,
    }

    # 10) Populate the state derivative
    state_dot = np.zeros_like(state)
    state_dot[0] = theta_dot
    state_dot[1] = q_ob_b_dot[0]
    state_dot[2] = q_ob_b_dot[1]
    state_dot[3] = q_ob_b_dot[2]
    state_dot[4] = q_ob_b_dot[3]
    state_dot[5] = w_ob_b_dot[0]
    state_dot[6] = w_ob_b_dot[1]
    state_dot[7] = w_ob_b_dot[2]

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


# ------------------------------------------------------------------------------
# MAIN EXECUTION CODE
# ------------------------------------------------------------------------------
if __name__ == "__main__":

    # Orbital parameters for testing
    params = {
        "t_0": 0,  # Initial time (s)
        "ra": earth_radius + 10000,  # Apogee radius (km)
        "rp": earth_radius + 400,  # Perigee radius (km)
        "omega": np.radians(0),  # Argument of perigee (rad)
        "OMEGA": np.radians(0),  # Right ascension of ascending node (rad)
        "i": np.radians(90),  # Inclination (rad)
        "mu": mu,  # Gravitational parameter (km^3/s^2)
        "J": np.eye(3),  # Inertia matrix (3x3)
        "tau_p_b": np.zeros(3),  # Disturbance torque (3,)
        "kp": 0.1,  # Proportional gain
        "kd": 0.1,  # Derivative gain
        "q_od": np.array([0, 0, 1, 0]),  # Desired quaternion
        "w_od_d": np.array(
            [np.radians(0.0), np.radians(0.0), np.radians(0.0)]
        ),  # Desired angular velocity (body)
    }

    # Initial conditions
    theta = np.array([0.0])  # Initial true anomaly (rad)
    q_ob = np.array([0.0, 1.0, 0.0, 0.0])  # Initial quaternion
    w_ob_b = np.array([0.1, 0.2, -0.3])  # Initial angular velocity
    initial_state = np.concatenate((theta, q_ob, w_ob_b))

    # Time span and steps. Solver options.
    t_span = [0, 3600 * 4]  # Time span (s)
    n_steps = 1000  # Number of steps

    # Solve the system
    result = solve_ivp(
        satellite_dynamic_loop_wrapper,
        t_span,
        initial_state,
        method="RK45",
        args=(params,),
        t_eval=np.linspace(t_span[0], t_span[1], n_steps),
        rtol=1e-6,
        atol=1e-9,
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
        data_log["q_ib"].append(log_entry["q_ib"])
        data_log["q_io"].append(log_entry["q_io"])
        data_log["q_oi"].append(log_entry["q_oi"])
        data_log["R_i_o_normal"].append(log_entry["R_i_o_normal"])
        data_log["R_i_o_quaternion"].append(log_entry["R_i_o_quaternion"])
        data_log["r_i"].append(log_entry["r_i"])
        data_log["v_i"].append(log_entry["v_i"])
        data_log["a_i"].append(log_entry["a_i"])
        data_log["w_io_i"].append(log_entry["w_io_i"])
        data_log["q_ob"].append(log_entry["q_ob"])
        data_log["w_ob_b"].append(log_entry["w_ob_b"])
        data_log["THETA_ib"].append(log_entry["THETA_ib"])
        data_log["THETA_io"].append(log_entry["THETA_io"])

    # Convert lists to NumPy arrays for easier slicing
    time = np.array(data_log["time"])
    q_ob = np.array(data_log["q_ob"])  # Shape: (N, 4)
    w_ob_b = np.array(data_log["w_ob_b"])  # Shape: (N, 3)
    theta = np.array(data_log["theta"])  # Shape: (N,)

    # Desired values
    q_od = params["q_od"]  # Desired quaternion, shape (4,)
    w_od_d = params["w_od_d"]  # Desired angular velocity, shape (3,)

    # Gains for display
    kp = params["kp"]  # Proportional gain
    kd = params["kd"]  # Derivative gain

    # Choose static scene or animation
    static = False
    if static:
        plotter = visualize_scene(data_log, time_index=100)
        plotter.show()

    else:
        animate_satellite(
            t,
            data_log,
            f"Assignment 4/SatelliteAnimation_i{round(params['i'],2)}_omega{round(params['omega'],2)}_OMEGA{round(params['OMEGA'],2)}_q{np.round(params['q_od'],2)}_{round(t_span[1]/3600,2)}h_{n_steps}steps.gif",
        )
