import numpy as np
from datetime import datetime
import ppigrf

from attitude_dynamics_5 import *

# Set global NumPy print options
np.set_printoptions(
    precision=3,  # Limit the precision to 3 decimal places
    suppress=True,  # Avoid scientific notation for small numbers
)


# Constants
a_e = 6378.137  # Semi-major axis in km
b_e = 6356.725  # Semi-minor axis in km
omega_ie = 7.292115e-5  # Angular speed of Earth in rad/s
e_e = 0.0818  # Eccentricity of the reference ellipsoid


def calculate_eccentricity(ra, rp):
    """
    Calculate the orbital eccentricity.
    :param ra: Apogee radius (km)
    :param rp: Perigee radius (km)
    :return: Eccentricity (dimensionless)
    """
    return (ra - rp) / (ra + rp)


def calculate_semimajor_axis(ra, rp):
    """
    Calculate the semi-major axis.
    :param ra: Apogee radius (km)
    :param rp: Perigee radius (km)
    :return: Semi-major axis (km)
    """
    return (ra + rp) / 2


def calculate_mean_motion(a, mu):
    """
    Calculate the mean motion.
    :param a: Semi-major axis (km)
    :param mu: Standard gravitational parameter (km^3/s^2)
    :return: Mean motion (rad/s)
    """
    return np.sqrt(mu / a**3)


def calculate_orbital_period(n):
    """
    Calculate the orbital period.
    :param n: Mean motion (rad/s)
    :return: Orbital period (s)
    """
    return 2 * np.pi / n


def calculate_eccentric_anomaly(M, e, tolerance=1e-6):
    """
    Calculate the eccentric anomaly using iterative method.
    :param M: Mean anomaly (rad)
    :param e: Eccentricity (dimensionless)
    :param tolerance: Convergence threshold
    :return: Eccentric anomaly (rad)
    """
    E = M  # Initial guess
    while True:
        E_new = M + e * np.sin(E)
        if abs(E_new - E) < tolerance:
            break
        E = E_new
    return E


def calculate_true_anomaly(e, E):
    """
    Calculate the true anomaly.
    :param e: Eccentricity (dimensionless)
    :param E: Eccentric anomaly (rad)
    :return: True anomaly (rad)
    """
    return np.arccos((np.cos(E) - e) / (1 - e * np.cos(E)))


def calculate_true_anomaly_derivative(e, theta, n):
    """
    Calculate the derivative of the true anomaly.
    :param e: Eccentricity (dimensionless)
    :param theta: True anomaly (rad)
    :param n: Mean motion (rad/s)
    :return: Derivative of true anomaly (rad/s)
    """
    return n * (1 + e * np.cos(theta)) ** 2 / (1 - e**2) ** 1.5


def calculate_rotation_matrix_from_inertial_to_pqw(omega, OMEGA, i):
    """
    Calculate the rotation matrix from inertial to PQW frame.
    :param omega: Argument of perigee (rad)
    :param OMEGA: Right ascension of ascending node (rad)
    :param i: Inclination (rad)
    :return: Rotation matrix (3x3)
    """
    cos_OMEGA, sin_OMEGA = np.cos(OMEGA), np.sin(OMEGA)
    cos_omega, sin_omega = np.cos(omega), np.sin(omega)
    cos_i, sin_i = np.cos(i), np.sin(i)

    R_pqw_i = np.array(
        [
            [
                cos_omega * cos_OMEGA - sin_omega * cos_i * sin_OMEGA,
                cos_omega * sin_OMEGA + sin_omega * cos_i * cos_OMEGA,
                sin_omega * sin_i,
            ],
            [
                -sin_omega * cos_OMEGA - cos_omega * cos_i * sin_OMEGA,
                -sin_omega * sin_OMEGA + cos_omega * cos_i * cos_OMEGA,
                cos_omega * sin_i,
            ],
            [sin_i * sin_OMEGA, -sin_i * cos_OMEGA, cos_i],
        ]
    )

    return R_pqw_i


def calculate_rotation_matrix_from_inertial_to_orbit(omega, theta, OMEGA, i):
    """
    Calculate the rotation matrix from inertial to orbit frame.
    :param omega: Argument of perigee (rad)
    :param theta: True anomaly (rad)
    :param OMEGA: Right ascension of ascending node (rad)
    :param i: Inclination (rad)
    :return: Rotation matrix (3x3)
    """
    return calculate_rotation_matrix_from_inertial_to_pqw(omega + theta, OMEGA, i)


def calculate_rotation_matrix_from_orbit_to_inertial(omega, theta, OMEGA, i):
    """
    Calculate the rotation matrix from orbit frame to inertial frame
    :param omega: Argument of perigee (rad)
    :param theta: True anomaly (rad)
    :param OMEGA: Right ascension of ascending node (rad)
    :param i: Inclination (rad)
    :return: Rotation matrix (3x3)
    """
    return calculate_rotation_matrix_from_inertial_to_orbit(omega, theta, OMEGA, i).T


def calculate_angular_velocity_of_orbit_relative_to_inertial_referenced_in_inertial(
    r, v
):
    """
    Calculate angular velocity of orbit relative to inertial frame.
    :param r: Radius vector (km)
    :param v: Velocity vector (km/s)
    :return: Angular velocity vector (rad/s)
    """
    return np.cross(r, v) / np.dot(r, r)


def calculate_angular_acceleration_of_orbit_relative_to_inertial_referenced_in_inertial(
    r, v, a
):
    """
    Calculate angular acceleration of orbit relative to inertial frame.
    :param r: Radius vector (km)
    :param v: Velocity vector (km/s)
    :param a: Acceleration vector (km/s^2)
    :return: Angular acceleration vector (rad/s^2)
    """

    return (np.cross(r, a) * np.dot(r, r) - 2 * np.cross(r, v) * np.dot(v, r)) / np.dot(
        r, r
    ) ** 2


def calculate_radius_vector_in_pqw(a, e, E):
    """
    Calculate radius vector in PQW frame.
    :param a: Semi-major axis (km)
    :param e: Eccentricity (dimensionless)
    :param E: Eccentric anomaly (rad)
    :return: Radius vector in PQW frame (km)
    """

    return np.array([a * np.cos(E) - a * e, a * np.sqrt(1 - e**2) * np.sin(E), 0])


def calculate_velocity_vector_in_pqw(a, e, n, r, E):
    """
    Calculate velocity vector in PQW frame.
    :param a: Semi-major axis (km)
    :param e: Eccentricity (dimensionless)
    :param n: Mean motion (rad/s)
    :param r: Radius (km)
    :param E: Eccentric anomaly (rad)
    :return: Velocity vector in PQW frame (km/s)
    """
    return np.array(
        [-(a**2) * n / r * np.sin(E), a**2 * n / r * np.sqrt(1 - e**2) * np.cos(E), 0]
    )


def calculate_acceleration_vector_in_pqw(a, e, n, r, E):
    """
    Calculate acceleration vector in PQW frame.
    :param a: Semi-major axis (km)
    :param e: Eccentricity (dimensionless)
    :param n: Mean motion (rad/s)
    :param r: Radius (km)
    :param E: Eccentric anomaly (rad)
    :return: Acceleration vector in PQW frame (km/s^2)
    """
    return np.array(
        [
            -(a**3) * n**2 / r**2 * np.cos(E),
            -(a**3) * n**2 / r**2 * np.sqrt(1 - e**2) * np.sin(E),
            0,
        ]
    )


def calculate_radius_vector_in_inertial(r_pqw, R_pqw_i):
    """
    Calculate radius vector in inertial frame.
    :param r_pqw: Radius vector in PQW frame (km)
    :param R_pqw_i: Rotation matrix to interial frame from PQW  (3x3)
    :return: Radius vector in inertial frame (km)
    """
    return np.dot(R_pqw_i.T, r_pqw)


def calculate_velocity_vector_in_inertial(v_pqw, R_pqw_i):
    """
    Calculate velocity vector in inertial frame.
    :param v_pqw: Velocity vector in PQW frame (km/s)
    :param R_pqw_i: Rotation matrix to interial frame from PQW (3x3)
    :return: Velocity vector in inertial frame (km/s)
    """
    return np.dot(R_pqw_i.T, v_pqw)


def calculate_acceleration_vector_in_inertial(a_pqw, R_pqw_i):
    """
    Calculate acceleration vector in inertial frame.
    :param a_pqw: Acceleration vector in PQW frame (km/s^2)
    :param R_pqw_i: Rotation matrix to interial frame from PQW (3x3)
    :return: Acceleration vector in inertial frame (km/s^2)
    """
    return np.dot(R_pqw_i.T, a_pqw)


def calculate_quaternion_product(q1, q2):
    """
    Returns the quaternion product q1 ⊗ q2.

    Each quaternion is a 4-element array-like: [w, x, y, z].

    Inputs:
        q1 -- The first quaternion.
        q2 -- The second quaternion.

    Outputs:
        q -- The quaternion product
    """
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2

    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2

    return np.array([w, x, y, z], dtype=float)


def calculate_quaternion_from_orbital_parameters(omega, OMEGA, i, theta, degrees=False):
    """
    Calculate the orbit-to-inertial quaternion from the orbital parameters:
      - omega (argument of perigee)  [rad]
      - OMEGA (right ascension)      [rad]
      - i (inclination)             [rad]
      - theta (true anomaly)        [rad]

    Returns a 4-element numpy array: [w, x, y, z].
    """
    # convert to radians if needed
    if degrees:
        omega = np.radians(omega)
        OMEGA = np.radians(OMEGA)
        i = np.radians(i)
        theta = np.radians(theta)

    # 1) Quaternion for rotation by OMEGA around the Z-axis
    q_OMEGA = np.array([np.cos(OMEGA / 2.0), 0.0, 0.0, np.sin(OMEGA / 2.0)])

    # 2) Quaternion for rotation by i around the X-axis
    q_i = np.array([np.cos(i / 2.0), np.sin(i / 2.0), 0.0, 0.0])

    # 3) Quaternion for rotation by (omega + theta) around the Z-axis
    q_omega_plus_theta = np.array(
        [np.cos((omega + theta) / 2.0), 0.0, 0.0, np.sin((omega + theta) / 2.0)]
    )

    # Using the quaternion product:
    # q_i,o = q_Ω ⊗ q_i ⊗ q_(ω+θ)
    q_i_o = calculate_quaternion_product(
        q_OMEGA, calculate_quaternion_product(q_i, q_omega_plus_theta)
    )

    return q_i_o


# Function: Calculate Rotation Matrix from Inertial to ECEF
def calculate_rotation_matrix_from_inertial_to_ecef(t):

    theta = omega_ie * t
    R_i_e = np.array(
        [
            [np.cos(theta), -np.sin(theta), 0],
            [np.sin(theta), np.cos(theta), 0],
            [0, 0, 1],
        ]
    )

    R_e_i = R_i_e.T

    return R_e_i


# Function: Calculate LLA from ECEF
def calculate_lla_from_ecef(r_e, out_in_degrees=True):

    x, y, z = r_e
    p = np.sqrt(x**2 + y**2)
    mu = np.arctan((z / p) * (1 - e_e**2) ** -1)

    mu_old = 10
    while abs(mu - mu_old) > 1e-6:
        mu_old = mu
        N = a_e**2 / np.sqrt(a_e**2 * np.cos(mu) ** 2 + b_e**2 * np.sin(mu) ** 2)
        h = p / np.cos(mu) - N
        mu = np.arctan((z / p) * (1 - e_e**2 * (N / (N + h))) ** -1)

    l = np.atan2(y, x)
    if out_in_degrees:
        latitude = np.degrees(mu)
        longitude = np.degrees(l)
    altitude = h

    return latitude, longitude, altitude


# Function: Calculate Rotation Matrix from ECEF to NED
def calculate_rotation_matrix_from_ecef_to_ned(lat, lon, degrees=True):
    """
    Calculates the rotation matrix from ECEF to NED coordinates.

    Parameters:
    - lat: Latitude (in degrees or radians).
    - lon: Longitude (in degrees or radians).
    - degrees: True if input is in degrees, False if in radians.

    Returns:
    - R_NED_ECEF: 3x3 rotation matrix from ECEF to NED.
    """
    # Convert degrees to radians if necessary
    if degrees:
        lat = np.radians(lat)
        lon = np.radians(lon)

    # Precompute sine and cosine of latitude and longitude
    sin_lat, cos_lat = np.sin(lat), np.cos(lat)
    sin_lon, cos_lon = np.sin(lon), np.cos(lon)

    # Rotation matrix from NED to ECEF
    R_ECEF_NED = np.array(
        [
            [-cos_lon * sin_lat, -sin_lon, -cos_lon * cos_lat],
            [-sin_lon * sin_lat, cos_lon, -sin_lon * cos_lat],
            [cos_lat, 0, -sin_lat],
        ]
    )

    # Transpose to get ECEF to NED
    R_NED_ECEF = R_ECEF_NED.T

    return R_NED_ECEF


# Function: Calculate Rotation Matrix from NED to ENU
def calculate_rotation_matrix_from_ned_to_enu():
    R_n_u = np.array([[0, 1, 0], [1, 0, 0], [0, 0, -1]])
    return R_n_u


if __name__ == "__main__":

    omega = 20 * np.pi / 180
    OMEGA = 10 * np.pi / 180
    theta = 75 * np.pi / 180
    i = 56 * np.pi / 180

    R_o_i = calculate_rotation_matrix_from_inertial_to_orbit(omega, theta, OMEGA, i)

    q_io = calculate_quaternion_from_orbital_parameters(omega, OMEGA, i, theta)
    R_i_o_QUAT = calculate_rotation_matrix_from_quaternion(q_io)
    R_o_i_QUAT = R_i_o_QUAT.T

    print(f"Rotation Matrix from inertial to orbit (Standard):\n{R_o_i}\n")
    print(f"Quaternion from orbital parameters:\n{q_io}\n")
    print(f"Rotation Matrix from inertial to orbit (from Quaternion):\n{R_o_i_QUAT}\n")

    print("\n\n------------------------------------------------------------------\n\n")

    OMEGA = np.radians(0)
    omega = np.radians(0)
    i = np.radians(75)
    theta = np.radians(30)

    t = 30  # s

    r_o = [6420.652, 5236.678, 1111.957]
    print(f"Position in orbital frame (km): {r_o} \n")

    date = datetime(2025, 1, 10)

    R_i_o = calculate_rotation_matrix_from_orbit_to_inertial(omega, theta, OMEGA, i)
    print(f"Rotation Matrix from orbit to inertial: \n {R_i_o} \n")

    r_i = R_i_o @ r_o
    print(f"Position in ECI (km): {r_i} \n")

    R_ECEF_i = calculate_rotation_matrix_from_inertial_to_ecef(t)
    print(f"Rotation Matrix from inertial to ECEF: \n {R_ECEF_i} \n")

    r_ECEF = R_ECEF_i @ r_i
    print(f"Position in ECEF (km): {r_ECEF}\n")

    lat, lon, alt = calculate_lla_from_ecef(r_ECEF)
    print(
        f"Latitude: {round(lat,4)}º\nLongitude: {round(lon,4)}º\nAltitude: {round(alt,4)} km\n"
    )

    R_NED_ECEF = calculate_rotation_matrix_from_ecef_to_ned(lat, lon)
    print(f"Rotation Matrix from ECEF to NED: \n {R_NED_ECEF} \n")
    r_NED = R_NED_ECEF @ r_ECEF
    print(f"Position in NED: {r_NED}\n")

    R_ENU_NED = calculate_rotation_matrix_from_ned_to_enu()
    print(f"Rotation Matrix from NED to ENU: \n {R_ENU_NED} \n")
    r_ENU = R_ENU_NED @ r_NED
    print(f"Position in ENU (km): {r_ENU}\n")

    Be, Bn, Bu = ppigrf.igrf(lon, lat, alt, date)
    B_ENU = np.array([Be, Bn, Bu]) * 1e-9

    np.set_printoptions(
        precision=3,  # Limit the precision to 3 decimal places
        suppress=False,  # Scientific notation for small numbers
    )

    B_NED = R_ENU_NED.T @ B_ENU
    B_ECEF = R_NED_ECEF.T @ B_NED
    B_i = R_ECEF_i.T @ B_ECEF
    B_o = R_i_o.T @ B_i

    print(f"B in ENU frame: \n{B_ENU*1e9}\n")
    print(f"B in orbit frame: \n{B_o*1e9}\n")
