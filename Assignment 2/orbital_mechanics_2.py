import numpy as np


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
