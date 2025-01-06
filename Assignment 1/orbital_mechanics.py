import numpy as np

# Gravitational parameter for Earth (km^3/s^2)
MU_EARTH = 398600.0


def calculate_circular_angular_speed(r):
    """
    Calculate the angular speed for a circular orbit of radius r.
    Uses the relation V = sqrt(mu / r), and dot_varphi = V / r.

    Parameters
    ----------
    r : float
        Orbit radius (in km if MU_EARTH is in km^3/s^2).

    Returns
    -------
    float
        Angular speed dot_varphi (in rad/s).
    """
    # Orbital velocity for circular orbit
    V = np.sqrt(MU_EARTH / r)

    # Angular speed = V / r
    dot_varphi = V / r
    return dot_varphi


def calculate_satellite_position_in_circular_orbit(varphi, r):
    """
    Calculate the satellite's position in a circular orbit, given
    the current angular position varphi (in radians) and orbit radius r.

    Based on the equations:
        x = 0
        y = r * cos(varphi)
        z = r * sin(varphi)

    Parameters
    ----------
    varphi : float
        Angular position in radians.
    r : float
        Radius of the circular orbit.

    Returns
    -------
    numpy.ndarray of shape (3,)
        The satellite position [x, y, z] in the inertial frame (km if r is in km).
    """
    x = 0.0
    y = r * np.cos(varphi)
    z = r * np.sin(varphi)

    return np.array([x, y, z])


def calculate_circular_orbit_velocity(varphi, r):
    """
    Calculate the satellite's velocity vector in a circular orbit,
    given the current angular position varphi (in radians) and orbit radius r.

    The inertial-frame components are:
        dot{x} = 0
        dot{y} = -r * dot_varphi * sin(varphi)
        dot{z} =  r * dot_varphi * cos(varphi)

    Parameters
    ----------
    varphi : float
        Angular position in radians.
    r : float
        Radius of the circular orbit.

    Returns
    -------
    numpy.ndarray of shape (3,)
        The velocity vector [dot{x}, dot{y}, dot{z}] in the inertial frame (km/s if r is in km).
    """
    # Compute angular speed
    dot_varphi = calculate_circular_angular_speed(r)

    dx = 0.0
    dy = -r * dot_varphi * np.sin(varphi)
    dz = r * dot_varphi * np.cos(varphi)

    return np.array([dx, dy, dz])
