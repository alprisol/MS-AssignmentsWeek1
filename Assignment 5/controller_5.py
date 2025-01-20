import numpy as np

from attitude_dynamics_5 import *


def pd_attitude_controller(
    q_ob,
    w_ob_b,
    q_od,
    w_od_d,
    kp,
    kd,
):
    """
    PD Attitude Controller

    Parameters
    ----------
    q_ob   : ndarray(4,)
        Current quaternion [eta, eps1, eps2, eps3] describing body wrt orbit.
    w_ob_b : ndarray(3,)
        Current angular velocity of the body wrt orbit, expressed in body.
    q_od   : ndarray(4,)
        Desired quaternion describing body wrt orbit.
    w_od_d : ndarray(3,)
        Desired angular velocity of the body wrt orbit (frame in which it's expressed
        may vary depending on your convention; see usage below).
    kp     : float
        Proportional gain.
    kd     : float
        Derivative gain.

    Returns
    -------
    t_db : ndarray(3,)
        The commanded torque in the body frame.
    """
    # ---------------------------------------------------------------------------
    # 1) Form the quaternion error q_{d,b}.
    #    Based on your description:
    #        q_{d,b} = q_{d,o} * q_{o,b}
    #    where q_{d,o} can be the inverse of q_od if needed,
    #    or directly q_od if that is indeed the "orbit->desired" orientation.
    # ---------------------------------------------------------------------------
    # If q_od is the "desired body wrt orbit," then the inverse (q_do) would be
    # "orbit wrt desired body," so you have to decide which you actually need.
    # The problem statement suggests q_{d,o} might be the inverse of q_od, so
    # we do that here:
    q_do = calculate_inverse_quaternion(q_od)
    q_db = T(q_do) @ q_ob  # This yields the quaternion "d->b" (error)

    # Extract the vector part epsilon_{d,b}, which we want to drive to zero.
    eps_db = q_db[1:]  # The last three components

    # ---------------------------------------------------------------------------
    # 2) Form the desired body-rate error in body frame:
    #      w_{d,b}^b = w_{o,b}^b - R_{d}^b * w_{o,d}^d
    #
    #    We must first compute R_{d}^b from the quaternion q_{d,b}.
    # ---------------------------------------------------------------------------
    R_db = calculate_rotation_matrix_from_quaternion(q_db)
    # w_od_d is presumably in the "desired" frame (d).  We'll rotate it into b:
    w_db_b = w_ob_b - (R_db @ w_od_d)

    # ---------------------------------------------------------------------------
    # 3) PD torque: tau_d^b = -k_p * eps_{d,b} - k_d * w_{d,b}^b
    # ---------------------------------------------------------------------------
    t_db = -kp * eps_db - kd * w_db_b

    return t_db
