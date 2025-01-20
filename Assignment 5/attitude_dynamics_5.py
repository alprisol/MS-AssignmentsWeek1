import numpy as np


def S(w):
    """
    Creates the cross-product operator matrix for a vector w.

    Parameters:
    w (ndarray): A 3-element vector.

    Returns:
    ndarray: A 3x3 skew-symmetric matrix.
    """
    return np.array([[0, -w[2], w[1]], [w[2], 0, -w[0]], [-w[1], w[0], 0]])


def T(q):
    """
    Creates the transformation matrix for a quaternion q.

    Parameters:
    q (ndarray): A 4-element quaternion [eta, epsilon1, epsilon2, epsilon3], where
                 eta is the scalar part, and epsilon is the vector part.

    Returns:
    ndarray: A 4x4 transformation matrix for quaternion multiplication.
    """
    eta = q[0]
    epsilon = np.array([q[1], q[2], q[3]])
    I = np.eye(3)

    top_left = np.array([[eta]])
    top_right = -epsilon.reshape(1, 3)
    bottom_left = epsilon.reshape(3, 1)
    bottom_right = eta * I + S(epsilon)

    return np.block([[top_left, top_right], [bottom_left, bottom_right]])


def calculate_rotation_matrix_from_quaternion(q_ab):
    """
    Calculates the rotation matrix corresponding to a quaternion q_ab.

    Parameters:
    q_ab (ndarray): A 4-element quaternion [eta, epsilon1, epsilon2, epsilon3].

    Returns:
    ndarray: A 3x3 rotation matrix.

    The rotation matrix R is calculated using the formula:
    R = I + 2 * eta * S(epsilon) + 2 * S(epsilon) @ S(epsilon)
    where:
    - I is the 3x3 identity matrix
    - eta is the scalar part of the quaternion
    - epsilon is the vector part of the quaternion
    - S(epsilon) is the skew-symmetric matrix of epsilon
    """
    eta = q_ab[0]
    epsilon = np.array([q_ab[1], q_ab[2], q_ab[3]])
    S_epsilon = S(epsilon)

    R_b_a = np.eye(3) + 2 * eta * S_epsilon + 2 * np.dot(S_epsilon, S_epsilon)

    return R_b_a


def calculate_quaternion_from_rotation_matrix(R):
    """
    Calculates the quaternion corresponding to a given rotation matrix.

    Parameters:
    R (ndarray): A 3x3 rotation matrix.

    Returns:
    ndarray: A 4-element quaternion [eta, epsilon1, epsilon2, epsilon3],
             where:
             - eta is the scalar part of the quaternion
             - epsilon1, epsilon2, epsilon3 are the vector components.
    """
    # Ensure R is a proper rotation matrix
    if not (
        np.allclose(np.dot(R.T, R), np.eye(3)) and np.isclose(np.linalg.det(R), 1.0)
    ):
        raise ValueError("Input matrix is not a valid rotation matrix.")

    # Trace of the matrix
    trace = np.trace(R)

    # Initialize quaternion
    q = np.zeros(4)

    if trace > 0:
        # Case 1: Positive trace
        S = 2.0 * np.sqrt(1.0 + trace)
        q[0] = 0.25 * S  # eta
        q[1] = (R[2, 1] - R[1, 2]) / S  # epsilon1
        q[2] = (R[0, 2] - R[2, 0]) / S  # epsilon2
        q[3] = (R[1, 0] - R[0, 1]) / S  # epsilon3
    elif (R[0, 0] > R[1, 1]) and (R[0, 0] > R[2, 2]):
        # Case 2: R[0,0] is the largest diagonal element
        S = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
        q[0] = (R[2, 1] - R[1, 2]) / S  # eta
        q[1] = 0.25 * S  # epsilon1
        q[2] = (R[0, 1] + R[1, 0]) / S  # epsilon2
        q[3] = (R[0, 2] + R[2, 0]) / S  # epsilon3
    elif R[1, 1] > R[2, 2]:
        # Case 3: R[1,1] is the largest diagonal element
        S = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
        q[0] = (R[0, 2] - R[2, 0]) / S  # eta
        q[1] = (R[0, 1] + R[1, 0]) / S  # epsilon1
        q[2] = 0.25 * S  # epsilon2
        q[3] = (R[1, 2] + R[2, 1]) / S  # epsilon3
    else:
        # Case 4: R[2,2] is the largest diagonal element
        S = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
        q[0] = (R[1, 0] - R[0, 1]) / S  # eta
        q[1] = (R[0, 2] + R[2, 0]) / S  # epsilon1
        q[2] = (R[1, 2] + R[2, 1]) / S  # epsilon2
        q[3] = 0.25 * S  # epsilon3

    return q


def calculate_inverse_quaternion(q_ab):
    """
    Calculates the inverse (conjugate) of a quaternion q_ab.

    Quaternion conjugation involves negating the vector part of the quaternion.
    If q_ab = [eta, epsilon1, epsilon2, epsilon3], then the conjugate q_ab_c is
    given by [eta, -epsilon1, -epsilon2, -epsilon3].

    Parameters:
    q_ab_c (ndarray): A 4-element quaternion [eta, epsilon1, epsilon2, epsilon3].

    Returns:
    ndarray: A 4-element quaternion representing the inverse of q_ab.
    """
    q_ab_c = q_ab.copy()
    q_ab_c[1:] = -q_ab[1:]

    return q_ab_c


def quaternion_kinematics(q_ob, w_ob_b):
    """
    Calculates the kinematic equation for quaternion q_ob given angular velocity w_ob_b.

    The quaternion derivative q_dot is calculated using the formula:
    q_dot = 0.5 * T(q_ob) * w_quaternion
    where:
    - T(q_ob) is the transformation matrix for quaternion q_ob
    - w_quaternion is the angular velocity vector w_ob_b represented as a quaternion [0, w_ob_b1, w_ob_b2, w_ob_b3]

    Parameters:
    q_ob (ndarray): A 4-element quaternion [eta, epsilon1, epsilon2, epsilon3].
    w_ob_b (ndarray): A 3-element angular velocity vector of the body frame relative to the orbit frame.

    Returns:
    ndarray: A 4-element quaternion derivative (q_dot).
    """
    w_quaternion = np.array([0, w_ob_b[0], w_ob_b[1], w_ob_b[2]])
    q_dot = 0.5 * np.dot(T(q_ob), w_quaternion)

    return q_dot


def attitude_dyanamics(J, q_ob, w_ob_b, w_io_i, w_io_i_dot, R_i_o, tau_a_b, tau_p_b):
    """
    Computes the time derivative of the body angular velocity (body wrt orbit),
    expressed in the body frame, using the provided attitude-dynamics equation.

    Parameters
    ----------
    J          : ndarray of shape (3,3)
                 Inertia matrix in the body frame.
    q_ob       : ndarray of shape (4,)
                 Quaternion [eta, eps1, eps2, eps3] describing body (b)
                 relative to orbit (o).
    w_ob_b     : ndarray of shape (3,)
                 Angular velocity of body wrt orbit, expressed in body.
                 (i.e. omega_{o,b}^b)
    w_io_i     : ndarray of shape (3,)
                 Angular velocity of orbit wrt inertial, expressed in inertial.
                 (i.e. omega_{i,o}^i)
    w_io_i_dot : ndarray of shape (3,)
                 Time derivative of w_io_i (if needed).
    R_i_o      : ndarray of shape (3,3)
                 Rotation matrix to inertial frame (i) from orbit frame (o).
                 That is, v_i = R_i_o * v_o.
    tau_a_b    : ndarray of shape (3,)
                 Actuator torque in body.
    tau_p_b    : ndarray of shape (3,)
                 Disturbance (or other) torque in body.

    Returns
    -------
    w_dot_ob_b : ndarray of shape (3,)
                 Time derivative of w_ob_b, expressed in the body frame.
    """

    # 1) Rotation from orbit->body using q_ob
    R_b_o = calculate_rotation_matrix_from_quaternion(q_ob)

    # 2) Rotation from inertial->orbit is R_o_i = (R_i_o)^T
    R_o_i = R_i_o.T

    # 3) Combine them to get inertial->body
    #    If v_i is in inertial, then v_b = R_b_o * v_o,
    #    and v_o = R_o_i * v_i, so v_b = R_b_o * R_o_i * v_i = R_b_i * v_i.
    R_b_i = R_b_o @ R_o_i

    #   w_ob_b = w_ib_b - R_b_i w_io_i
    # =>  w_ib_b = w_ob_b + R_b_i w_io_i
    #
    w_ib_b = w_ob_b + R_b_i @ w_io_i

    # Now form the right-hand side of:
    #
    #   J * w_dot_ob_b
    #       = - S(w_ib_b) * J * w_ib_b
    #         + tau_a_b
    #         + tau_p_b
    #         + J * S(w_ib_b) * (R_b_i * w_io_i)
    #         - J * (R_b_i * w_io_i_dot)
    #
    # Then solve for w_dot_ob_b = J^{-1} [ ... ].

    term1 = -S(w_ib_b) @ J @ w_ib_b
    term2 = tau_a_b + tau_p_b
    term3 = J @ S(w_ib_b) @ (R_b_i @ w_io_i)
    term4 = J @ (R_b_i @ w_io_i_dot)

    rhs = term1 + term2 + term3 - term4
    w_dot_ob_b = np.linalg.inv(J) @ rhs

    return w_dot_ob_b


def calculate_euler_angles_from_quaternion(q_ab, degrees=True):
    """
    Calculates the Euler angles [phi, theta, psi] from a given quaternion [q0, q1, q2, q3].

    Parameters
    ----------
    q_ab : list or tuple of float
        The quaternion [q0, q1, q2, q3], with q0 as the scalar part.
    degrees : bool, optional
        If True, the output angles are in degrees. If False (default), the angles are in radians.

    Returns
    -------
    list of float
        A list [phi, theta, psi], where:
          phi   = roll
          theta = pitch
          psi   = yaw
    """
    # Unpack the quaternion
    q0, q1, q2, q3 = q_ab

    # Roll (phi)
    phi = np.atan2(2.0 * (q0 * q1 + q2 * q3), 1.0 - 2.0 * (q1**2 + q2**2))

    # Pitch (theta)
    s = 2.0 * (q0 * q2 - q1 * q3)
    if abs(s) >= 1.0:
        # Use 90 degrees if out of domain for asin
        theta = np.copysign(np.pi / 2.0, s)
    else:
        theta = np.asin(s)

    # Yaw (psi)
    psi = np.atan2(2.0 * (q0 * q3 + q1 * q2), 1.0 - 2.0 * (q2**2 + q3**2))

    # Convert to degrees if requested
    if degrees:
        phi = np.degrees(phi)
        theta = np.degrees(theta)
        psi = np.degrees(psi)

    return np.array([phi, theta, psi])


def calculate_euler_angles_from_rotation_matrix(R, out_in_rad=True):
    """
    Calculates the Euler angles [phi, theta, psi] from a 3x3 rotation matrix R.
    The angles follow the roll-pitch-yaw (x-y-z) convention:
       - phi   = roll   about X-axis
       - theta = pitch  about Y-axis
       - psi   = yaw    about Z-axis

    Parameters
    ----------
    R : np.ndarray of shape (3, 3)
        A valid rotation matrix.

    out_in_rad : bool, optional
        If True (default), the output angles are in radians.
        If False, the output angles are converted to degrees.

    Returns
    -------
    list of float
        [phi, theta, psi] either in radians or degrees based on `out_in_rad`.
    """
    # Ensure R is a numpy array
    R = np.asarray(R)

    # Validate the shape of R
    if R.shape != (3, 3):
        raise ValueError("Input rotation matrix R must be a 3x3 matrix.")

    # Compute Roll (phi)
    phi = np.arctan2(R[2, 1], R[2, 2])

    # Compute Pitch (theta)
    s = -R[2, 0]
    if abs(s) >= 1.0:
        # Use ±90 degrees (in radians) if out of domain for asin
        theta = np.copysign(np.pi / 2.0, s)
    else:
        theta = np.arcsin(s)

    # Compute Yaw (psi)
    psi = np.arctan2(R[1, 0], R[0, 0])

    # Collect the angles
    angles = [phi, theta, psi]

    # Convert to degrees if requested
    if not out_in_rad:
        angles = np.degrees(angles)

    return angles
