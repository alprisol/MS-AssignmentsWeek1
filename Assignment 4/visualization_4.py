import pyvista as pv
import numpy as np
import matplotlib.pyplot as plt
import vtk

from orbital_mechanics_4 import *
from attitude_dynamics_4 import *
from sensors import *

# Set global NumPy print options
np.set_printoptions(
    precision=3,  # Limit the precision to 3 decimal places
    suppress=True,  # Avoid scientific notation for small numbers
)

# Enable LaTeX rendering and importing the xcolor package
plt.rcParams["text.usetex"] = True
plt.rcParams["text.latex.preamble"] = r"\usepackage{xcolor}"

earth_rot_vel = 7.2722e-5  # rad/s


def create_reference_frame(plotter, labels, scale=1):
    """
    Creates a 3D reference frame with labeled x, y, and z axes in a PyVista plot.

    Parameters:
        plotter (pyvista.Plotter): The PyVista plotter to add the reference frame to.
        labels (list of str): List of labels for the x, y, and z axes.
        scale (float, optional): Scale factor for the size of the arrows and labels. Default is 1.

    Returns:
        dict: A dictionary containing the arrow meshes and label text objects for the reference frame.
    """

    # Create an arrow mesh for the x-axis
    x_arrow = pv.Arrow(
        start=(0, 0, 0),
        direction=(1, 0, 0),
        tip_length=0.25,
        tip_radius=0.1,
        tip_resolution=20,
        shaft_radius=0.05,
        shaft_resolution=20,
        scale=scale,
    )

    # Create an arrow mesh for the y-axis
    y_arrow = pv.Arrow(
        start=(0, 0, 0),
        direction=(0, 1, 0),
        tip_length=0.25,
        tip_radius=0.1,
        tip_resolution=20,
        shaft_radius=0.05,
        shaft_resolution=20,
        scale=scale,
    )

    # Create an arrow mesh for the z-axis
    z_arrow = pv.Arrow(
        start=(0, 0, 0),
        direction=(0, 0, 1),
        tip_length=0.25,
        tip_radius=0.1,
        tip_resolution=20,
        shaft_radius=0.05,
        shaft_resolution=20,
        scale=scale,
    )

    # Add the arrow meshes and labels to the plotter
    reference_frame_mesh = {
        "scale": scale,
        "x": plotter.add_mesh(x_arrow, color="red", show_edges=False),
        "y": plotter.add_mesh(y_arrow, color="blue", show_edges=False),
        "z": plotter.add_mesh(z_arrow, color="green", show_edges=False),
        "x_label": pv.Label(
            text=labels[0], position=np.array([1, 0, 0]) * scale, size=20
        ),
        "y_label": pv.Label(
            text=labels[1], position=np.array([0, 1, 0]) * scale, size=20
        ),
        "z_label": pv.Label(
            text=labels[2], position=np.array([0, 0, 1]) * scale, size=20
        ),
    }

    return reference_frame_mesh


def create_satellite(plotter, size=0.5):
    """
    Creates a 3D model of a satellite with a textured body, solar panels, and a scientific instrument.

    Parameters:
        plotter (pyvista.Plotter): The PyVista plotter to add the satellite to.
        size (float, optional): Scaling factor for the satellite size. Default is 0.5.

    Returns:
        dict: A dictionary containing the meshes for the satellite body, solar panels, and scientific instrument.
    """

    # Create the satellite body with a texture
    body_texture = pv.read_texture("Assignment 1/satellite_texture.png")
    body_b = pv.Box(bounds=(-size, size, -size, size, -size, size))
    body_b.texture_map_to_plane(inplace=True)

    # Create two solar panels with textures
    solar_panel_texture = pv.read_texture(
        "Assignment 1/high_quality_solar_panel_texture.png"
    )
    panel_width = 1.5 * size
    panel_length = 5 * size
    panel_thickness = 0.1 * size
    center_offset = 0.5 * size

    # Solar panel 1
    solar_panel_b_1 = pv.Box(
        bounds=(
            -panel_thickness / 2,
            panel_thickness / 2,
            -panel_width / 2,
            panel_width / 2,
            center_offset,
            panel_length,
        )
    )
    solar_panel_b_1.texture_map_to_plane(
        origin=(-size - panel_thickness / 2, 0, 0),
        point_u=(-size - panel_thickness / 2, panel_width / 2, 0),
        point_v=(-size - panel_thickness / 2, 0, panel_length / 2),
        inplace=True,
    )

    # Solar panel 2
    solar_panel_b_2 = pv.Box(
        bounds=(
            -panel_thickness / 2,
            panel_thickness / 2,
            -panel_width / 2,
            panel_width / 2,
            -center_offset,
            -panel_length,
        )
    )
    solar_panel_b_2.texture_map_to_plane(
        origin=(-size - panel_thickness / 2, 0, 0),
        point_u=(-size - panel_thickness / 2, panel_width / 2, 0),
        point_v=(-size - panel_thickness / 2, 0, panel_length / 2),
        inplace=True,
    )

    # Create a scientific instrument along the x-axis
    scientific_instrument_texture = pv.read_texture("Assignment 1/camera_texture.png")
    scientific_instrument_b = pv.Cone(
        center=(size - 0.01, 0, 0),
        direction=(-1, 0, 0),
        height=0.5 * size,
        radius=0.5 * size,
        resolution=50,
    )
    scientific_instrument_b.texture_map_to_sphere(inplace=True)

    # Add all components to the plotter
    satellite_mesh = {
        "Body": plotter.add_mesh(body_b, texture=body_texture, show_edges=True),
        "Solar Panels": [
            plotter.add_mesh(
                solar_panel_b_1, texture=solar_panel_texture, show_edges=True
            ),
            plotter.add_mesh(
                solar_panel_b_2, texture=solar_panel_texture, show_edges=True
            ),
        ],
        "Scientific Instrument": plotter.add_mesh(
            scientific_instrument_b,
            texture=scientific_instrument_texture,
            show_edges=False,
        ),
    }

    return satellite_mesh


def create_earth(plotter, radius):
    """
    Creates a 3D textured model of Earth and adds it to the PyVista plotter.

    Parameters:
        plotter (pyvista.Plotter): The PyVista plotter to add the Earth model to.
        radius (float): Radius of the Earth model.

    Returns:
        pyvista.Plotting: The mesh object of the Earth added to the plotter.
    """

    earth = pv.examples.planets.load_earth(radius=radius)
    earth_texture = pv.examples.load_globe_texture()
    earth_mesh = plotter.add_mesh(earth, texture=earth_texture, smooth_shading=True)

    return earth_mesh


def create_sensor_cone(
    plotter,
    r_i,
    R_i_b,
    raycasting_length,
    field_of_view_half_deg,
    number_of_raycasting_points,
    earth_radius,
):
    # Finding the intersection points
    intersection_points_i = calculate_intersection_points_in_inertial_frame(
        r_i,
        R_i_b,
        raycasting_length,
        field_of_view_half_deg,
        number_of_raycasting_points,
        earth_radius,
    )
    # Create triangles for the cone
    triangles = []
    for i in range(len(intersection_points_i[:, 0]) - 1):
        triangles.append([r_i, intersection_points_i[i + 1], intersection_points_i[i]])

    # Close the cone by connecting the last point to the first
    triangles.append([r_i, intersection_points_i[0], intersection_points_i[-1]])

    # Create the mesh
    cone_mesh = pv.PolyData()
    for tri in triangles:
        cone_mesh += pv.Triangle(tri)

    plotter.add_mesh(cone_mesh, opacity=0.25, color="green")

    return cone_mesh


def Rz(angle):
    """
    Creates a rotation matrix for a rotation around the z-axis.

    Parameters:
        angle (float): The rotation angle in radians.

    Returns:
        numpy.ndarray: A 3x3 rotation matrix.
    """
    return np.array(
        [
            [np.cos(angle), -np.sin(angle), 0],
            [np.sin(angle), np.cos(angle), 0],
            [0, 0, 1],
        ]
    )


def Ry(angle):
    """
    Creates a rotation matrix for a rotation around the y-axis.

    Parameters:
        angle (float): The rotation angle in radians.

    Returns:
        numpy.ndarray: A 3x3 rotation matrix.
    """
    return np.array(
        [
            [np.cos(angle), 0, np.sin(angle)],
            [0, 1, 0],
            [-np.sin(angle), 0, np.cos(angle)],
        ]
    )


def Rx(angle):
    """
    Creates a rotation matrix for a rotation around the x-axis.

    Parameters:
        angle (float): The rotation angle in radians.

    Returns:
        numpy.ndarray: A 3x3 rotation matrix.
    """
    return np.array(
        [
            [1, 0, 0],
            [0, np.cos(angle), -np.sin(angle)],
            [0, np.sin(angle), np.cos(angle)],
        ]
    )


def pyvista_rotation_matrix_from_euler_angles(orientation_euler, degrees=False):
    """
    Computes the rotation matrix from Euler angles in degrees or radians.
    Parameters:
        orientation_euler (list): List of Euler angles [phi, theta, psi] in degrees or radians.
        degrees (bool, optional): Whether the input angles are in degrees. Default is False.

    Returns:
        numpy.ndarray: The 3x3 rotation matrix.

    """

    # Pyvista rotates in the order y-x-z
    if degrees:
        phi = orientation_euler[0]
        theta = orientation_euler[1]
        psi = orientation_euler[2]
    else:
        phi = orientation_euler[0] * np.pi / 180
        theta = orientation_euler[1] * np.pi / 180
        psi = orientation_euler[2] * np.pi / 180

    R = Rz(psi).dot(Rx(phi)).dot(Ry(theta))
    return R


def update_satellite_pose(satellite_mesh, r_i, THETA_ib, degrees=True):
    """
    Update the satellite's position and orientation in the PyVista plot.
    Parameters:
        satellite_mesh (dict): Dictionary containing the satellite's mesh objects.
        r_i (list): Satellite's position in the inertial frame.
        THETA_ib (list): Euler angles [phi, theta, psi] in degrees or radians.
        degrees (bool, optional): Whether the input angles are in degrees. Default is False.

    """

    if not degrees:

        THETA_ib = np.rad2deg(THETA_ib)

    satellite_mesh["Body"].SetPosition(r_i)
    satellite_mesh["Solar Panels"][0].SetPosition(r_i)
    satellite_mesh["Solar Panels"][1].SetPosition(r_i)
    satellite_mesh["Scientific Instrument"].SetPosition(r_i)

    satellite_mesh["Body"].SetOrientation(THETA_ib)
    satellite_mesh["Solar Panels"][0].SetOrientation(THETA_ib)
    satellite_mesh["Solar Panels"][1].SetOrientation(THETA_ib)
    satellite_mesh["Scientific Instrument"].SetOrientation(THETA_ib)


def update_earth_orientation(earth_mesh, t):
    """
    Update the Earth orientation about its z-axis, given time t [s].
    Earth rotates with an angular rate of ~7.2921159e-5 rad/s.
    """
    # Convert the Earth's rotation rate from rad/s to deg/s
    w_ie_deg = np.rad2deg(earth_rot_vel)  # ≈ 0.004178074 deg/s

    # Calculate the accumulated rotation in degrees after t seconds
    orientation_degrees = w_ie_deg * t

    # PyVista's SetOrientation typically expects [x_rot, y_rot, z_rot] in degrees
    # Here, we only rotate about Earth's z-axis
    earth_mesh.SetOrientation([0.0, 0.0, orientation_degrees])


def update_sensor_cone_points(
    plotter,
    line_actor,
    sensor_cone_mesh,
    r_i,
    R_i_b,
    raycasting_length,
    field_of_view_half_deg,
    number_of_raycasting_points,
    earth_radius,
):
    #########################
    # Updating Cone Triangles
    #########################
    # Recalculates the intersection points
    intersection_points_i = calculate_intersection_points_in_inertial_frame(
        r_i,
        R_i_b,
        raycasting_length,
        field_of_view_half_deg,
        number_of_raycasting_points,
        earth_radius,
    )
    # Combine origin with intersection points to form new vertex positions
    new_points = np.vstack([r_i, intersection_points_i])
    # Update the mesh points
    sensor_cone_mesh.points = new_points

    #########################
    # Update the circle lines
    #########################
    circle_lines = []
    for i in range(len(intersection_points_i) - 1):
        circle_lines.append([intersection_points_i[i], intersection_points_i[i + 1]])

    circle_lines.append([intersection_points_i[-1], intersection_points_i[0]])

    # Convert lines to a NumPy array
    line_points = np.array(circle_lines).reshape(-1, 3)

    # Remove the old lines if line_actor exists
    if line_actor is not None:
        plotter.remove_actor(line_actor)

    line_actor = plotter.add_lines(line_points, color="green", width=2)

    return line_actor


def update_body_frame_pose(body_frame, r_i, THETA_ib, degrees=True):
    """
    Update the body-frame axes (mesh) and their labels.

    Parameters
    ----------
    body_frame : dict
        Dictionary containing:
          - 'x', 'y', 'z':  the pyvista mesh objects for each axis arrow
          - 'x_label', 'y_label', 'z_label':  text/label objects
          - 'scale':  a float scale factor for arrow and label size
    r_i : array-like
        3D position of the body-frame origin in the inertial frame (e.g., ECI)
    THETA_ib : list or tuple of float
        Euler angles [phi, theta, psi] in degrees, used by `SetOrientation`
    degrees : bool, optional
        Whether the input angles are in degrees. Default is True.
    """

    if not degrees:
        THETA_ib = np.rad2deg(THETA_ib)

    # 1) Update the arrow mesh positions and orientations
    body_frame["x"].SetPosition(r_i)
    body_frame["y"].SetPosition(r_i)
    body_frame["z"].SetPosition(r_i)

    body_frame["x"].SetOrientation(THETA_ib)
    body_frame["y"].SetOrientation(THETA_ib)
    body_frame["z"].SetOrientation(THETA_ib)

    # 2) Obtain the rotation matrix from body to inertial using your helper function
    R_i_b = pyvista_rotation_matrix_from_euler_angles(THETA_ib)

    # 3) Update label positions
    scale = body_frame["scale"]
    body_frame["x_label"].position = r_i + R_i_b.dot([1, 0, 0]) * scale
    body_frame["y_label"].position = r_i + R_i_b.dot([0, 1, 0]) * scale
    body_frame["z_label"].position = r_i + R_i_b.dot([0, 0, 1]) * scale


def update_orbit_frame_pose(orbit_frame, r_i, THETA_io, degrees=True):
    """
    Updates the orbit-frame axes so that:
      x^o points radially outward (r_i / ||r_i||),
      z^o is normal to the orbit plane (h / ||h||, where h = r x v),
      y^o = z^o x x^o,
    and positions the frame at the satellite location.

    Parameters
    ----------
    orbit_frame : dict
        The dictionary returned by create_orbit_frame().
    r_i : ndarray of shape (3,)
        The satellite's position vector in the inertial frame [km or whatever units].
    THETA_io : list or tuple of float
        Euler angles [phi, theta, psi] in degrees, used by `SetOrientation`.
    degrees : bool, optional
        Whether the input angles are in degrees. Default is True.
    """
    if not degrees:
        THETA_io = np.degrees(THETA_io)

    # 1) Update the arrow mesh positions and orientations
    orbit_frame["x"].SetPosition(r_i)
    orbit_frame["y"].SetPosition(r_i)
    orbit_frame["z"].SetPosition(r_i)

    orbit_frame["x"].SetOrientation(THETA_io)
    orbit_frame["y"].SetOrientation(THETA_io)
    orbit_frame["z"].SetOrientation(THETA_io)

    # 2) Compute the rotation matrix from orbit to inertial using your helper function
    R_i_o = pyvista_rotation_matrix_from_euler_angles(THETA_io)

    # 3) Update label positions
    scale = orbit_frame["scale"]
    orbit_frame["x_label"].position = r_i + R_i_o[:, 0] * scale
    orbit_frame["y_label"].position = r_i + R_i_o[:, 1] * scale
    orbit_frame["z_label"].position = r_i + R_i_o[:, 2] * scale


def update_ecef_frame_orientation(ecef_frame, t):
    """
    Update the ECEF frame orientation about its z^e-axis and
    properly position the axis labels.

    Parameters
    ----------
    ecef_frame : dict
        Dictionary containing:
         - "x", "y", "z":         the pyvista mesh objects for each axis arrow
         - "x_label", "y_label", "z_label": text/label objects
         - "scale":               a float scale factor for arrow/label length
         - "w_ie_deg":            Earth rotation rate in deg/s (if you choose to store it here)
    t : float
        Current time in seconds
    """
    # 1) Compute Earth's rotation angle about z in DEGREES after time t
    #    If you store w_ie_deg in ecef_frame, you can do:
    w_ie_deg = ecef_frame.get("w_ie_deg", np.rad2deg(earth_rot_vel))
    angle_z_deg = w_ie_deg * t

    # 2) Set orientation of the ECEF axes about z^e
    #    The order here is [rotX, rotY, rotZ] in degrees for PyVista's SetOrientation
    ecef_frame["x"].SetOrientation([0.0, 0.0, angle_z_deg])
    ecef_frame["y"].SetOrientation([0.0, 0.0, angle_z_deg])
    ecef_frame["z"].SetOrientation([0.0, 0.0, angle_z_deg])

    # 3) Update label positions
    #    The rotation matrix about z requires angle in RADIANS:
    angle_z_rad = np.deg2rad(angle_z_deg)

    # Build the 3x3 rotation matrix about z^e
    RotZ = Rz(angle_z_rad)

    scale = ecef_frame["scale"]

    # x_label is at RotZ * [1, 0, 0]*scale
    ecef_frame["x_label"].position = RotZ.dot([1, 0, 0]) * scale
    # y_label is at RotZ * [0, 1, 0]*scale
    ecef_frame["y_label"].position = RotZ.dot([0, 1, 0]) * scale
    # z_label is at RotZ * [0, 0, 1]*scale
    ecef_frame["z_label"].position = RotZ.dot([0, 0, 1]) * scale


def visualize_scene(
    data_log,
    time_index=-1,
    off_screen=False,
):
    """
    Creates a single still scene of the satellite around Earth with various reference frames.

    Parameters:
    -----------
    data_log : dict
        Dictionary containing the orbit and attitude information. Must contain at least:
        - "time" (1D array of time [s])
        - "r_i" (Nx3 array of satellite positions in the inertial frame)
        - "THETA_ib" (Nx3 array of Euler angles from inertial to body)
        - "THETA_io" (Nx3 array of Euler angles from inertial to orbit)
    time_index : int, optional
        Index into data_log arrays. Default is -1 (the last time step).
    off_screen : bool, optional
        If True, PyVista’s Plotter is created in off-screen mode. Default is False.

    Returns:
    --------
    plotter : pyvista.Plotter
        The PyVista plotter containing the scene. Use plotter.show() to display it.
    """

    # Function constants:
    earth_radius = 6378
    raycasting_length = 10000
    field_of_view_half_deg = 15
    number_of_raycasting_points = 100
    frames_scale_multiplicators = [2, 1.5, 0.5]
    satellite_size_multiplier = 0.1

    # --- Retrieve single time/index from data_log ---
    t_array = np.array(data_log["time"])
    r_i_array = np.array(data_log["r_i"])
    THETA_ib_array = np.array(data_log["THETA_ib"])
    THETA_io_array = np.array(data_log["THETA_io"])

    # If the user passes an out-of-bounds time_index, clamp it
    if time_index >= len(t_array):
        time_index = len(t_array) - 1

    # Extract desired instant
    time_value = t_array[time_index]
    r_i = r_i_array[time_index]
    THETA_ib = THETA_ib_array[time_index]
    THETA_io = THETA_io_array[time_index]

    # --- Create the PyVista plotter ---
    plotter = pv.Plotter(off_screen=off_screen)

    # --- Create Earth and satellite ---
    satellite_mesh = create_satellite(
        plotter=plotter,
        size=satellite_size_multiplier * earth_radius,
    )
    earth_mesh = create_earth(
        plotter=plotter,
        radius=earth_radius,
    )

    # -- Create Sensor Cone --
    R_i_b = pyvista_rotation_matrix_from_euler_angles(THETA_ib)
    line_actor = plotter.add_lines(np.empty((0, 3)), color="green", width=2)
    sensor_cone_mesh = create_sensor_cone(
        plotter,
        r_i,
        R_i_b,
        raycasting_length=raycasting_length,
        field_of_view_half_deg=field_of_view_half_deg,
        number_of_raycasting_points=number_of_raycasting_points,
        earth_radius=earth_radius,
    )

    # --- Create reference frames ---
    orbit_frame = create_reference_frame(
        plotter,
        labels=["$\\mathbf{x}^o$", "$\\mathbf{y}^o$", "$\\mathbf{z}^o$"],
        scale=frames_scale_multiplicators[2] * earth_radius,
    )
    eci_frame = create_reference_frame(
        plotter,
        labels=["$\\mathbf{x}^i$", "$\\mathbf{y}^i$", "$\\mathbf{z}^i$"],
        scale=frames_scale_multiplicators[0] * earth_radius,
    )
    ecef_frame = create_reference_frame(
        plotter,
        labels=["$\\mathbf{x}^e$", "$\\mathbf{y}^e$", "$\\mathbf{z}^e$"],
        scale=frames_scale_multiplicators[1] * earth_radius,
    )
    body_frame = create_reference_frame(
        plotter,
        labels=["$\\mathbf{x}^b$", "$\\mathbf{y}^b$", "$\\mathbf{z}^b$"],
        scale=frames_scale_multiplicators[2] * earth_radius,
    )

    # --- Add the text labels (PyVista handles them as actors) ---
    plotter.add_actor(orbit_frame["x_label"])
    plotter.add_actor(orbit_frame["y_label"])
    plotter.add_actor(orbit_frame["z_label"])
    plotter.add_actor(eci_frame["x_label"])
    plotter.add_actor(eci_frame["y_label"])
    plotter.add_actor(eci_frame["z_label"])
    plotter.add_actor(ecef_frame["x_label"])
    plotter.add_actor(ecef_frame["y_label"])
    plotter.add_actor(ecef_frame["z_label"])
    plotter.add_actor(body_frame["x_label"])
    plotter.add_actor(body_frame["y_label"])
    plotter.add_actor(body_frame["z_label"])

    # --- Update positions/orientations for this snapshot ---
    #   1) Satellite pose
    update_satellite_pose(satellite_mesh, r_i, THETA_ib, degrees=True)
    #   2) Earth orientation
    update_earth_orientation(earth_mesh, time_value)
    #   3) ECEF frame orientation
    update_ecef_frame_orientation(ecef_frame, time_value)
    #   4) Body frame pose
    update_body_frame_pose(body_frame, r_i, THETA_ib, degrees=True)
    #   5) Orbit frame pose
    update_orbit_frame_pose(orbit_frame, r_i, THETA_io, degrees=True)
    #   6) Sensor cone
    line_actor = update_sensor_cone_points(
        plotter,
        line_actor,
        sensor_cone_mesh,
        r_i,
        R_i_b,
        raycasting_length,
        field_of_view_half_deg,
        number_of_raycasting_points,
        earth_radius,
    )

    # --- Optional camera setup (you can adjust as you prefer) ---
    # Example: place the camera far above the North Pole
    # plotter.camera.position = (0, 0, 20 * earth_radius)
    plotter.camera.focal_point = (0, 0, 0)  # look at Earth's center
    # plotter.camera.up = (0, 1, 0)

    # Return the plotter so user can call plotter.show() or do further manipulations
    return plotter


def animate_satellite(t, data_log, gif_name):
    """
    Animates the satellite's position and orientation over time using PyVista.
    Parameters:
        t (ndarray): Array of time values [s].
        data_log (dict): Dictionary containing logged data for the satellite's position and orientation.
        gif_name (str): Name of the output GIF file.
    """

    # Create a PyVista plotter
    plotter = pv.Plotter(off_screen=False)

    # Constants
    earth_radius = 6378
    raycasting_length = 10000
    field_of_view_half_deg = 15
    number_of_raycasting_points = 100
    frames_scale_multiplicators = [2, 1.5, 0.5]
    satellite_size_multiplier = 0.1

    # Create the satellite, Earth
    satellite_mesh = create_satellite(
        plotter=plotter,
        size=satellite_size_multiplier * earth_radius,
    )
    earth_mesh = create_earth(
        plotter=plotter,
        radius=earth_radius,
    )

    # Create sensor cone
    R_i_b = pyvista_rotation_matrix_from_euler_angles(data_log["THETA_ib"][0])
    r_i = data_log["r_i"][0]
    sensor_cone_mesh = create_sensor_cone(
        plotter,
        r_i,
        R_i_b,
        raycasting_length=raycasting_length,
        field_of_view_half_deg=field_of_view_half_deg,
        number_of_raycasting_points=number_of_raycasting_points,
        earth_radius=earth_radius,
    )
    line_actor = plotter.add_lines(np.empty((0, 3)), color="green", width=2)

    # Create the reference frames
    orbit_frame = create_reference_frame(
        plotter,
        labels=np.array(["$\mathbf{x}^o$", "$\mathbf{y}^o$", "$\mathbf{z}^o$"]),
        scale=frames_scale_multiplicators[2] * earth_radius,
    )
    eci_frame = create_reference_frame(
        plotter,
        labels=np.array(["$\mathbf{x}^i$", "$\mathbf{y}^i$", "$\mathbf{z}^i$"]),
        scale=frames_scale_multiplicators[0] * earth_radius,
    )
    ecef_frame = create_reference_frame(
        plotter,
        labels=np.array(["$\mathbf{x}^e$", "$\mathbf{y}^e$", "$\mathbf{z}^e$"]),
        scale=frames_scale_multiplicators[1] * earth_radius,
    )
    body_frame = create_reference_frame(
        plotter,
        labels=np.array(["$\mathbf{x}^b$", "$\mathbf{y}^b$", "$\mathbf{z}^b$"]),
        scale=frames_scale_multiplicators[2] * earth_radius,
    )

    # Add the meshes to the plotter
    plotter.add_actor(orbit_frame["x_label"])
    plotter.add_actor(orbit_frame["y_label"])
    plotter.add_actor(orbit_frame["z_label"])
    plotter.add_actor(eci_frame["x_label"])
    plotter.add_actor(eci_frame["y_label"])
    plotter.add_actor(eci_frame["z_label"])
    plotter.add_actor(ecef_frame["x_label"])
    plotter.add_actor(ecef_frame["y_label"])
    plotter.add_actor(ecef_frame["z_label"])
    plotter.add_actor(body_frame["x_label"])
    plotter.add_actor(body_frame["y_label"])
    plotter.add_actor(body_frame["z_label"])

    # Trajectory of the Satellite
    trajectory_points = []
    trajectory_actor = None

    # Extracting satellite position
    r_i_array = np.array(data_log["r_i"])

    # plotter.view_vector([0, 0, -1])
    # plotter.camera.position = (0, 0, 20 * earth_radius)

    # Initialize the gif
    plotter.open_gif(gif_name)

    # Animate and update the satellite's position and orientation
    for i in range(len(t)):

        # Extracting the data
        time = t[i]
        r_i = r_i_array[i]
        THETA_ib = np.array(data_log["THETA_ib"][i])
        THETA_io = np.array(data_log["THETA_io"][i])
        R_i_b = pyvista_rotation_matrix_from_euler_angles(THETA_ib)

        # Updating the satellite's position and orientation
        update_satellite_pose(satellite_mesh, r_i, THETA_ib)
        update_earth_orientation(earth_mesh, time)
        update_ecef_frame_orientation(ecef_frame, time)
        update_body_frame_pose(body_frame, r_i, THETA_ib)
        update_orbit_frame_pose(orbit_frame, r_i, THETA_io)
        line_actor = update_sensor_cone_points(
            plotter,
            line_actor,
            sensor_cone_mesh,
            r_i,
            R_i_b,
            raycasting_length,
            field_of_view_half_deg,
            number_of_raycasting_points,
            earth_radius,
        )

        # Update Trajectory:
        trajectory_points.append(np.array(r_i))
        if len(trajectory_points) > 1:
            pts = np.array(trajectory_points)
            n_pts = pts.shape[0]
            cells = np.hstack([[n_pts], np.arange(n_pts)])
            traj_poly = pv.PolyData(pts, lines=cells)
            if trajectory_actor is not None:
                plotter.remove_actor(trajectory_actor)
            trajectory_actor = plotter.add_mesh(traj_poly, color="purple", line_width=3)

        # Update the plotter
        plotter.write_frame()

    # Closing and finalizing the gif
    plotter.close()


if __name__ == "__main__":

    pass
