import pyvista as pv
import numpy as np
import matplotlib.pyplot as plt
import vtk

from orbital_mechanics import *

# Set global NumPy print options
np.set_printoptions(
    precision=3,  # Limit the precision to 3 decimal places
    suppress=True,  # Avoid scientific notation for small numbers
)

# Enable LaTeX rendering and importing the xcolor package
plt.rcParams["text.usetex"] = True
plt.rcParams["text.latex.preamble"] = r"\usepackage{xcolor}"

earth_rot_vel = 7.2722e-5  # rad/s


def geometry_examples():

    cyl = pv.Cylinder()
    arrow = pv.Arrow()
    sphere = pv.Sphere()
    plane = pv.Plane()
    line = pv.Line()
    box = pv.Box()
    cone = pv.Cone()
    poly = pv.Polygon()
    disc = pv.Disc()

    p = pv.Plotter(shape=(3, 3))
    # Top row
    p.subplot(0, 0)
    p.add_mesh(cyl, color="red", show_edges=True)
    p.subplot(0, 1)
    p.add_mesh(arrow, color="red", show_edges=False)
    p.subplot(0, 2)
    p.add_mesh(sphere, color="red", show_edges=True)
    # Middle row
    p.subplot(1, 0)
    p.add_mesh(plane, color="yellow", show_edges=True)
    p.subplot(1, 1)
    p.add_mesh(line, color="yellow", line_width=3)
    p.subplot(1, 2)
    p.add_mesh(box, color="yellow", show_edges=True)
    # Bottom row
    p.subplot(2, 0)
    p.add_mesh(cone, color="red", show_edges=True)
    p.subplot(2, 1)
    p.add_mesh(poly, color="red", show_edges=False)
    p.subplot(2, 2)
    p.add_mesh(disc, color="red", show_edges=True)
    # Render all of them
    p.show()


def create_reference_frame(plotter, labels, scale=1):

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

    # Satellite's body
    body_texture = pv.read_texture("Assignment 1/satellite_texture.png")
    body_b = pv.Box(bounds=(-size, size, -size, size, -size, size))
    u = np.array([0, 1, 1, 0] * 6)
    v = np.array([0, 0, 1, 1] * 6)
    texture_coordinates = np.c_[u, v]
    body_b.texture_map_to_plane(inplace=True)

    # Solar panels
    solar_panel_texture = pv.read_texture(
        "Assignment 1/high_quality_solar_panel_texture.png"
    )
    panel_width = 1.5 * size
    panel_length = 5 * size
    panel_thickness = 0.1 * size
    center_offset = 0.5 * size
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

    # Scientific instrument along x_b axis
    scientific_instrument_texture = pv.read_texture("Assignment 1/camera_texture.png")
    scientific_instrument_b = pv.Cone(
        center=(size - 0.01, 0, 0),
        direction=(-1, 0, 0),
        height=0.5 * size,
        radius=0.5 * size,
        resolution=50,
    )
    scientific_instrument_b.texture_map_to_sphere(inplace=True)

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

    earth = pv.examples.planets.load_earth(radius=radius)
    earth_texture = pv.examples.load_globe_texture()
    earth_mesh = plotter.add_mesh(earth, texture=earth_texture, smooth_shading=True)

    return earth_mesh


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


def update_satellite_pose(satellite_mesh, r_i, Theta_ib, degrees=False):

    if not degrees:

        Theta_ib = np.rad2deg(Theta_ib)

    satellite_mesh["Body"].SetPosition(r_i)
    satellite_mesh["Solar Panels"][0].SetPosition(r_i)
    satellite_mesh["Solar Panels"][1].SetPosition(r_i)
    satellite_mesh["Scientific Instrument"].SetPosition(r_i)

    satellite_mesh["Body"].SetOrientation(Theta_ib)
    satellite_mesh["Solar Panels"][0].SetOrientation(Theta_ib)
    satellite_mesh["Solar Panels"][1].SetOrientation(Theta_ib)
    satellite_mesh["Scientific Instrument"].SetOrientation(Theta_ib)


def update_earth_orientation(earth_mesh, t):
    """
    Update the Earth orientation about its z-axis, given time t [s].
    Earth rotates with an angular rate of ~7.2921159e-5 rad/s.
    """
    # Convert the Earth's rotation rate from rad/s to deg/s
    w_ie_deg = np.rad2deg(earth_rot_vel)

    # Calculate the accumulated rotation in degrees after t seconds
    orientation_degrees = w_ie_deg * t

    # PyVista's SetOrientation typically expects [x_rot, y_rot, z_rot] in degrees
    # Here, we only rotate about Earth's z-axis
    earth_mesh.SetOrientation([0.0, 0.0, orientation_degrees])


def update_body_frame_pose(body_frame, r_i, Theta_ib, degrees=False):
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
    Theta_ib : list or tuple of float
        Euler angles [phi, theta, psi] in degrees, used by `SetOrientation`
    degrees : bool, optional
    """

    if not degrees:
        Theta_ib = np.rad2deg(Theta_ib)

    # 1) Update the arrow mesh positions and orientations
    body_frame["x"].SetPosition(r_i)
    body_frame["y"].SetPosition(r_i)
    body_frame["z"].SetPosition(r_i)

    body_frame["x"].SetOrientation(Theta_ib)
    body_frame["y"].SetOrientation(Theta_ib)
    body_frame["z"].SetOrientation(Theta_ib)

    # 2) Obtain the rotation matrix from body to inertial using your helper function
    R_i_b = pyvista_rotation_matrix_from_euler_angles(Theta_ib)
    # R_i_b transforms a vector written in body-frame coords into inertial coords

    # 3) Update label positions
    #    Each label is placed at the tip of its respective axis arrow in the inertial frame
    scale = body_frame["scale"]
    # x_label is at r_i + R_i_b*[1, 0, 0]*(scale)
    body_frame["x_label"].position = r_i + R_i_b.dot([1, 0, 0]) * scale
    body_frame["y_label"].position = r_i + R_i_b.dot([0, 1, 0]) * scale
    body_frame["z_label"].position = r_i + R_i_b.dot([0, 0, 1]) * scale


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


def visualize_scene():
    """
    Create the scene with:
      - Earth at center (radius = Re)
      - Satellite at 3*Re along y_i with orientation [90, 45, 0]
      - ECI frame (2*Re)
      - ECEF frame (1.5*Re)
      - Body frame (0.5*Re)
      - Add all reference-frame labels as actors
      - Update satellite and body orientation
    """

    # 1) Create a Plotter
    p = pv.Plotter()

    # 2) Define Earth's radius (you could pick your own scale, e.g. 6378 or 1.0)
    Re = 6378.0

    # 3) Create Earth
    earth_mesh = create_earth(p, Re)

    # 4) Create ECI reference frame
    eci_frame = create_reference_frame(
        p, labels=["$x_i$", "$y_i$", "$z_i$"], scale=2 * Re
    )
    # Add label actors explicitly
    p.add_actor(eci_frame["x_label"])
    p.add_actor(eci_frame["y_label"])
    p.add_actor(eci_frame["z_label"])

    # 5) Create ECEF reference frame
    ecef_frame = create_reference_frame(
        p, labels=["$x_e$", "$y_e$", "$z_e$"], scale=1.5 * Re
    )
    p.add_actor(ecef_frame["x_label"])
    p.add_actor(ecef_frame["y_label"])
    p.add_actor(ecef_frame["z_label"])

    # 6) Create body reference frame
    body_frame = create_reference_frame(
        p, labels=["$x_b$", "$y_b$", "$z_b$"], scale=0.5 * Re
    )
    p.add_actor(body_frame["x_label"])
    p.add_actor(body_frame["y_label"])
    p.add_actor(body_frame["z_label"])

    # 7) Create the satellite (size = 0.1*Re)
    satellite_mesh = create_satellite(p, size=0.1 * Re)

    # 8) Update the satellite pose
    #    Satellite at 3*Re => [0, 2.121*Re, 2.121*Re]
    #    Orientation = [90, 45, 0] in degrees
    r_i = [0.0 * Re, 2.121 * Re, 2.121 * Re]
    Theta_i_b_deg = [90, 45, 0]
    update_satellite_pose(satellite_mesh, r_i, Theta_i_b_deg, degrees=True)

    # 9) Update the body frame orientation
    update_body_frame_pose(body_frame, r_i, Theta_i_b_deg, degrees=True)

    # 10) Optionally, set ECEF orientation at t=0 (no rotation yet)
    update_ecef_frame_orientation(ecef_frame, t=0.0)

    # 11) Show the scene
    p.show()


def visualize_reference_frame(labels=["$x$", "$y$", "$z$"], scale=1.0):
    """
    Visualize a single reference frame in an isolated PyVista scene.
    """
    p = pv.Plotter()
    ref_frame = create_reference_frame(p, labels, scale=scale)

    # Add label actors
    p.add_actor(ref_frame["x_label"])
    p.add_actor(ref_frame["y_label"])
    p.add_actor(ref_frame["z_label"])

    p.add_text("Reference Frame Visualization", font_size=12)

    # Show the scene
    p.show()


def visualize_satellite(size=0.5):
    """
    Visualize a single satellite in an isolated PyVista scene.
    """
    p = pv.Plotter()

    # Create the satellite
    sat_mesh = create_satellite(p, size=size)

    p.add_text("Satellite Visualization", font_size=12)

    # Show the scene
    p.show()


def visualize_earth(radius=6378.0):
    """
    Visualize the Earth in an isolated PyVista scene.
    """
    p = pv.Plotter()

    # Create the Earth
    earth_mesh = create_earth(p, radius)

    p.add_text("Earth Visualization", font_size=12)

    # Show the scene
    p.show()


if __name__ == "__main__":

    geometry_examples()

    visualize_reference_frame()

    visualize_satellite()

    visualize_earth()

    vector = np.array([1, 2, 3])

    print("Rotation in X")
    vector_rotX = vector @ Rx(np.pi / 2)
    print(Rx(np.pi / 2))
    print(f"Rotated vector: {vector_rotX}")
    print()

    print("Rotation in Y")
    vector_rotY = vector @ Ry(np.pi / 2)
    print(Ry(np.pi / 2))
    print(f"Rotated vector: {vector_rotY}")
    print()

    print("Rotation in Z")
    vector_rotZ = vector @ Rz(np.pi / 2)
    print(Rz(np.pi / 2))
    print(f"Rotated vector: {vector_rotZ}")
    print()

    visualize_scene()
