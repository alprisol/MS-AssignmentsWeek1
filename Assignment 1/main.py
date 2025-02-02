from visualization import *
from orbital_mechanics import *
import numpy as np
import pyvista as pv


def animate_circular_orbit():
    """
    Create a GIF animation of a satellite in a circular orbit around Earth
    while Earth rotates about its own axis. In addition, a 'stella' (trajectory
    line) is added to visualize the history of positions of the satellite's body-frame
    origin.
    """

    # 1) Create a Plotter
    p = pv.Plotter()

    # 2) Define Earth’s radius (e.g., 6378 km)
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

    # 7) Create satellite (size = 0.1*Re)
    satellite_mesh = create_satellite(p, size=0.1 * Re)

    # 8) Set up orbit parameters
    #    Let us consider a circular orbit with radius = 3 * Re (just for illustration).
    orbit_radius = 3.0 * Re
    # Initial orbit angle
    varphi = 0.0

    # 9) Time parameters for the animation
    simulation_time = 8 * 3600  # 8 hours in seconds
    n_frames = 1000
    time_step = 10  # e.g., 10 seconds per step
    frame_interval = simulation_time // n_frames  # integer division
    # Angular speed for circular orbit
    dot_varphi = calculate_circular_angular_speed(orbit_radius)

    # 10) Open GIF file
    gif_filename = "Assignment 1/satellite_animation.gif"
    p.open_gif(gif_filename)

    # 11) Prepare trajectory storage for the satellite's body-frame origin
    trajectory_points = []  # This list will store positions (history)
    trajectory_actor = None  # Will hold the actor for the trajectory line

    # 12) Main simulation loop
    #     We'll step through the entire simulation_time in increments of time_step
    #     and record the scene whenever we hit the interval for each GIF frame.
    t = 0
    frame_counter = 0

    while t <= simulation_time:

        # (a) Update the orbit angle using a simple Euler forward method
        varphi += dot_varphi * time_step

        # (b) Compute the satellite's new position (and velocity if needed)
        r_i = calculate_satellite_position_in_circular_orbit(varphi, orbit_radius)

        # We'll assume a simple orientation for demonstration,
        # e.g., keep the satellite "facing Earth" with a nominal orientation.
        Theta_i_b_deg = [90.0, 45.0, np.rad2deg(varphi)]  # example orientation

        # (c) Update the satellite's pose in the scene
        update_satellite_pose(satellite_mesh, r_i, Theta_i_b_deg, degrees=True)
        update_body_frame_pose(body_frame, r_i, Theta_i_b_deg, degrees=True)

        # (d) Update Earth's rotation about its own axis at time t
        update_earth_orientation(earth_mesh, t)

        # (e) Update ECEF frame orientation at time t
        update_ecef_frame_orientation(ecef_frame, t)

        # (f) Update the trajectory (stella) of the satellite.
        #     Append the current body-frame origin position to the history.
        trajectory_points.append(np.array(r_i))

        # If we have at least two points, update the trajectory line.
        if len(trajectory_points) > 1:
            pts = np.array(trajectory_points)
            n_pts = pts.shape[0]
            # Create a cell array for a polyline: first element is number of points,
            # followed by the indices [0, 1, ..., n_pts-1].
            cells = np.hstack([[n_pts], np.arange(n_pts)])
            traj_poly = pv.PolyData(pts, lines=cells)
            # Remove the previous trajectory actor if it exists
            if trajectory_actor is not None:
                p.remove_actor(trajectory_actor)
            trajectory_actor = p.add_mesh(traj_poly, color="purple", line_width=3)

        # (g) Check if it’s time to write a frame to the GIF
        if (t % frame_interval) == 0:
            p.write_frame()
            frame_counter += 1
            print(f"Frame {frame_counter} / {n_frames} written at t={t} s")

        # Advance time
        t += time_step

        # Safety stop if we have already stored the required n_frames
        if frame_counter >= n_frames:
            break

    # 13) Close and finalize the GIF
    p.close()
    print(f"GIF animation saved as {gif_filename}")


if __name__ == "__main__":
    animate_circular_orbit()
