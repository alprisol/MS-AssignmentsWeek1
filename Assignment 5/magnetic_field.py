import ppigrf
from datetime import datetime

from orbital_mechanics_5 import *

# Set global NumPy print options
np.set_printoptions(
    precision=3,  # Limit the precision to 3 decimal places
    suppress=False,  # Avoid scientific notation for small numbers
)


# Function to Calculate Magnetic Field in Orbit Frame
def calculate_magnetic_field_in_orbit_frame(r_i, date, omega, theta, OMEGA, i, t):
    """
    Calculate magnetic field in orbit frame given the position vector, rotation matrix, and date.
    """
    # Convert r_i to LLA (latitude in º, longitude in º, altitude in km)
    latitude, longitude, altitude = calculate_lla_from_ecef(r_i)
    print(f"Latitude:{latitude}º\nLongitude:{longitude}º\nAltitude: {altitude} km\n")

    # Obtain magnetic field in ENU using IGRF model
    Be, Bn, Bu = ppigrf.igrf(longitude, latitude, altitude, date)

    # Magnetic field vector in ENU (in Teslas, multiply by 1e-9)
    B_ENU = np.array([Be, Bn, Bu]) * 1e-9

    # Transformation: ENU -> NED -> ECEF -> ECI -> Orbit
    R_i_o = calculate_rotation_matrix_from_orbit_to_inertial(omega, theta, OMEGA, i)
    R_ECEF_i = calculate_rotation_matrix_from_inertial_to_ecef(t)
    R_NED_ECEF = calculate_rotation_matrix_from_ecef_to_ned(latitude, longitude)
    R_ENU_NED = calculate_rotation_matrix_from_ned_to_enu()

    B_NED = R_ENU_NED.T @ B_ENU
    B_ECEF = R_NED_ECEF.T @ B_NED
    B_i = R_ECEF_i.T @ B_ECEF
    B_o = R_i_o.T @ B_i

    return B_ENU, B_o


if __name__ == "__main__":

    OMEGA = 0
    omega = 0
    i = np.radians(75)
    theta = np.radians(30)
    t = 30

    r_i = [2938.363, 942.355, 7769.299]
    date = datetime(2025, 1, 10)

    B_ENU, B_o = calculate_magnetic_field_in_orbit_frame(
        r_i, date, omega, theta, OMEGA, i, t
    )
    print(f"B in ENU frame: \n{B_ENU*1e9}\n")
    print(f"B in orbit frame: \n{B_o*1e9}\n")
