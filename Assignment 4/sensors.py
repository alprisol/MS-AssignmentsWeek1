import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def calculate_line_sphere_intersection_point(p1, p2, p3, r):
    """
    Calculate the intersection point(s) of a line (p1->p2) and a sphere
    centered at p3 with radius r.

    :param p1: First point on the line (3D vector)
    :param p2: Second point on the line (3D vector)
    :param p3: Center of the sphere (3D vector)
    :param r: Radius of the sphere
    :return: None if no intersection, a single point if tangent,
             or a tuple of two points if two intersection points exist.
    """
    p1 = np.array(p1, dtype=float)
    p2 = np.array(p2, dtype=float)
    p3 = np.array(p3, dtype=float)

    d = p2 - p1  # Direction vector of the line
    f = p1 - p3  # Vector from sphere center to p1
    m = (p1 + p2) / 2

    a = np.dot(d, d)
    b = 2 * np.dot(f, d)
    c = np.dot(f, f) - r**2

    discriminant = b**2 - 4 * a * c

    if discriminant < 0:
        # No intersection
        return None, False
    elif np.isclose(discriminant, 0.0):
        # One intersection
        i = p1 + (-b / (2 * a)) * d
        return i, True
    else:
        # Two intersection points
        sqrt_disc = np.sqrt(discriminant)
        i1 = p1 + ((-b + sqrt_disc) / (2 * a)) * d
        i2 = p1 + ((-b - sqrt_disc) / (2 * a)) * d
        # Find the closest intersection point to the midpoint of the line
        if np.linalg.norm(i1 - m) < np.linalg.norm(i2 - m):
            return i1, True
        else:
            return i2, True


def calculate_raycasting_points(
    R_i_b,
    r_i,
    raycasting_length=10000,
    field_of_view_half=np.radians(30 / 2),
    number_of_raycasting_points=10,
):

    # Initializes the raycasting points
    raycasting_points_i = []
    radius = raycasting_length * np.tan(field_of_view_half)

    # Generating a number of points in a sphere at a point along the x^b axis
    for theta in np.linspace(0, 2 * np.pi, number_of_raycasting_points, endpoint=False):
        points_i = R_i_b @ np.array(
            [raycasting_length, radius * np.cos(theta), radius * np.sin(theta)]
        )
        raycasting_points_i.append(r_i + points_i)

    raycasting_points_i = np.array(raycasting_points_i)

    return raycasting_points_i


def calculate_intersection_points_in_inertial_frame(
    r_i,
    R_i_b,
    raycasting_length,
    field_of_view_half_deg,
    number_of_raycasting_points,
    radius,
):

    fov_half_rad = np.radians(field_of_view_half_deg)
    raycasting_points_i = calculate_raycasting_points(
        R_i_b,
        r_i,
        raycasting_length=raycasting_length,
        field_of_view_half=fov_half_rad,
        number_of_raycasting_points=number_of_raycasting_points,
    )

    # Satellite and Earth are fixed
    p1 = r_i
    p3 = np.array([0, 0, 0])  # Earth is at the origin
    intersection_points_i = np.zeros_like(raycasting_points_i)

    # Looping through the different raycasting points
    for i in range(0, len(raycasting_points_i[:, 0])):
        p2 = raycasting_points_i[i]
        closest_intersection_point, is_line_intersecting = (
            calculate_line_sphere_intersection_point(p1, p2, p3, radius)
        )
        if is_line_intersecting == True:
            intersection_points_i[i] = closest_intersection_point
        else:
            intersection_points_i[i] = raycasting_points_i[i]

    return intersection_points_i


# Example usage:
if __name__ == "__main__":

    p1 = [1, 2, 3]
    p2 = [4, 5, 6]
    p3 = [0, 0, 0]
    r = 5

    intersection_point, exists_intersection = calculate_line_sphere_intersection_point(
        p1, p2, p3, r
    )
    print(intersection_point)
