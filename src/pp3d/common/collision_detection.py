import numpy as np
from scipy.interpolate import RegularGridInterpolator


def check_vertical_collision(
    xx: np.ndarray, yy: np.ndarray, zz: np.ndarray, point: np.ndarray, safe_distance: float = 1.0
) -> bool:
    """
    Check if the path points collide with the terrain vertically.

    Args:
        xx (np.ndarray): The x coordinates of the terrain height map.
        yy (np.ndarray): The y coordinates of the terrain height map.
        zz (np.ndarray): The z coordinates of the terrain height map.
        point (np.ndarray): The point to check for collision.
        safe_distance (float, optional): The safe distance to the terrain vertically. Defaults to 1.0.

    Returns:
        bool: True if the path points collide with the terrain vertically, False otherwise.
    """
    terrain_x = xx[0, :]
    terrain_y = yy[:, 0]

    # NOTE: The `points` is (terrain_y, terrain_x) here, NOT (terrain_x, terrain_y)
    interpolator = RegularGridInterpolator(points=(terrain_y, terrain_x), values=zz, bounds_error=False, fill_value=0)

    point_x, point_y, point_z = point
    point_x = np.clip(point_x, xx.min(), xx.max())
    point_y = np.clip(point_y, yy.min(), yy.max())

    terrain_z = interpolator([point_y, point_x])

    return point_z < terrain_z + safe_distance


def check_horizontal_collision(
    point: np.ndarray, peaks: list[tuple[float, float, float, float]], safe_distance: float = 1.0
) -> bool:
    """
    Check if the path points collide with the terrain horizontally.

    Args:
        point (np.ndarray): The point to check for collision.
        peaks (list[tuple[float, float, float, float]]): The list of peaks.
            Each peak is a tuple of (center_x, center_y, amplitude, radius).
        safe_distance (float, optional): The safe distance to the terrain horizontally. Defaults to 1.0.

    Returns:
        bool: True if the path points collide with the terrain horizontally, False otherwise.
    """
    point_x, point_y, point_z = point
    for peak in peaks:
        center_x, center_y, _, radius = peak
        distance = np.sqrt((point_x - center_x) ** 2 + (point_y - center_y) ** 2)
        if distance < radius + safe_distance:
            return True
    return False
