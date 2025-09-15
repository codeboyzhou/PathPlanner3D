import numpy as np
from scipy.interpolate import RegularGridInterpolator


def check(xx: np.ndarray, yy: np.ndarray, zz: np.ndarray, point: np.ndarray) -> bool:
    """
    Check if the path points collide with the terrain.

    Args:
        xx (np.ndarray): The x coordinates of the terrain height map.
        yy (np.ndarray): The y coordinates of the terrain height map.
        zz (np.ndarray): The z coordinates of the terrain height map.
        point (np.ndarray): The point to check for collision.

    Returns:
        bool: True if the path points collide with the terrain, False otherwise.
    """
    terrain_x = xx[0, :]
    terrain_y = yy[:, 0]

    # NOTE: The `points` is (terrain_y, terrain_x) here, NOT (terrain_x, terrain_y)
    interpolator = RegularGridInterpolator(points=(terrain_y, terrain_x), values=zz, bounds_error=False, fill_value=0)

    point_x, point_y, point_z = point
    point_x = np.clip(point_x, xx.min(), xx.max())
    point_y = np.clip(point_y, yy.min(), yy.max())

    terrain_z = interpolator([point_y, point_x])

    return point_z < terrain_z * 1.1
