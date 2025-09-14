import numpy as np


def check_collision(point: np.ndarray, terrain_height_map: np.ndarray) -> bool:
    """
    Check if the path points collide with the terrain.

    Args:
        point (np.ndarray): The point to check for collision.
        terrain_height_map (np.ndarray): The height map of the terrain.

    Returns:
        bool: True if the path points collide with the terrain, False otherwise.
    """
    point_x, point_y, point_z = point
    terrain_height_map_rows, terrain_height_map_cols = terrain_height_map.shape

    idx_x = int(np.round(point_x))
    idx_y = int(np.round(point_y))

    if not (0 <= idx_x < terrain_height_map_rows and 0 <= idx_y < terrain_height_map_cols):
        return False

    terrain_height = terrain_height_map[idx_x, idx_y]
    return point_z <= terrain_height * 1.1
