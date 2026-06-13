export const defaultTerrainCode = `import numpy as np
from scipy.ndimage import gaussian_filter

def generate_terrain(xx: np.ndarray, yy: np.ndarray) -> np.ndarray:
    """Generate a smooth multi-peak terrain height map."""
    peaks = [
        (20, 20, 5, 8),
        (20, 70, 5, 8),
        (60, 20, 5, 8),
        (60, 70, 5, 8),
    ]

    zz = np.zeros_like(xx)
    for center_x, center_y, amplitude, radius in peaks:
        distance = (xx - center_x) ** 2 + (yy - center_y) ** 2
        zz += 10 * amplitude * np.exp(-distance / (2 * radius ** 2))

    zz += 0.2 * np.sin(0.5 * np.sqrt(xx**2 + yy**2))
    return np.clip(gaussian_filter(zz, sigma=3), 0, None)`

export const defaultFitnessCode = `def fitness_function(path_points: np.ndarray) -> float:
    """Balance safety, path efficiency and flight constraints."""
    points = path_points.reshape(-1, 3)
    full_path = np.vstack([start_point, points, destination])
    smooth_path = interpolate.smooth_path_with_cubic_spline(full_path)

    path_diff = np.diff(smooth_path, axis=0)
    horizontal_length = np.linalg.norm(path_diff[:, :2], axis=1).sum()
    height_change = np.abs(path_diff[:, 2]).sum()

    vertical_collisions = collision_detection.check_vertical_collision_batch(
        xx, yy, zz, smooth_path, min_safe_distance=1
    ).sum()

    slope_cost = (
        flight_angle_calculator.calculate_slope_angles_batch(
            smooth_path[:-1], smooth_path[1:]
        ) > 45
    ).sum()

    return (
        horizontal_length
        + height_change
        + vertical_collisions * 1e4
        + slope_cost
    )`
