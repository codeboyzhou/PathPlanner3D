# The code template for terrain generation function
TERRAIN_GENERATION_CODE_TEMPLATE = """
def generate_terrain(xx: np.ndarray, yy: np.ndarray) -> np.ndarray:
    zz = np.zeros_like(xx)
    
    peaks=[
        # (center_x, center_y, amplitude, radius)
        (20, 20, 5, 8),
        (20, 70, 5, 8),
        (60, 20, 5, 8),
        (60, 70, 5, 8)
    ]
    
    for peak in peaks:
        center_x, center_y, amplitude, radius = peak
        zz += 10 * amplitude * np.exp(-((xx - center_x) ** 2 + (yy - center_y) ** 2) / (2 * radius ** 2))
    
    zz += 0.2 * np.sin(0.5 * np.sqrt(xx**2 + yy**2)) + 0.1 * np.random.normal(size=xx.shape)
    
    zz = gaussian_filter(zz, sigma=3)
    
    zz = np.clip(zz, 0, None)
    
    return zz
""".strip()

# The code template for fitness function
FITNESS_FUNCTION_CODE_TEMPLATE = """
def fitness_function(path_points: np.ndarray) -> float:
    reshaped_path_points = path_points.reshape(-1, 3)
    full_path_points = np.vstack([start_point, reshaped_path_points, destination])
    
    x = full_path_points[:, 0]
    y = full_path_points[:, 1]
    z = full_path_points[:, 2]
    
    splreps = np.linspace(0, 1, len(full_path_points))
    splevs = np.linspace(0, 1, 100)
    
    splrep_x = splrep(splreps, x)
    splrep_y = splrep(splreps, y)
    splrep_z = splrep(splreps, z)
    
    splev_x = splev(splevs, splrep_x)
    splev_y = splev(splevs, splrep_y)
    splev_z = splev(splevs, splrep_z)
    
    full_path_points = np.column_stack((splev_x, splev_y, splev_z))
    
    # Calculate the collision cost
    collision_cost = np.sum([1e4 if collision_detection.check(p, terrain_height_map) else 0 for p in full_path_points])
    
    # Calculate the path length cost
    path_diff = np.diff(full_path_points, axis=0)
    path_length = np.sum(np.sqrt(np.sum(path_diff**2, axis=1)))
    
    # Calculate the average height cost
    average_height = np.mean(splev_z)
    
    return collision_cost + path_length + average_height
""".strip()
