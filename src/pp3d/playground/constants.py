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
def fitness_function(path_point_positions: np.ndarray) -> float:
    start_point = np.array([0, 0, 5])
    destination = np.array([90, 90, 5])
    
    path_point_positions = np.insert(path_point_positions, 0, start_point)
    path_point_positions = np.append(path_point_positions, destination)
    waypoints = path_point_positions.reshape(-1, 3)
    
    x = waypoints[:, 0]
    y = waypoints[:, 1]
    z = waypoints[:, 2]
    
    splreps = np.linspace(0, 1, len(waypoints))
    splevs = np.linspace(0, 1, 100)
    
    splrep_x = splrep(splreps, x)
    splrep_y = splrep(splreps, y)
    splrep_z = splrep(splreps, z)
    
    splev_x = splev(splevs, splrep_x)
    splev_y = splev(splevs, splrep_y)
    splev_z = splev(splevs, splrep_z)
    
    waypoints = np.column_stack((splev_x, splev_y, splev_z))
    
    dx, dy, dz = np.diff(splev_x), np.diff(splev_y), np.diff(splev_z)
    path_length = np.sum(np.sqrt(dx**2 + dy**2 + dz**2))
    
    average_height = np.mean(splev_z)
    
    return path_length + average_height
""".strip()
