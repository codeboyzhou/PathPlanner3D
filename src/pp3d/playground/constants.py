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
    return 0.0
""".strip()
