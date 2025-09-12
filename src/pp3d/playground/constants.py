# The code template for terrain generation function
TERRAIN_GENERATION_CODE_TEMPLATE = """
def generate_terrain(xx: np.ndarray, yy: np.ndarray) -> np.ndarray:
    zz = np.zeros_like(xx)
    return zz
""".strip()

# The code template for fitness function
FITNESS_FUNCTION_CODE_TEMPLATE = """
def fitness_function(path_point_positions: np.ndarray) -> float:
    return 0.0
""".strip()
