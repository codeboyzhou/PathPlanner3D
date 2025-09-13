import numpy as np
from plotly import graph_objects
from plotly.graph_objs import Figure


def calculate_camera_eye(elev: float, azim: float, distance: float = 2.5) -> dict:
    """Calculate the camera eye position for Plotly 3D scene.

    Args:
        elev: Elevation angle in degrees.
        azim: Azimuth angle in degrees.
        distance: Distance from the camera to the scene.

    Returns:
        A dictionary with "x", "y", and "z" keys representing the camera eye position.
    """
    # Convert angles to radians
    elev_rad = np.radians(elev)
    azim_rad = np.radians(azim)

    # Calculate camera position
    x = np.cos(elev_rad) * np.cos(azim_rad) * distance
    y = np.cos(elev_rad) * np.sin(azim_rad) * distance
    z = np.sin(elev_rad) * distance
    return {"x": x, "y": y, "z": z}


def get_terrain_figure(xx: np.ndarray, yy: np.ndarray, zz: np.ndarray) -> Figure:
    """Get the terrain figure using Plotly."""
    fig = graph_objects.Figure(data=[graph_objects.Surface(x=xx, y=yy, z=zz, showscale=False)])
    fig.update_layout(
        title="Terrain and Path",
        height=800,
        scene={
            "xaxis": {"title": "X", "dtick": 10},
            "yaxis": {"title": "Y", "dtick": 10},
            "zaxis": {"title": "Z", "dtick": 5},
            "aspectmode": "cube",
            "camera_eye": calculate_camera_eye(elev=30, azim=240),
        },
    )
    return fig


def get_fitness_curve_figure(fitness_values: list[float]) -> Figure:
    """Get the fitness curve figure using Plotly."""
    fig = graph_objects.Figure(data=[graph_objects.Scatter(y=fitness_values)])
    fig.update_layout(title="Fitness Curve", xaxis_title="Iteration", yaxis_title="Fitness")
    return fig
