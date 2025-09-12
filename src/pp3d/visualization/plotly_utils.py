import numpy as np
from plotly import graph_objects
from plotly.graph_objs import Figure


def get_terrain_figure(xx: np.ndarray, yy: np.ndarray, zz: np.ndarray) -> Figure:
    """Get the terrain figure using Plotly."""
    fig = graph_objects.Figure(data=[graph_objects.Surface(x=xx, y=yy, z=zz)])
    fig.update_layout(title="Terrain and Path")
    return fig


def get_fitness_curve_figure(fitness_values: list[float]) -> Figure:
    """Get the fitness curve figure using Plotly."""
    fig = graph_objects.Figure(data=[graph_objects.Scatter(y=fitness_values)])
    fig.update_layout(title="Fitness Curve", xaxis_title="Iteration", yaxis_title="Fitness")
    return fig
