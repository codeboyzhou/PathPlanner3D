import numpy as np
import streamlit as st
from plotly import graph_objects
from scipy.interpolate import make_interp_spline


def _calculate_camera_eye(elev: float, azim: float, distance: float = 2.5) -> dict:
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


def _make_path_smooth(path_points: np.ndarray):
    """Make the path smooth using B-spline.

    Args:
        path_points: Path points to be smoothed, shape: (n, 3), i.e. [[x1, y1, z1], [x2, y2, z2], ...]
    """
    x = path_points[:, 0]
    y = path_points[:, 1]
    z = path_points[:, 2]

    t = np.linspace(start=0, stop=1, num=len(x))
    x_spline = make_interp_spline(t, x, k=3)
    y_spline = make_interp_spline(t, y, k=3)
    z_spline = make_interp_spline(t, z, k=3)

    t_new = np.linspace(start=t.min(), stop=t.max(), num=100)
    x_smooth = x_spline(t_new)
    y_smooth = y_spline(t_new)
    z_smooth = z_spline(t_new)

    smooth_path_points = np.column_stack((x_smooth, y_smooth, z_smooth))
    return smooth_path_points


def plot_terrain_and_path(
    xx: np.ndarray,
    yy: np.ndarray,
    zz: np.ndarray,
    start_point: np.ndarray,
    destination: np.ndarray,
    path_points: np.ndarray,
) -> None:
    """Plot the terrain and path using Plotly.

    Args:
        xx: X coordinates of the terrain.
        yy: Y coordinates of the terrain.
        zz: Z coordinates of the terrain.
        start_point: Start point of the path.
        destination: Destination of the path.
        path_points: Path points to be plotted, shape: (n, 3), i.e. [[x1, y1, z1], [x2, y2, z2], ...]
    """
    terrain = graph_objects.Surface(x=xx, y=yy, z=zz, showscale=False)

    start_point_scatter = graph_objects.Scatter3d(
        x=[start_point[0]],
        y=[start_point[1]],
        z=[start_point[2]],
        mode="markers",
        marker={"size": 5, "color": "green"},
        name="Start Point",
    )

    destination_scatter = graph_objects.Scatter3d(
        x=[destination[0]],
        y=[destination[1]],
        z=[destination[2]],
        mode="markers",
        marker={"size": 5, "color": "red"},
        name="Destination",
    )

    smooth_path_points = _make_path_smooth(path_points)
    smooth_x = smooth_path_points[:, 0]
    smooth_y = smooth_path_points[:, 1]
    smooth_z = smooth_path_points[:, 2]
    path = graph_objects.Scatter3d(
        x=smooth_x, y=smooth_y, z=smooth_z, mode="lines", line={"width": 6, "color": "springgreen"}, name="Target Path"
    )

    fig = graph_objects.Figure(data=[terrain, path, start_point_scatter, destination_scatter])
    fig.update_layout(
        title="Terrain and Path",
        height=800,
        scene={
            "xaxis": {"title": "X", "dtick": 10},
            "yaxis": {"title": "Y", "dtick": 10},
            "zaxis": {"title": "Z", "dtick": 5},
            "aspectmode": "cube",
            "camera_eye": _calculate_camera_eye(elev=30, azim=240),
        },
    )
    st.plotly_chart(fig, use_container_width=True)


def plot_fitness_curve(fitness_values: list[float]) -> None:
    """Plot the fitness curve using Plotly.

    Args:
        fitness_values (list[float]): Fitness values of each iteration.
    """
    fig = graph_objects.Figure(data=[graph_objects.Scatter(y=fitness_values)])
    fig.update_layout(title="Fitness Curve", xaxis_title="Iteration", yaxis_title="Fitness")
    st.plotly_chart(fig, use_container_width=True)
