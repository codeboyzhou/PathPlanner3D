import numpy as np
import streamlit as st
from plotly import graph_objects

from pp3d.common import interpolates
from pp3d.playground.types import AlgorithmIterationResult


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
    terrain = graph_objects.Surface(x=xx, y=yy, z=zz, showscale=False, name="Terrain")

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

    smooth_path_points = interpolates.smooth_path_with_cubic_spline(path_points)
    smooth_x, smooth_y, smooth_z = smooth_path_points[:, 0], smooth_path_points[:, 1], smooth_path_points[:, 2]
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


def plot_terrain_and_multipath(
    xx: np.ndarray,
    yy: np.ndarray,
    zz: np.ndarray,
    start_point: np.ndarray,
    destination: np.ndarray,
    algorithm_iteration_result: AlgorithmIterationResult,
) -> None:
    """Plot the terrain and multipath using Plotly.

    Args:
        xx: X coordinates of the terrain.
        yy: Y coordinates of the terrain.
        zz: Z coordinates of the terrain.
        start_point: Start point of the path.
        destination: Destination of the path.
        algorithm_iteration_result: Algorithm iteration result.
    """
    fig = graph_objects.Figure()

    terrain = graph_objects.Surface(x=xx, y=yy, z=zz, showscale=False, name="Terrain")
    fig.add_trace(terrain)

    start_point_scatter = graph_objects.Scatter3d(
        x=[start_point[0]],
        y=[start_point[1]],
        z=[start_point[2]],
        mode="markers",
        marker={"size": 5, "color": "green"},
        name="Start Point",
    )
    fig.add_trace(start_point_scatter)

    destination_scatter = graph_objects.Scatter3d(
        x=[destination[0]],
        y=[destination[1]],
        z=[destination[2]],
        mode="markers",
        marker={"size": 5, "color": "red"},
        name="Destination",
    )
    fig.add_trace(destination_scatter)

    pso_path_points = algorithm_iteration_result.pso_full_path_points
    pso_smooth_path_points = interpolates.smooth_path_with_cubic_spline(pso_path_points)
    pso_smooth_x, pso_smooth_y, pso_smooth_z = (
        pso_smooth_path_points[:, 0],
        pso_smooth_path_points[:, 1],
        pso_smooth_path_points[:, 2],
    )
    pso_path = graph_objects.Scatter3d(
        x=pso_smooth_x,
        y=pso_smooth_y,
        z=pso_smooth_z,
        mode="lines",
        line={"width": 6, "color": "#ff4848"},
        name="PSO Path",
    )
    fig.add_trace(pso_path)

    ga_path_points = algorithm_iteration_result.ga_full_path_points
    ga_smooth_path_points = interpolates.smooth_path_with_cubic_spline(ga_path_points)
    ga_smooth_x, ga_smooth_y, ga_smooth_z = (
        ga_smooth_path_points[:, 0],
        ga_smooth_path_points[:, 1],
        ga_smooth_path_points[:, 2],
    )
    ga_path = graph_objects.Scatter3d(
        x=ga_smooth_x,
        y=ga_smooth_y,
        z=ga_smooth_z,
        mode="lines",
        line={"width": 6, "color": "springgreen"},
        name="GA Path",
    )
    fig.add_trace(ga_path)

    pso_ga_hybrid_path_points = algorithm_iteration_result.pso_ga_hybrid_full_path_points
    pso_ga_hybrid_smooth_path_points = interpolates.smooth_path_with_cubic_spline(pso_ga_hybrid_path_points)
    pso_ga_hybrid_smooth_x, pso_ga_hybrid_smooth_y, pso_ga_hybrid_smooth_z = (
        pso_ga_hybrid_smooth_path_points[:, 0],
        pso_ga_hybrid_smooth_path_points[:, 1],
        pso_ga_hybrid_smooth_path_points[:, 2],
    )
    pso_ga_hybrid_path = graph_objects.Scatter3d(
        x=pso_ga_hybrid_smooth_x,
        y=pso_ga_hybrid_smooth_y,
        z=pso_ga_hybrid_smooth_z,
        mode="lines",
        line={"width": 6, "color": "yellow"},
        name="PSO-GA Hybrid Path",
    )
    fig.add_trace(pso_ga_hybrid_path)

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


def plot_multiple_fitness_curves(algorithm_iteration_result: AlgorithmIterationResult) -> None:
    """Plot multiple fitness curves using Plotly.

    Args:
        algorithm_iteration_result (AlgorithmIterationResult): Algorithm iteration result.
    """
    fig = graph_objects.Figure()

    pso_best_fitness_values = algorithm_iteration_result.pso_best_fitness_values
    fig.add_trace(graph_objects.Scatter(y=pso_best_fitness_values, name="PSO", line={"color": "#ff4848"}))

    ga_best_fitness_values = algorithm_iteration_result.ga_best_fitness_values
    fig.add_trace(graph_objects.Scatter(y=ga_best_fitness_values, name="GA", line={"color": "springgreen"}))

    pso_ga_hybrid_best_fitness_values = algorithm_iteration_result.pso_ga_hybrid_best_fitness_values
    fig.add_trace(
        graph_objects.Scatter(y=pso_ga_hybrid_best_fitness_values, name="PSO-GA Hybrid", line={"color": "yellow"})
    )

    fig.update_layout(title="Fitness Curve", xaxis_title="Iteration", yaxis_title="Fitness")
    st.plotly_chart(fig, use_container_width=True)
