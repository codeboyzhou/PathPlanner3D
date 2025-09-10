import streamlit as st

from pp3d.algorithm.pso.types import AlgorithmArguments


def init_pso_args() -> AlgorithmArguments:
    """Initialize PSO algorithm arguments for the 3D Path Planning Playground.

    Returns:
        AlgorithmArguments: The initialized PSO algorithm arguments for the 3D Path Planning Playground.
    """
    with st.expander(label="PSO Arguments", expanded=True):
        num_particles = st.number_input("Number of Particles", min_value=10, max_value=100, value=50, step=10)
        num_waypoints = st.number_input("Number of Waypoints", min_value=5, max_value=100, value=10, step=5)
        max_iterations = st.number_input("Max Iterations", min_value=10, max_value=1000, value=100, step=10)
        inertia_weight_min = st.number_input("Min Inertia Weight", min_value=0.1, max_value=1.0, value=0.4, step=0.1)
        inertia_weight_max = st.number_input("Max Inertia Weight", min_value=0.1, max_value=1.0, value=0.9, step=0.1)
        cognitive_weight = st.number_input("Cognitive Weight", min_value=0.1, max_value=2.1, value=1.5, step=0.1)
        social_weight = st.number_input("Social Weight", min_value=0.1, max_value=2.1, value=1.5, step=0.1)
        with st.expander(label="Max Velocities", expanded=True):
            max_velocity_x = st.number_input("Max Velocity X", min_value=0.1, max_value=10.0, value=1.0, step=0.1)
            max_velocity_y = st.number_input("Max Velocity Y", min_value=0.1, max_value=10.0, value=1.0, step=0.1)
            max_velocity_z = st.number_input("Max Velocity Z", min_value=0.1, max_value=10.0, value=1.0, step=0.1)
        with st.expander(label="Axes Min", expanded=True):
            axes_min_x = st.number_input("Axis Min X", min_value=-100.0, max_value=100.0, value=-10.0, step=1.0)
            axes_min_y = st.number_input("Axis Min Y", min_value=-100.0, max_value=100.0, value=-10.0, step=1.0)
            axes_min_z = st.number_input("Axis Min Z", min_value=-100.0, max_value=100.0, value=-10.0, step=1.0)
        with st.expander(label="Axes Max", expanded=True):
            axes_max_x = st.number_input("Axis Max X", min_value=-100.0, max_value=100.0, value=10.0, step=1.0)
            axes_max_y = st.number_input("Axis Max Y", min_value=-100.0, max_value=100.0, value=10.0, step=1.0)
            axes_max_z = st.number_input("Axis Max Z", min_value=-100.0, max_value=100.0, value=10.0, step=1.0)
        random_seed = st.number_input(
            "Random Seed (0 means None, for non-deterministic)", min_value=0, max_value=1000, value=0, step=1
        )
        verbose = st.checkbox("Verbose")
        return AlgorithmArguments(
            num_particles=num_particles,
            num_waypoints=num_waypoints,
            max_iterations=max_iterations,
            inertia_weight_min=inertia_weight_min,
            inertia_weight_max=inertia_weight_max,
            cognitive_weight=cognitive_weight,
            social_weight=social_weight,
            max_velocities=(max_velocity_x, max_velocity_y, max_velocity_z),
            axes_min=(axes_min_x, axes_min_y, axes_min_z),
            axes_max=(axes_max_x, axes_max_y, axes_max_z),
            random_seed=random_seed,
            verbose=verbose,
        )
