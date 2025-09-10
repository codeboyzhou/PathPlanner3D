import numpy as np
from pydantic import BaseModel


class AlgorithmArguments(BaseModel):
    """A class for PSO algorithm arguments."""

    num_particles: int
    """The number of particles in the swarm."""

    num_waypoints: int
    """The number of waypoints in the path."""

    max_iterations: int
    """The maximum number of iterations."""

    inertia_weight_min: float
    """The minimum inertia weight."""

    inertia_weight_max: float
    """The maximum inertia weight."""

    cognitive_weight: float
    """The cognitive weight."""

    social_weight: float
    """The social weight."""

    max_velocities: tuple[float, float, float]
    """The maximum velocities of particle, corresponding to the axis x, y and z."""

    axes_min: tuple[float, float, float]
    """The minimum value of axis, which is the lower bound of the search space, corresponding to the axis x, y and z."""

    axes_max: tuple[float, float, float]
    """The maximum value of axis, which is the upper bound of the search space, corresponding to the axis x, y and z."""

    random_seed: int | None = None
    """Random seed for reproducible results. If `None`, results will be non-deterministic. Default is `None`."""

    verbose: bool = False
    """Whether to print the progress of the algorithm. Default is `False`."""


class Particle(BaseModel):
    """A class for a particle in the swarm."""

    position: np.ndarray
    """The position of the particle.
    
    A one-dimensional array, in the form of [x1, y1, z1, x2, y2, z2, ...], is used to represent the coordinates of all
    waypoints on the target path. The use of a one-dimensional array aims to reduce code complexity and better adapt to
    the PSO position update formula.
    """

    velocity: np.ndarray
    """The velocity of the particle.
    
    A one-dimensional array, in the form of [vx1, vy1, vz1, vx2, vy2, vz2, ...], is used to represent the velocities
    of all waypoints on the target path. The use of a one-dimensional array aims to reduce code complexity and better
    adapt to the PSO velocity update formula.
    """

    best_position: np.ndarray
    """The best position of the particle. It has the same shape as `position`."""

    best_fitness_value: float
    """The best fitness value of the particle."""
