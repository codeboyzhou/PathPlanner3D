from datetime import datetime
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field


class AlgorithmName(StrEnum):
    """Algorithms exposed by the planning API."""

    PSO = "PSO"
    GA = "GA"
    PSO_GA = "PSO-GA"


class JobStatus(StrEnum):
    """Planner job lifecycle states."""

    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class Point3D(BaseModel):
    """A point in the planner coordinate system."""

    x: float
    y: float
    z: float


class PlannerRequest(BaseModel):
    """Configuration accepted from the React planning workspace."""

    model_config = ConfigDict(populate_by_name=True)

    algorithm: AlgorithmName = AlgorithmName.PSO_GA
    particles: int = Field(default=50, ge=2, le=1000)
    iterations: int = Field(default=300, ge=1, le=5000)
    waypoints: int = Field(default=6, ge=2, le=50)
    multiple_runs: int = Field(default=1, alias="multipleRuns", ge=1, le=100)
    random_seed: int = Field(default=42, alias="randomSeed", ge=0)
    start: Point3D = Field(default_factory=lambda: Point3D(x=0, y=0, z=5))
    end: Point3D = Field(default_factory=lambda: Point3D(x=90, y=90, z=8))
    inertia: float = Field(default=0.58, ge=0.0, le=2.0)
    cognitive: float = Field(default=1.45, ge=0.0, le=3.0)
    social: float = Field(default=1.7, ge=0.0, le=3.0)
    max_velocity: float = Field(default=1.0, alias="maxVelocity", gt=0.0, le=20.0)
    verbose: bool = False


class AlgorithmResult(BaseModel):
    """Result for one planning algorithm."""

    algorithm: AlgorithmName
    fitness: float
    duration: float
    path_length: float = Field(serialization_alias="pathLength")
    collisions: int
    convergence: list[float]
    path: list[Point3D]
    color: str


class PlannerResult(BaseModel):
    """Frontend-compatible planner result."""

    selected: AlgorithmResult
    comparisons: list[AlgorithmResult]
    generated_at: datetime = Field(serialization_alias="generatedAt")


class PlannerJob(BaseModel):
    """Public state of an asynchronous planning job."""

    id: str
    status: JobStatus
    created_at: datetime = Field(serialization_alias="createdAt")
    updated_at: datetime = Field(serialization_alias="updatedAt")
    error: str | None = None


class AlgorithmMetadata(BaseModel):
    """Description of an available planner."""

    id: AlgorithmName
    name: str
    description: str
