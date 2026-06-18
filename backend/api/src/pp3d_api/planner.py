import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from queue import Empty, Full, Queue
from uuid import uuid4

import numpy as np
from pp3d.algorithm.genetic.genetic import GeneticAlgorithm
from pp3d.algorithm.genetic.types import GeneticAlgorithmArguments
from pp3d.algorithm.hybrid.pso_ga_hybrid import HybridPSOAlgorithm
from pp3d.algorithm.hybrid.pso_types import HybridPSOAlgorithmArguments
from pp3d.algorithm.pso.pso import PSOAlgorithm
from pp3d.algorithm.pso.types import PSOAlgorithmArguments
from pp3d.common.flight_angle_calculator import calculate_slope_angles_batch
from pp3d.common.interpolate import smooth_path_with_cubic_spline

from pp3d_api.models import (
    AlgorithmName,
    AlgorithmResult,
    JobStatus,
    PlannerJob,
    PlannerRequest,
    PlannerResult,
    Point3D,
)

ALGORITHM_COLORS = {
    AlgorithmName.PSO: "#187cba",
    AlgorithmName.GA: "#e69a38",
    AlgorithmName.PSO_GA: "#0b8663",
}

MAX_WORK_UNITS = 2_000_000
MAX_QUEUED_JOBS = 32
MAX_RETAINED_JOBS = 1_000
JOB_TTL_SECONDS = 60 * 60


class WorkloadLimitError(ValueError):
    """Raised when a planner request exceeds the configured work budget."""


class PlannerQueueFullError(RuntimeError):
    """Raised when the planner queue cannot accept more work."""


@dataclass(frozen=True, slots=True)
class PlannerTask:
    job_id: str
    request: PlannerRequest
    compare: bool


_jobs: dict[str, PlannerJob] = {}
_results: dict[str, PlannerResult] = {}
_jobs_lock = threading.Lock()
# Existing optimizers seed NumPy's global RNG, so executions must be serialized.
_planner_lock = threading.Lock()
_task_queue: Queue[PlannerTask] = Queue(maxsize=MAX_QUEUED_JOBS)
_worker_lock = threading.Lock()
_worker_stop = threading.Event()
_worker_thread: threading.Thread | None = None


def _as_array(point: Point3D) -> np.ndarray:
    return np.array([point.x, point.y, point.z], dtype=float)


def _terrain_height(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    peaks = (
        (20.0, 20.0, 12.0, 10.0),
        (20.0, 70.0, 16.0, 11.0),
        (62.0, 22.0, 14.0, 10.0),
        (62.0, 68.0, 19.0, 13.0),
    )
    height = np.zeros_like(x, dtype=float)
    for center_x, center_y, amplitude, radius in peaks:
        distance = (x - center_x) ** 2 + (y - center_y) ** 2
        height += amplitude * np.exp(-distance / (2 * radius**2))
    return height


def _collision_count(path: np.ndarray) -> int:
    terrain_clearance = path[:, 2] - _terrain_height(path[:, 0], path[:, 1])
    return int(np.count_nonzero(terrain_clearance < 1.0))


def _create_fitness_function(request: PlannerRequest) -> Callable[[np.ndarray], float]:
    start = _as_array(request.start)
    end = _as_array(request.end)

    def fitness(path_points: np.ndarray) -> float:
        waypoints = path_points.reshape(-1, 3)
        smooth_path = smooth_path_with_cubic_spline(np.vstack((start, waypoints, end)), num_interpolated_points=80)
        path_diff = np.diff(smooth_path, axis=0)
        path_length = np.linalg.norm(path_diff, axis=1).sum()
        height_change = np.abs(path_diff[:, 2]).sum()
        collision_count = _collision_count(smooth_path)
        slope_angles = calculate_slope_angles_batch(smooth_path[:-1], smooth_path[1:])
        steep_segment_count = np.count_nonzero(slope_angles > 45.0)
        return float(path_length + height_change + collision_count * 10_000 + steep_segment_count * 25)

    return fitness


def _axes(request: PlannerRequest) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    margin = 10.0
    axes_min = (
        min(request.start.x, request.end.x, 0.0) - margin,
        min(request.start.y, request.end.y, 0.0) - margin,
        0.0,
    )
    axes_max = (
        max(request.start.x, request.end.x, 100.0) + margin,
        max(request.start.y, request.end.y, 100.0) + margin,
        max(request.start.z, request.end.z, 50.0) + margin,
    )
    return axes_min, axes_max


def _run_algorithm(request: PlannerRequest, algorithm: AlgorithmName) -> AlgorithmResult:
    axes_min, axes_max = _axes(request)
    fitness_function = _create_fitness_function(request)
    common = {
        "num_waypoints": request.waypoints,
        "max_iterations": request.iterations,
        "axes_min": axes_min,
        "axes_max": axes_max,
        "random_seed": request.random_seed,
        "verbose": request.verbose,
    }
    max_velocities = (request.max_velocity, request.max_velocity, request.max_velocity)
    started_at = time.perf_counter()

    if algorithm == AlgorithmName.PSO:
        planner = PSOAlgorithm(
            PSOAlgorithmArguments(
                num_particles=request.particles,
                inertia_weight=request.inertia,
                cognitive_weight=request.cognitive,
                social_weight=request.social,
                max_velocities=max_velocities,
                **common,
            ),
            fitness_function,
        )
    elif algorithm == AlgorithmName.GA:
        planner = GeneticAlgorithm(
            GeneticAlgorithmArguments(
                population_size=request.particles,
                tournament_size=min(3, request.particles),
                crossover_rate=0.8,
                mutation_rate=0.2,
                **common,
            ),
            fitness_function,
        )
    else:
        planner = HybridPSOAlgorithm(
            HybridPSOAlgorithmArguments(
                num_particles=request.particles,
                inertia_weight_min=max(0.1, request.inertia - 0.2),
                inertia_weight_max=min(2.0, request.inertia + 0.3),
                cognitive_weight_min=max(0.1, request.cognitive - 0.7),
                cognitive_weight_max=min(3.0, request.cognitive + 0.7),
                social_weight_min=max(0.1, request.social - 0.7),
                social_weight_max=min(3.0, request.social + 0.7),
                max_velocities=max_velocities,
                **common,
            ),
            fitness_function,
        )

    waypoints, convergence = planner.run()
    full_path = np.vstack((_as_array(request.start), waypoints, _as_array(request.end)))
    smooth_path = smooth_path_with_cubic_spline(full_path, num_interpolated_points=80)
    path_length = float(np.linalg.norm(np.diff(full_path, axis=0), axis=1).sum())
    duration = time.perf_counter() - started_at
    path = [Point3D(x=float(point[0]), y=float(point[1]), z=float(point[2])) for point in full_path]
    return AlgorithmResult(
        algorithm=algorithm,
        fitness=float(convergence[-1]),
        duration=duration,
        path_length=path_length,
        collisions=_collision_count(smooth_path),
        convergence=[float(value) for value in convergence],
        path=path,
        color=ALGORITHM_COLORS[algorithm],
    )


def run_planner(request: PlannerRequest, compare: bool = False) -> PlannerResult:
    """Run one or all planners and return a frontend-compatible result."""
    validate_workload(request, compare)
    algorithms = list(AlgorithmName) if compare else [request.algorithm]
    with _planner_lock:
        comparisons = []
        for algorithm in algorithms:
            samples = []
            for run_index in range(request.multiple_runs):
                seed = request.random_seed + run_index
                samples.append(_run_algorithm(request.model_copy(update={"random_seed": seed}), algorithm))
            best_sample = min(samples, key=lambda sample: sample.fitness)
            total_duration = sum(sample.duration for sample in samples)
            comparisons.append(best_sample.model_copy(update={"duration": total_duration}))
    selected = next((result for result in comparisons if result.algorithm == request.algorithm), comparisons[0])
    return PlannerResult(selected=selected, comparisons=comparisons, generated_at=datetime.now(UTC))


def estimate_work_units(request: PlannerRequest, compare: bool = False) -> int:
    algorithm_count = len(AlgorithmName) if compare else 1
    return request.particles * request.iterations * request.waypoints * request.multiple_runs * algorithm_count


def validate_workload(request: PlannerRequest, compare: bool = False) -> None:
    work_units = estimate_work_units(request, compare)
    if work_units > MAX_WORK_UNITS:
        raise WorkloadLimitError(f"Planner workload {work_units:,} exceeds the limit of {MAX_WORK_UNITS:,}.")


def _prune_jobs_locked(now: datetime) -> None:
    terminal_statuses = {JobStatus.COMPLETED, JobStatus.FAILED}
    expired_ids = [
        job_id
        for job_id, job in _jobs.items()
        if job.status in terminal_statuses and (now - job.updated_at).total_seconds() >= JOB_TTL_SECONDS
    ]
    for job_id in expired_ids:
        _jobs.pop(job_id, None)
        _results.pop(job_id, None)

    overflow = len(_jobs) - MAX_RETAINED_JOBS
    if overflow <= 0:
        return
    oldest_terminal_ids = [
        job.id for job in sorted(_jobs.values(), key=lambda item: item.updated_at) if job.status in terminal_statuses
    ]
    for job_id in oldest_terminal_ids[:overflow]:
        _jobs.pop(job_id, None)
        _results.pop(job_id, None)


def submit_job(request: PlannerRequest, compare: bool = False) -> PlannerJob:
    validate_workload(request, compare)
    now = datetime.now(UTC)
    job = PlannerJob(id=uuid4().hex, status=JobStatus.QUEUED, created_at=now, updated_at=now)
    with _jobs_lock:
        _prune_jobs_locked(now)
        try:
            _task_queue.put_nowait(PlannerTask(job_id=job.id, request=request, compare=compare))
        except Full as exc:
            raise PlannerQueueFullError("Planner queue is full.") from exc
        _jobs[job.id] = job
        _prune_jobs_locked(now)
    return job


def get_job(job_id: str) -> PlannerJob | None:
    with _jobs_lock:
        _prune_jobs_locked(datetime.now(UTC))
        return _jobs.get(job_id)


def get_result(job_id: str) -> PlannerResult | None:
    with _jobs_lock:
        _prune_jobs_locked(datetime.now(UTC))
        return _results.get(job_id)


def execute_job(job_id: str, request: PlannerRequest, compare: bool = False) -> None:
    with _jobs_lock:
        job = _jobs[job_id]
        _jobs[job_id] = job.model_copy(update={"status": JobStatus.RUNNING, "updated_at": datetime.now(UTC)})
    try:
        result = run_planner(request, compare=compare)
    except Exception as exc:
        with _jobs_lock:
            job = _jobs[job_id]
            _jobs[job_id] = job.model_copy(
                update={"status": JobStatus.FAILED, "updated_at": datetime.now(UTC), "error": str(exc)}
            )
        return

    with _jobs_lock:
        _results[job_id] = result
        job = _jobs[job_id]
        _jobs[job_id] = job.model_copy(update={"status": JobStatus.COMPLETED, "updated_at": datetime.now(UTC)})


def _planner_worker() -> None:
    while not _worker_stop.is_set():
        try:
            task = _task_queue.get(timeout=0.2)
        except Empty:
            continue
        try:
            execute_job(task.job_id, task.request, task.compare)
        finally:
            _task_queue.task_done()


def start_worker() -> None:
    global _worker_thread
    with _worker_lock:
        if _worker_thread is not None and _worker_thread.is_alive():
            return
        _worker_stop.clear()
        _worker_thread = threading.Thread(target=_planner_worker, name="pp3d-planner", daemon=True)
        _worker_thread.start()


def stop_worker() -> None:
    global _worker_thread
    with _worker_lock:
        worker = _worker_thread
        if worker is None:
            return
        _worker_stop.set()
        worker.join(timeout=5)
        if not worker.is_alive():
            _worker_thread = None
