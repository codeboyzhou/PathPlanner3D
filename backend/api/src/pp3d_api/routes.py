from typing import Annotated

from fastapi import APIRouter, HTTPException, Query, status

from pp3d_api.models import AlgorithmMetadata, AlgorithmName, JobStatus, PlannerJob, PlannerRequest, PlannerResult
from pp3d_api.planner import (
    PlannerQueueFullError,
    WorkloadLimitError,
    get_job,
    get_result,
    run_planner,
    submit_job,
)

router = APIRouter(prefix="/api/v1")


@router.get("/health")
def health_check() -> dict[str, str]:
    return {"status": "ok"}


@router.get("/algorithms", response_model=list[AlgorithmMetadata])
def list_algorithms() -> list[AlgorithmMetadata]:
    return [
        AlgorithmMetadata(id=AlgorithmName.PSO, name="Particle Swarm Optimization", description="Standard PSO."),
        AlgorithmMetadata(id=AlgorithmName.GA, name="Genetic Algorithm", description="Standard genetic algorithm."),
        AlgorithmMetadata(
            id=AlgorithmName.PSO_GA,
            name="PSO-GA Hybrid",
            description="PSO with adaptive genetic crossover and mutation.",
        ),
    ]


@router.post("/planner/run", response_model=PlannerResult)
def run_planner_synchronously(
    request: PlannerRequest,
    compare: Annotated[bool, Query(description="Run every available algorithm.")] = False,
) -> PlannerResult:
    try:
        return run_planner(request, compare=compare)
    except WorkloadLimitError as exc:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_CONTENT, detail=str(exc)) from exc


@router.post("/planner/jobs", response_model=PlannerJob, status_code=status.HTTP_202_ACCEPTED)
def submit_planner_job(
    request: PlannerRequest,
    compare: Annotated[bool, Query(description="Run every available algorithm.")] = False,
) -> PlannerJob:
    try:
        return submit_job(request, compare)
    except WorkloadLimitError as exc:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_CONTENT, detail=str(exc)) from exc
    except PlannerQueueFullError as exc:
        raise HTTPException(status_code=status.HTTP_429_TOO_MANY_REQUESTS, detail=str(exc)) from exc


@router.get("/planner/jobs/{job_id}", response_model=PlannerJob)
def read_planner_job(job_id: str) -> PlannerJob:
    job = get_job(job_id)
    if job is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Planner job not found.")
    return job


@router.get("/planner/jobs/{job_id}/result", response_model=PlannerResult)
def read_planner_result(job_id: str) -> PlannerResult:
    job = get_job(job_id)
    if job is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Planner job not found.")
    if job.status == JobStatus.FAILED:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT, detail=job.error or "Planner job failed."
        )
    result = get_result(job_id)
    if result is None:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="Planner job is not complete.")
    return result
