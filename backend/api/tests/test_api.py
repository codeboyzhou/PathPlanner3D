import asyncio
from collections.abc import Iterator
from queue import Queue

import httpx
import pp3d_api.planner as planner
import pytest
from pp3d_api.main import app
from pp3d_api.planner import PlannerTask, start_worker, stop_worker

pytestmark = pytest.mark.anyio


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


@pytest.fixture(scope="session", autouse=True)
def planner_worker() -> Iterator[None]:
    start_worker()
    yield
    stop_worker()


@pytest.fixture
async def client() -> httpx.AsyncClient:
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as async_client:
        yield async_client


async def test_health_check(client: httpx.AsyncClient) -> None:
    response = await client.get("/api/v1/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


async def test_list_algorithms(client: httpx.AsyncClient) -> None:
    response = await client.get("/api/v1/algorithms")
    assert response.status_code == 200
    assert [algorithm["id"] for algorithm in response.json()] == ["PSO", "GA", "PSO-GA"]


async def test_run_small_pso_job(client: httpx.AsyncClient) -> None:
    response = await client.post(
        "/api/v1/planner/run",
        json={
            "algorithm": "PSO",
            "particles": 4,
            "iterations": 2,
            "waypoints": 2,
            "randomSeed": 42,
            "start": {"x": 0, "y": 0, "z": 5},
            "end": {"x": 20, "y": 20, "z": 8},
        },
    )
    assert response.status_code == 200
    result = response.json()
    assert result["selected"]["algorithm"] == "PSO"
    assert "generatedAt" in result
    assert "pathLength" in result["selected"]
    assert len(result["selected"]["path"]) == 4
    assert len(result["selected"]["convergence"]) == 2


async def test_zero_random_seed_is_reproducible(client: httpx.AsyncClient) -> None:
    payload = {
        "algorithm": "PSO",
        "particles": 4,
        "iterations": 2,
        "waypoints": 2,
        "randomSeed": 0,
    }
    first = await client.post("/api/v1/planner/run", json=payload)
    second = await client.post("/api/v1/planner/run", json=payload)

    assert first.status_code == 200
    assert second.status_code == 200
    assert first.json()["selected"]["fitness"] == second.json()["selected"]["fitness"]


async def test_rejects_workload_above_budget(client: httpx.AsyncClient) -> None:
    response = await client.post(
        "/api/v1/planner/jobs?compare=true",
        json={
            "particles": 1000,
            "iterations": 5000,
            "waypoints": 50,
            "multipleRuns": 100,
        },
    )

    assert response.status_code == 422
    assert "exceeds the limit" in response.json()["detail"]


async def test_returns_too_many_requests_when_queue_is_full(
    client: httpx.AsyncClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    stop_worker()
    try:
        full_queue: Queue[PlannerTask] = Queue(maxsize=1)
        request = planner.PlannerRequest()
        full_queue.put_nowait(PlannerTask(job_id="occupied", request=request, compare=False))
        with monkeypatch.context() as context:
            context.setattr(planner, "_task_queue", full_queue)
            response = await client.post("/api/v1/planner/jobs", json={})
    finally:
        start_worker()

    assert response.status_code == 429


async def test_submit_and_read_job(client: httpx.AsyncClient, monkeypatch: pytest.MonkeyPatch) -> None:
    response = await client.post(
        "/api/v1/planner/jobs",
        json={
            "algorithm": "GA",
            "particles": 4,
            "iterations": 2,
            "waypoints": 2,
            "randomSeed": 7,
            "start": {"x": 0, "y": 0, "z": 5},
            "end": {"x": 20, "y": 20, "z": 8},
        },
    )
    assert response.status_code == 202
    job_id = response.json()["id"]

    for _ in range(100):
        job_response = await client.get(f"/api/v1/planner/jobs/{job_id}")
        if job_response.json()["status"] in {"completed", "failed"}:
            break
        await asyncio.sleep(0.01)

    assert job_response.status_code == 200
    assert job_response.json()["status"] == "completed"

    result_response = await client.get(f"/api/v1/planner/jobs/{job_id}/result")
    assert result_response.status_code == 200
    assert result_response.json()["selected"]["algorithm"] == "GA"

    monkeypatch.setattr(planner, "JOB_TTL_SECONDS", 0)
    expired_response = await client.get(f"/api/v1/planner/jobs/{job_id}")
    assert expired_response.status_code == 404
