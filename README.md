[简体中文](README_zh_CN.md) | English

# PathPlanner3D

PathPlanner3D is a terrain-aware 3D path-planning platform for algorithm research and product experimentation. The
repository uses a React frontend and a Python backend organized as a `uv` workspace.

## Repository Structure

```text
PathPlanner3D/
├── backend/
│   ├── api/                  FastAPI HTTP service (`pp3d-api`)
│   ├── algorithm/            GA, PSO, PSO-GA and numerical utilities (`pp3d-core`)
│   ├── pyproject.toml        uv workspace and shared Python tooling
│   └── uv.lock               reproducible backend dependency lock
└── frontend/                 React, Three.js, ECharts and Monaco
```

The algorithm package has no dependency on FastAPI, React, Streamlit or visualization libraries. The API package
depends on the algorithm package through the local `uv` workspace.

## Backend

Requirements: Python 3.12+ and `uv`.

```bash
cd backend
uv sync --all-packages
uv run uvicorn pp3d_api.main:app --reload
```

The API is available at `http://127.0.0.1:8000`, with OpenAPI documentation at
`http://127.0.0.1:8000/docs`.

Main endpoints:

- `GET /api/v1/health`
- `GET /api/v1/algorithms`
- `POST /api/v1/planner/run`
- `POST /api/v1/planner/jobs`
- `GET /api/v1/planner/jobs/{job_id}`
- `GET /api/v1/planner/jobs/{job_id}/result`

Planner jobs run on a dedicated single-worker queue because the current optimizers use NumPy's global random state.
The queue accepts up to 32 waiting jobs and returns `429` when full. Requests whose estimated work
(`particles * iterations * waypoints * runs * algorithms`) exceeds 2,000,000 units return `422`. Completed and failed
jobs are retained for one hour, with at most 1,000 job records kept in memory.

Run backend checks:

```bash
uv run pytest
uv run ruff check .
uv run pyright
```

## Frontend

```bash
cd frontend
npm install
npm run dev
```

Vite proxies `/api` requests to the local FastAPI server on port `8000`.

## Current Status

The backend now exposes real GA, PSO and PSO-GA execution. The React UI still uses its local simulator by default;
connecting its run workflow to the asynchronous planner job endpoints is the next integration step.
