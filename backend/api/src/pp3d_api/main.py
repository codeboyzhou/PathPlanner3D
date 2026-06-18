import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from pp3d_api.planner import start_worker, stop_worker
from pp3d_api.routes import router


@asynccontextmanager
async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
    start_worker()
    try:
        yield
    finally:
        stop_worker()


def create_app() -> FastAPI:
    app = FastAPI(
        title="PathPlanner3D API",
        version="0.3.0",
        description="HTTP API for terrain-aware 3D path planning.",
        lifespan=lifespan,
    )
    frontend_origins = os.getenv(
        "PP3D_FRONTEND_ORIGINS",
        "http://localhost:5173,http://127.0.0.1:5173,http://localhost:4173,http://127.0.0.1:4173",
    ).split(",")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[origin.strip() for origin in frontend_origins if origin.strip()],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.include_router(router)
    return app


app = create_app()
