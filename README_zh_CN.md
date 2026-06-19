简体中文 | [English](README.md)

# PathPlanner3D

PathPlanner3D 是一个面向算法研究与产品实验的地形感知三维路径规划平台。仓库采用 React 前端和 Python
后端，Python 后端使用 `uv workspace` 管理。

## 仓库结构

```text
PathPlanner3D/
├── backend/
│   ├── api/                  FastAPI 接口服务（`pp3d-api`）
│   ├── core/                 GA、PSO、PSO-GA 与数值工具（`pp3d-core`）
│   ├── pyproject.toml        uv workspace 与共享 Python 工具配置
│   └── uv.lock               后端统一依赖锁文件
└── frontend/                 React、Three.js、ECharts、Monaco
```

算法包不依赖 FastAPI、React、Streamlit 或可视化库。API 包通过本地 `uv workspace` 依赖算法包，两者可以
独立开发和测试。

## 后端

环境要求：Python 3.12+、`uv`。

```bash
cd backend
uv sync --all-packages
uv run uvicorn pp3d_api.main:app --reload
```

接口地址为 `http://127.0.0.1:8000`，OpenAPI 文档位于 `http://127.0.0.1:8000/docs`。

主要接口：

- `GET /api/v1/health`
- `GET /api/v1/algorithms`
- `POST /api/v1/planner/run`
- `POST /api/v1/planner/jobs`
- `GET /api/v1/planner/jobs/{job_id}`
- `GET /api/v1/planner/jobs/{job_id}/result`

当前优化器使用 NumPy 全局随机状态，因此规划任务通过专用的单 worker 队列执行。队列最多等待 32 个任务，
队列满时返回 `429`。预估工作量（`粒子数 * 迭代数 * 航点数 * 运行次数 * 算法数`）超过 2,000,000 的请求
返回 `422`。完成和失败的任务保留一小时，内存中最多保留 1,000 条任务记录。

运行后端检查：

```bash
uv run pytest
uv run ruff check .
uv run pyright
```

## 前端

```bash
cd frontend
npm install
npm run dev
```

Vite 会把 `/api` 请求代理到本地 `8000` 端口的 FastAPI 服务。

## 当前状态

后端已经能够真实运行 GA、PSO 和 PSO-GA。React 页面目前默认仍使用本地模拟器，下一步将把运行流程
接入异步规划任务接口。
