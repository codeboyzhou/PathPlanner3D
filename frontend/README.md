# PathPlanner3D Frontend

独立于 Streamlit 的 React 前端工作台。

## 技术栈

- React 19 + TypeScript + Vite
- Three.js + React Three Fiber
- Apache ECharts
- Monaco Editor
- Lucide Icons

## 本地运行

```bash
npm install
npm run dev
```

默认访问地址为 `http://localhost:5173`。

## 当前能力

- PSO、GA、PSO-GA 参数配置
- 起终点、航点与搜索规模设置
- 三维地形、平滑航迹及多算法路径展示
- 模拟运行进度、关键指标和收敛曲线
- 多算法结果对比
- 地形生成函数与适应度函数编辑
- 桌面、平板和移动端响应式布局

当前使用浏览器端模拟数据完成界面交互。后续 FastAPI 接入时，可将模拟运行器替换为 REST 任务创建接口，并通过 WebSocket 或 SSE 接收迭代进度。
