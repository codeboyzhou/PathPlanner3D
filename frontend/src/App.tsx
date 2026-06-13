import { lazy, Suspense, useEffect, useMemo, useRef, useState } from 'react'
import {
  Activity,
  ChevronDown,
  Code2,
  Compass,
  Languages,
  Map,
  Menu,
  PanelLeftClose,
  PanelLeftOpen,
  Waypoints,
  X,
} from 'lucide-react'
import './App.css'
import { ParameterPanel } from './components/ParameterPanel'
import { defaultFitnessCode, defaultTerrainCode } from './data/codeTemplates'
import { createRunResult, defaultParameters } from './data/simulation'
import type { Algorithm, PlannerParameters, RunResult, WorkspaceTab } from './types'

const TerrainScene = lazy(() =>
  import('./components/TerrainScene').then((module) => ({ default: module.TerrainScene })),
)
const MetricsPanel = lazy(() =>
  import('./components/MetricsPanel').then((module) => ({ default: module.MetricsPanel })),
)
const CodeWorkspace = lazy(() =>
  import('./components/CodeWorkspace').then((module) => ({ default: module.CodeWorkspace })),
)

const navItems = [
  { id: 'workspace', label: '规划工作台', icon: Compass },
  { id: 'experiments', label: '实验对比', icon: Activity },
  { id: 'terrain', label: '地形资产', icon: Map },
]

function App() {
  const [parameters, setParameters] = useState<PlannerParameters>(defaultParameters)
  const [result, setResult] = useState<RunResult>(() => createRunResult(defaultParameters))
  const [isRunning, setIsRunning] = useState(false)
  const [progress, setProgress] = useState(0)
  const [activeTab, setActiveTab] = useState<WorkspaceTab>('scene')
  const [terrainCode, setTerrainCode] = useState(defaultTerrainCode)
  const [fitnessCode, setFitnessCode] = useState(defaultFitnessCode)
  const [sidebarOpen, setSidebarOpen] = useState(true)
  const [mobileMenuOpen, setMobileMenuOpen] = useState(false)
  const [compareMode, setCompareMode] = useState(false)
  const timerRef = useRef<number | null>(null)

  const estimatedSeconds = useMemo(
    () => Math.max(1.2, (parameters.iterations * parameters.particles) / 4300),
    [parameters.iterations, parameters.particles],
  )

  useEffect(() => {
    return () => {
      if (timerRef.current) window.clearInterval(timerRef.current)
    }
  }, [])

  const stopRun = () => {
    if (timerRef.current) window.clearInterval(timerRef.current)
    timerRef.current = null
    setIsRunning(false)
  }

  const runPlanner = (compare = false) => {
    stopRun()
    setCompareMode(compare)
    setIsRunning(true)
    setProgress(0)
    setActiveTab('scene')
    const startedAt = performance.now()
    const duration = Math.min(4200, Math.max(1400, estimatedSeconds * 680))

    timerRef.current = window.setInterval(() => {
      const elapsed = performance.now() - startedAt
      const nextProgress = Math.min(100, Math.round((elapsed / duration) * 100))
      setProgress(nextProgress)
      if (nextProgress >= 100) {
        stopRun()
        setResult(createRunResult(parameters, compare))
      }
    }, 60)
  }

  const resetParameters = () => {
    stopRun()
    setProgress(0)
    setParameters(defaultParameters)
    setResult(createRunResult(defaultParameters))
    setCompareMode(false)
  }

  const updateAlgorithm = (algorithm: Algorithm) => {
    setParameters((current) => ({ ...current, algorithm }))
  }

  return (
    <div className="app-shell">
      <aside className={`navigation ${sidebarOpen ? '' : 'navigation--collapsed'}`}>
        <div className="brand">
          <div className="brand__mark"><Waypoints size={21} /></div>
          {sidebarOpen && (
            <div className="brand__copy">
              <strong>PathPlanner</strong>
              <span>3D Research Studio</span>
            </div>
          )}
        </div>

        <nav className="nav-list" aria-label="主导航">
          {navItems.map(({ id, label, icon: Icon }, index) => (
            <button className={`nav-item ${index === 0 ? 'nav-item--active' : ''}`} key={id} title={label}>
              <Icon size={18} />
              {sidebarOpen && <span>{label}</span>}
            </button>
          ))}
        </nav>

        <div className="navigation__footer">
          <a className="nav-item" href="https://github.com/codeboyzhou/PathPlanner3D" target="_blank" rel="noreferrer">
            <Code2 size={18} />
            {sidebarOpen && <span>GitHub</span>}
          </a>
          <button className="nav-collapse" onClick={() => setSidebarOpen((value) => !value)}>
            {sidebarOpen ? <PanelLeftClose size={17} /> : <PanelLeftOpen size={17} />}
          </button>
        </div>
      </aside>

      <main className="main">
        <header className="topbar">
          <div className="topbar__title">
            <button className="icon-button mobile-only" onClick={() => setMobileMenuOpen(true)} aria-label="打开菜单">
              <Menu size={19} />
            </button>
            <div>
              <h1>三维路径规划工作台</h1>
              <p>复杂地形下的航迹生成、约束验证与算法对比</p>
            </div>
          </div>
          <div className="topbar__actions">
            <button className="quiet-button"><Languages size={16} /> 中文 <ChevronDown size={14} /></button>
            <div className="connection"><span /> 本地模拟器</div>
          </div>
        </header>

        <div className="workspace">
          <ParameterPanel
            parameters={parameters}
            onChange={setParameters}
            onAlgorithmChange={updateAlgorithm}
            onRun={() => runPlanner(false)}
            onCompare={() => runPlanner(true)}
            onReset={resetParameters}
            onStop={stopRun}
            isRunning={isRunning}
            progress={progress}
            estimatedSeconds={estimatedSeconds}
          />

          <section className="canvas-column">
            <div className="view-tabs">
              <div className="segmented-control">
                <button className={activeTab === 'scene' ? 'active' : ''} onClick={() => setActiveTab('scene')}>
                  <Compass size={16} /> 3D 场景
                </button>
                <button className={activeTab === 'code' ? 'active' : ''} onClick={() => setActiveTab('code')}>
                  <Code2 size={16} /> 模型代码
                </button>
              </div>
              <div className="view-meta">
                <span>100 × 100 m</span>
                <span>{parameters.waypoints + 2} 个路径节点</span>
              </div>
            </div>

            <div className="primary-view">
              {activeTab === 'scene' ? (
                <Suspense fallback={<ViewFallback label="正在加载三维场景" />}>
                  <TerrainScene
                    result={result}
                    parameters={parameters}
                    progress={progress}
                    isRunning={isRunning}
                    compareMode={compareMode}
                  />
                </Suspense>
              ) : (
                <Suspense fallback={<ViewFallback label="正在加载代码编辑器" />}>
                  <CodeWorkspace
                    terrainCode={terrainCode}
                    fitnessCode={fitnessCode}
                    onTerrainCodeChange={setTerrainCode}
                    onFitnessCodeChange={setFitnessCode}
                    onApply={() => runPlanner(false)}
                  />
                </Suspense>
              )}
            </div>

            <div className="scene-footer">
              <div className="legend">
                <span><i className="legend-dot legend-dot--start" /> 起点</span>
                <span><i className="legend-dot legend-dot--end" /> 终点</span>
                <span><i className="legend-line legend-line--primary" /> {parameters.algorithm}</span>
                {compareMode && <span><i className="legend-line legend-line--compare" /> 算法对比</span>}
              </div>
              <span className="scene-hint">拖动旋转 · 滚轮缩放 · 双击重置视角</span>
            </div>
          </section>

          <Suspense fallback={<MetricsFallback />}>
            <MetricsPanel
              result={result}
              parameters={parameters}
              isRunning={isRunning}
              progress={progress}
              compareMode={compareMode}
            />
          </Suspense>
        </div>
      </main>

      {mobileMenuOpen && (
        <div className="mobile-drawer" role="dialog" aria-modal="true">
          <button className="mobile-drawer__backdrop" onClick={() => setMobileMenuOpen(false)} aria-label="关闭菜单" />
          <div className="mobile-drawer__panel">
            <div className="brand">
              <div className="brand__mark"><Waypoints size={21} /></div>
              <div className="brand__copy"><strong>PathPlanner</strong><span>3D Research Studio</span></div>
              <button className="icon-button" onClick={() => setMobileMenuOpen(false)}><X size={18} /></button>
            </div>
            {navItems.map(({ id, label, icon: Icon }, index) => (
              <button className={`nav-item ${index === 0 ? 'nav-item--active' : ''}`} key={id}>
                <Icon size={18} /><span>{label}</span>
              </button>
            ))}
          </div>
        </div>
      )}

      <div className="sr-only" aria-live="polite">
        {isRunning ? `算法运行中，进度 ${progress}%` : '算法运行完成'}
      </div>
    </div>
  )
}

function ViewFallback({ label }: { label: string }) {
  return (
    <div className="view-fallback" role="status">
      <span className="loading-spinner" />
      <strong>{label}</strong>
    </div>
  )
}

function MetricsFallback() {
  return (
    <aside className="panel metrics-panel metrics-fallback" aria-label="正在加载结果面板">
      <div className="loading-block loading-block--title" />
      <div className="loading-grid">
        {Array.from({ length: 4 }, (_, index) => <div className="loading-block loading-block--metric" key={index} />)}
      </div>
      <div className="loading-block loading-block--chart" />
    </aside>
  )
}

export default App
