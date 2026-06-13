import ReactECharts from 'echarts-for-react'
import { Clock3, Gauge, Route, ShieldCheck } from 'lucide-react'
import type { PlannerParameters, RunResult } from '../types'

interface Props {
  result: RunResult
  parameters: PlannerParameters
  isRunning: boolean
  progress: number
  compareMode: boolean
}

export function MetricsPanel({ result, parameters, isRunning, progress, compareMode }: Props) {
  const selected = result.selected
  const chartResults = compareMode ? result.comparisons : [selected]
  const convergenceOption = {
    animationDuration: 500,
    grid: { left: 42, right: 12, top: 16, bottom: 30 },
    tooltip: { trigger: 'axis', backgroundColor: '#17252b', borderWidth: 0, textStyle: { fontSize: 10 } },
    xAxis: {
      type: 'category',
      data: selected.convergence.map((_, index) => index + 1),
      name: '迭代',
      nameTextStyle: { color: '#89959a', fontSize: 9 },
      axisLabel: { color: '#89959a', fontSize: 8 },
      axisLine: { lineStyle: { color: '#dce2e1' } },
    },
    yAxis: {
      type: 'value',
      axisLabel: { color: '#89959a', fontSize: 8 },
      splitLine: { lineStyle: { color: '#edf0ef' } },
    },
    series: chartResults.map((item) => ({
      name: item.algorithm,
      type: 'line',
      data: item.convergence,
      showSymbol: false,
      smooth: true,
      lineStyle: { width: 2, color: item.color },
      itemStyle: { color: item.color },
    })),
  }

  const comparisonOption = {
    animationDuration: 500,
    grid: { left: 35, right: 10, top: 10, bottom: 24 },
    tooltip: { trigger: 'axis', backgroundColor: '#17252b', borderWidth: 0, textStyle: { fontSize: 10 } },
    xAxis: {
      type: 'category',
      data: result.comparisons.map((item) => item.algorithm),
      axisLabel: { color: '#68767b', fontSize: 9 },
      axisLine: { lineStyle: { color: '#dce2e1' } },
    },
    yAxis: {
      type: 'value',
      axisLabel: { color: '#89959a', fontSize: 8 },
      splitLine: { lineStyle: { color: '#edf0ef' } },
    },
    series: [{
      type: 'bar',
      barWidth: 22,
      data: result.comparisons.map((item) => ({ value: item.fitness, itemStyle: { color: item.color, borderRadius: [3, 3, 0, 0] } })),
    }],
  }

  return (
    <aside className="panel metrics-panel">
      <div className="panel__section">
        <div className="section-heading"><h2>运行概览</h2><span>{result.generatedAt}</span></div>
        <div className={`status-line ${isRunning ? 'running' : ''}`}>
          <i /> {isRunning ? `正在优化第 ${Math.round(parameters.iterations * progress / 100)} / ${parameters.iterations} 代` : '规划完成，路径满足约束'}
        </div>
        <div className="metric-grid" style={{ marginTop: 12 }}>
          <div className="metric">
            <div className="metric__top"><span>最佳适应度</span><Gauge size={13} /></div>
            <strong>{isRunning ? '—' : selected.fitness.toFixed(2)}</strong>
            <small>越低越优</small>
          </div>
          <div className="metric">
            <div className="metric__top"><span>运行时间</span><Clock3 size={13} /></div>
            <strong>{isRunning ? `${(selected.duration * progress / 100).toFixed(1)}` : selected.duration.toFixed(2)}</strong>
            <small>秒</small>
          </div>
          <div className="metric">
            <div className="metric__top"><span>路径长度</span><Route size={13} /></div>
            <strong>{selected.pathLength.toFixed(1)}</strong>
            <small>米</small>
          </div>
          <div className="metric">
            <div className="metric__top"><span>碰撞点</span><ShieldCheck size={13} /></div>
            <strong>{selected.collisions}</strong>
            <small>安全可行</small>
          </div>
        </div>
      </div>

      <div className="panel__section">
        <div className="section-heading"><h2>收敛曲线</h2><span>{parameters.iterations} 次迭代</span></div>
        <ReactECharts option={convergenceOption} className="chart" opts={{ renderer: 'svg' }} />
      </div>

      <div className="panel__section">
        <div className="section-heading"><h2>算法结果</h2><span>{compareMode ? '对比模式' : '当前运行'}</span></div>
        <table className="result-table">
          <thead><tr><th>算法</th><th>适应度</th><th>耗时</th></tr></thead>
          <tbody>
            {result.comparisons.map((item) => (
              <tr key={item.algorithm}>
                <td><span className="algorithm-label"><i style={{ background: item.color }} />{item.algorithm}</span></td>
                <td>{item.fitness.toFixed(1)}</td>
                <td>{item.duration.toFixed(2)}s</td>
              </tr>
            ))}
          </tbody>
        </table>
        {compareMode && <ReactECharts option={comparisonOption} className="chart chart--small" opts={{ renderer: 'svg' }} />}
      </div>
    </aside>
  )
}
