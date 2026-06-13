import { CircleStop, Play, RotateCcw, Sparkles } from 'lucide-react'
import type { Algorithm, PlannerParameters, Point3D } from '../types'

interface Props {
  parameters: PlannerParameters
  onChange: (parameters: PlannerParameters) => void
  onAlgorithmChange: (algorithm: Algorithm) => void
  onRun: () => void
  onCompare: () => void
  onReset: () => void
  onStop: () => void
  isRunning: boolean
  progress: number
  estimatedSeconds: number
}

const algorithms: Algorithm[] = ['PSO', 'GA', 'PSO-GA']

function NumberField({
  label,
  value,
  unit,
  min,
  max,
  onChange,
}: {
  label: string
  value: number
  unit?: string
  min?: number
  max?: number
  onChange: (value: number) => void
}) {
  return (
    <div className="field">
      <label>{label}</label>
      <div className="input-shell">
        <input
          type="number"
          value={value}
          min={min}
          max={max}
          onChange={(event) => onChange(Number(event.target.value))}
        />
        {unit && <span>{unit}</span>}
      </div>
    </div>
  )
}

function PointFields({
  label,
  point,
  onChange,
}: {
  label: string
  point: Point3D
  onChange: (point: Point3D) => void
}) {
  return (
    <div className="field field--wide">
      <label>{label}</label>
      <div className="field-grid field-grid--three">
        {(['x', 'y', 'z'] as const).map((axis) => (
          <div className="input-shell" key={axis}>
            <span>{axis.toUpperCase()}</span>
            <input
              aria-label={`${label} ${axis}`}
              type="number"
              value={point[axis]}
              onChange={(event) => onChange({ ...point, [axis]: Number(event.target.value) })}
            />
          </div>
        ))}
      </div>
    </div>
  )
}

function RangeField({
  label,
  value,
  min,
  max,
  step,
  onChange,
}: {
  label: string
  value: number
  min: number
  max: number
  step: number
  onChange: (value: number) => void
}) {
  return (
    <div className="range-row">
      <label>{label}</label>
      <input
        type="range"
        min={min}
        max={max}
        step={step}
        value={value}
        onChange={(event) => onChange(Number(event.target.value))}
      />
      <output>{value.toFixed(step < 1 ? 2 : 0)}</output>
    </div>
  )
}

export function ParameterPanel({
  parameters,
  onChange,
  onAlgorithmChange,
  onRun,
  onCompare,
  onReset,
  onStop,
  isRunning,
  progress,
  estimatedSeconds,
}: Props) {
  const update = <Key extends keyof PlannerParameters>(key: Key, value: PlannerParameters[Key]) => {
    onChange({ ...parameters, [key]: value })
  }

  return (
    <aside className="panel parameters-panel">
      <div className="panel__section">
        <div className="section-heading">
          <h2>优化算法</h2>
          <span>单算法模式</span>
        </div>
        <div className="algorithm-options">
          {algorithms.map((algorithm) => (
            <button
              className={`algorithm-option ${parameters.algorithm === algorithm ? 'active' : ''}`}
              key={algorithm}
              onClick={() => onAlgorithmChange(algorithm)}
            >
              {algorithm}
            </button>
          ))}
        </div>
      </div>

      <div className="panel__section">
        <div className="section-heading"><h2>任务配置</h2></div>
        <div className="field-grid">
          <NumberField label="路径点数量" value={parameters.waypoints} min={2} max={30} onChange={(value) => update('waypoints', value)} />
          <NumberField label="多次运行" value={parameters.multipleRuns} min={1} max={100} onChange={(value) => update('multipleRuns', value)} />
          <PointFields label="起点坐标" point={parameters.start} onChange={(value) => update('start', value)} />
          <PointFields label="终点坐标" point={parameters.end} onChange={(value) => update('end', value)} />
        </div>
      </div>

      <div className="panel__section">
        <div className="section-heading"><h2>搜索规模</h2><span>影响运行时间</span></div>
        <div className="field-grid">
          <NumberField label="粒子 / 种群" value={parameters.particles} min={10} max={500} onChange={(value) => update('particles', value)} />
          <NumberField label="最大迭代" value={parameters.iterations} min={10} max={2000} onChange={(value) => update('iterations', value)} />
          <NumberField label="随机种子" value={parameters.randomSeed} min={0} onChange={(value) => update('randomSeed', value)} />
          <NumberField label="最大速度" value={parameters.maxVelocity} unit="m/s" min={0.1} max={10} onChange={(value) => update('maxVelocity', value)} />
        </div>
      </div>

      <div className="panel__section">
        <div className="section-heading"><h2>PSO 参数</h2><span>动态权重</span></div>
        <RangeField label="惯性" value={parameters.inertia} min={0.1} max={1} step={0.01} onChange={(value) => update('inertia', value)} />
        <RangeField label="认知" value={parameters.cognitive} min={0.1} max={2.5} step={0.05} onChange={(value) => update('cognitive', value)} />
        <RangeField label="社会" value={parameters.social} min={0.1} max={2.5} step={0.05} onChange={(value) => update('social', value)} />
      </div>

      <div className="panel__section">
        <div className="switch-row">
          <div><strong>详细运行日志</strong><span>会增加浏览器消息数量</span></div>
          <button
            aria-label="切换详细日志"
            className={`switch ${parameters.verbose ? 'active' : ''}`}
            onClick={() => update('verbose', !parameters.verbose)}
          />
        </div>
      </div>

      <div className="run-area">
        <div className="button-row">
          <button className={`primary-button ${isRunning ? 'primary-button--stop' : ''}`} onClick={isRunning ? onStop : onRun}>
            {isRunning ? <><CircleStop size={15} /> 停止运行 {progress}%</> : <><Play size={15} /> 运行 {parameters.algorithm}</>}
          </button>
          <button className="reset-button" onClick={onReset} title="重置参数"><RotateCcw size={15} /></button>
        </div>
        {isRunning && <div className="progress-track"><span style={{ width: `${progress}%` }} /></div>}
        <button className="secondary-button" onClick={onCompare} disabled={isRunning}>
          <Sparkles size={14} /> 一键对比全部算法
        </button>
        <p className="run-estimate">
          {isRunning ? '正在生成航迹与收敛数据' : `预计耗时约 ${estimatedSeconds.toFixed(1)} 秒 · ${parameters.particles * parameters.iterations} 次搜索单元`}
        </p>
      </div>
    </aside>
  )
}
