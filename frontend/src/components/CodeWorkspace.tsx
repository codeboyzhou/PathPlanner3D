import Editor from '@monaco-editor/react'
import { useState } from 'react'
import { Braces, Play } from 'lucide-react'

interface Props {
  terrainCode: string
  fitnessCode: string
  onTerrainCodeChange: (value: string) => void
  onFitnessCodeChange: (value: string) => void
  onApply: () => void
}

export function CodeWorkspace({
  terrainCode,
  fitnessCode,
  onTerrainCodeChange,
  onFitnessCodeChange,
  onApply,
}: Props) {
  const [active, setActive] = useState<'terrain' | 'fitness'>('terrain')
  const value = active === 'terrain' ? terrainCode : fitnessCode

  return (
    <div className="code-workspace">
      <div className="code-toolbar">
        <div className="code-tabs">
          <button className={active === 'terrain' ? 'active' : ''} onClick={() => setActive('terrain')}>
            地形生成函数
          </button>
          <button className={active === 'fitness' ? 'active' : ''} onClick={() => setActive('fitness')}>
            适应度函数
          </button>
        </div>
        <div className="code-actions">
          <span className="code-status"><Braces size={12} /> Python</span>
          <button className="code-apply" onClick={onApply}><Play size={11} /> 应用并运行</button>
        </div>
      </div>
      <div className="editor-frame">
        <Editor
          height="100%"
          language="python"
          value={value}
          onChange={(nextValue) => {
            const safeValue = nextValue ?? ''
            if (active === 'terrain') onTerrainCodeChange(safeValue)
            else onFitnessCodeChange(safeValue)
          }}
          theme="vs"
          options={{
            minimap: { enabled: false },
            fontSize: 12,
            lineHeight: 20,
            padding: { top: 14 },
            scrollBeyondLastLine: false,
            renderLineHighlight: 'line',
            overviewRulerBorder: false,
            automaticLayout: true,
            tabSize: 4,
          }}
        />
      </div>
    </div>
  )
}
