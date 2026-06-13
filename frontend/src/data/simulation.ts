import type { Algorithm, AlgorithmResult, PlannerParameters, Point3D, RunResult } from '../types'

export const defaultParameters: PlannerParameters = {
  algorithm: 'PSO-GA',
  particles: 50,
  iterations: 300,
  waypoints: 6,
  multipleRuns: 1,
  randomSeed: 42,
  start: { x: 0, y: 0, z: 5 },
  end: { x: 90, y: 90, z: 8 },
  inertia: 0.58,
  cognitive: 1.45,
  social: 1.7,
  maxVelocity: 1,
  verbose: false,
}

const palette: Record<Algorithm, string> = {
  PSO: '#187cba',
  GA: '#e69a38',
  'PSO-GA': '#0b8663',
}

const algorithmFactor: Record<Algorithm, number> = {
  PSO: 1.09,
  GA: 1.16,
  'PSO-GA': 1,
}

function seeded(seed: number) {
  let value = seed || 19
  return () => {
    value = (value * 9301 + 49297) % 233280
    return value / 233280
  }
}

function terrainHeight(x: number, z: number) {
  const peaks = [[20, 20, 12, 10], [20, 70, 16, 11], [62, 22, 14, 10], [62, 68, 19, 13]]
  return peaks.reduce((height, [px, pz, amplitude, radius]) => {
    const distance = (x - px) ** 2 + (z - pz) ** 2
    return height + amplitude * Math.exp(-distance / (2 * radius ** 2))
  }, 0)
}

function createPath(parameters: PlannerParameters, algorithm: Algorithm): Point3D[] {
  const random = seeded(parameters.randomSeed + algorithm.length * 31)
  const points: Point3D[] = [parameters.start]
  const count = parameters.waypoints
  const curveBias = algorithm === 'GA' ? 8 : algorithm === 'PSO' ? -5 : 2

  for (let index = 1; index <= count; index += 1) {
    const t = index / (count + 1)
    const baseX = parameters.start.x + (parameters.end.x - parameters.start.x) * t
    const baseY = parameters.start.y + (parameters.end.y - parameters.start.y) * t
    const bend = Math.sin(t * Math.PI) * curveBias
    const x = baseX + bend + (random() - 0.5) * 4
    const y = baseY - bend + (random() - 0.5) * 4
    const z = Math.max(terrainHeight(x, y) + 5.5, 9 + Math.sin(t * Math.PI) * 8)
    points.push({ x, y, z })
  }
  points.push(parameters.end)
  return points
}

function pathLength(path: Point3D[]) {
  return path.slice(1).reduce((total, point, index) => {
    const previous = path[index]
    return total + Math.hypot(point.x - previous.x, point.y - previous.y, point.z - previous.z)
  }, 0)
}

function createConvergence(parameters: PlannerParameters, algorithm: Algorithm) {
  const random = seeded(parameters.randomSeed + algorithm.charCodeAt(0))
  const length = Math.min(parameters.iterations, 300)
  const factor = algorithmFactor[algorithm]
  return Array.from({ length }, (_, index) => {
    const progress = index / Math.max(1, length - 1)
    const baseline = 760 * Math.exp(-5.1 * progress) + 132 * factor
    const noise = (random() - 0.5) * 18 * (1 - progress)
    return Number((baseline + noise).toFixed(2))
  })
}

function createAlgorithmResult(parameters: PlannerParameters, algorithm: Algorithm): AlgorithmResult {
  const path = createPath(parameters, algorithm)
  const convergence = createConvergence(parameters, algorithm)
  const length = pathLength(path)
  const complexity = parameters.particles * parameters.iterations
  return {
    algorithm,
    path,
    convergence,
    color: palette[algorithm],
    pathLength: Number(length.toFixed(2)),
    fitness: convergence.at(-1) ?? 0,
    collisions: 0,
    duration: Number(((complexity / 7000) * algorithmFactor[algorithm] + 0.65).toFixed(2)),
  }
}

export function createRunResult(parameters: PlannerParameters, compare = false): RunResult {
  const algorithms: Algorithm[] = compare ? ['PSO', 'GA', 'PSO-GA'] : [parameters.algorithm]
  const comparisons = algorithms.map((algorithm) => createAlgorithmResult(parameters, algorithm))
  return {
    selected: comparisons.find((item) => item.algorithm === parameters.algorithm) ?? comparisons[0],
    comparisons,
    generatedAt: new Date().toLocaleTimeString('zh-CN', { hour12: false }),
  }
}

export { terrainHeight }
