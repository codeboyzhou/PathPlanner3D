export type Algorithm = 'PSO' | 'GA' | 'PSO-GA'
export type WorkspaceTab = 'scene' | 'code'

export interface Point3D {
  x: number
  y: number
  z: number
}

export interface PlannerParameters {
  algorithm: Algorithm
  particles: number
  iterations: number
  waypoints: number
  multipleRuns: number
  randomSeed: number
  start: Point3D
  end: Point3D
  inertia: number
  cognitive: number
  social: number
  maxVelocity: number
  verbose: boolean
}

export interface AlgorithmResult {
  algorithm: Algorithm
  fitness: number
  duration: number
  pathLength: number
  collisions: number
  convergence: number[]
  path: Point3D[]
  color: string
}

export interface RunResult {
  selected: AlgorithmResult
  comparisons: AlgorithmResult[]
  generatedAt: string
}
