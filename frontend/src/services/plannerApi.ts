import type { PlannerParameters, RunResult } from '../types'

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL ?? '/api/v1'

export interface PlannerJob {
  id: string
  status: 'queued' | 'running' | 'completed' | 'failed'
}

export async function createPlannerJob(parameters: PlannerParameters): Promise<PlannerJob> {
  const response = await fetch(`${API_BASE_URL}/planner/jobs`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(parameters),
  })

  if (!response.ok) {
    throw new Error(`Unable to create planner job: ${response.status}`)
  }
  return response.json() as Promise<PlannerJob>
}

export async function getPlannerResult(jobId: string): Promise<RunResult> {
  const response = await fetch(`${API_BASE_URL}/planner/jobs/${jobId}/result`)
  if (!response.ok) {
    throw new Error(`Unable to load planner result: ${response.status}`)
  }
  return response.json() as Promise<RunResult>
}
