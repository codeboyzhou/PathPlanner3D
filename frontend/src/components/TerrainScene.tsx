import { Canvas } from '@react-three/fiber'
import { Environment, Grid, Line, OrbitControls, PerspectiveCamera, Sphere, Text } from '@react-three/drei'
import { useMemo } from 'react'
import * as THREE from 'three'
import type { PlannerParameters, Point3D, RunResult } from '../types'
import { terrainHeight } from '../data/simulation'

interface Props {
  result: RunResult
  parameters: PlannerParameters
  progress: number
  isRunning: boolean
  compareMode: boolean
}

function toScene(point: Point3D): [number, number, number] {
  return [point.x - 50, point.z, point.y - 50]
}

function Terrain() {
  const geometry = useMemo(() => {
    const size = 100
    const segments = 70
    const plane = new THREE.PlaneGeometry(size, size, segments, segments)
    const positions = plane.attributes.position
    const colors: number[] = []
    const low = new THREE.Color('#b9cfbf')
    const high = new THREE.Color('#687f6e')

    for (let index = 0; index < positions.count; index += 1) {
      const x = positions.getX(index) + 50
      const y = positions.getY(index) + 50
      const height = terrainHeight(x, y)
      positions.setZ(index, height)
      const color = low.clone().lerp(high, Math.min(1, height / 25))
      colors.push(color.r, color.g, color.b)
    }
    plane.setAttribute('color', new THREE.Float32BufferAttribute(colors, 3))
    plane.computeVertexNormals()
    plane.rotateX(-Math.PI / 2)
    return plane
  }, [])

  return (
    <mesh geometry={geometry} receiveShadow>
      <meshStandardMaterial vertexColors roughness={0.92} metalness={0} side={THREE.DoubleSide} />
    </mesh>
  )
}

function Marker({ point, color, label }: { point: Point3D; color: string; label: string }) {
  const position = toScene(point)
  return (
    <group position={position}>
      <Sphere args={[1.15, 22, 22]} castShadow>
        <meshStandardMaterial color={color} roughness={0.3} />
      </Sphere>
      <mesh position={[0, -1.9, 0]}>
        <cylinderGeometry args={[0.05, 0.42, 2.6, 16]} />
        <meshStandardMaterial color={color} />
      </mesh>
      <Text position={[0, 2.5, 0]} fontSize={1.45} color="#243238" anchorX="center" outlineColor="#ffffff" outlineWidth={0.06}>
        {label}
      </Text>
    </group>
  )
}

function Path({ points, color, visibleRatio = 1 }: { points: Point3D[]; color: string; visibleRatio?: number }) {
  const scenePoints = useMemo(() => {
    const controlPoints = points.map((point) => new THREE.Vector3(...toScene(point)))
    const curve = new THREE.CatmullRomCurve3(controlPoints, false, 'catmullrom', 0.35)
    const sampled = curve.getPoints(Math.max(48, points.length * 12)).map((point) => point.toArray() as [number, number, number])
    const count = Math.max(2, Math.ceil(sampled.length * visibleRatio))
    return sampled.slice(0, count)
  }, [points, visibleRatio])

  return (
    <>
      <Line points={scenePoints} color={color} lineWidth={3.2} />
      {points.slice(1, -1).map((point, index) => (
        <Sphere args={[0.55, 14, 14]} position={toScene(point)} key={index}>
          <meshStandardMaterial color={color} emissive={color} emissiveIntensity={0.15} />
        </Sphere>
      ))}
    </>
  )
}

export function TerrainScene({ result, parameters, progress, isRunning, compareMode }: Props) {
  const paths = compareMode ? result.comparisons : [result.selected]
  const visibleRatio = isRunning ? Math.max(0.18, progress / 100) : 1

  return (
    <>
      <div className="scene-overlay">
        <div className="scene-badge">地形安全裕度 <strong>5.5 m</strong></div>
        <div className="scene-badge">路径状态 <strong>{isRunning ? `搜索中 ${progress}%` : '可行'}</strong></div>
      </div>
      <Canvas shadows dpr={[1, 1.75]} gl={{ antialias: true }}>
        <color attach="background" args={['#e8efed']} />
        <fog attach="fog" args={['#e8efed', 95, 180]} />
        <PerspectiveCamera makeDefault position={[116, 82, 60]} fov={43} />
        <ambientLight intensity={1.25} />
        <directionalLight position={[30, 70, 25]} intensity={2.2} castShadow shadow-mapSize={[1024, 1024]} />
        <Terrain />
        <Grid
          args={[100, 20]}
          position={[0, -0.15, 0]}
          cellColor="#9eaaa6"
          sectionColor="#75827d"
          cellThickness={0.45}
          sectionThickness={0.7}
          fadeDistance={145}
          infiniteGrid={false}
        />
        {paths.map((item) => <Path key={item.algorithm} points={item.path} color={item.color} visibleRatio={visibleRatio} />)}
        <Marker point={parameters.start} color="#17946c" label="START" />
        <Marker point={parameters.end} color="#d85c50" label="GOAL" />
        <OrbitControls makeDefault minDistance={45} maxDistance={190} maxPolarAngle={Math.PI / 2.02} target={[0, 8, 0]} />
        <Environment preset="city" />
      </Canvas>
    </>
  )
}
