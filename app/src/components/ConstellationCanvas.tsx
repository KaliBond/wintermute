import { useRef, useEffect, useMemo } from 'react';
import { Canvas, useFrame, useThree } from '@react-three/fiber';
import * as THREE from 'three';

const NODE_COUNT = 80;
const CONNECTION_DIST = 5;
const SCENE_RADIUS = 12;

function NodeSphere({
  position,
  phase,
}: {
  position: THREE.Vector3;
  phase: number;
}) {
  const meshRef = useRef<THREE.Mesh>(null);

  useFrame(({ clock }) => {
    if (meshRef.current) {
      meshRef.current.position.y = position.y + Math.sin(clock.elapsedTime * 0.5 + phase) * 0.15;
    }
  });

  return (
    <mesh ref={meshRef} position={position}>
      <sphereGeometry args={[0.15, 16, 16]} />
      <meshStandardMaterial
        color="#5A7A6A"
        emissive="#5A7A6A"
        emissiveIntensity={0.4}
        roughness={0.7}
        metalness={0.1}
      />
    </mesh>
  );
}

function ConnectionLines({ nodes }: { nodes: { position: THREE.Vector3 }[] }) {
  const linesRef = useRef<THREE.LineSegments>(null);

  const { positions, linePairs } = useMemo(() => {
    const pairs: [number, number][] = [];
    for (let i = 0; i < nodes.length; i++) {
      for (let j = i + 1; j < nodes.length; j++) {
        const dist = nodes[i].position.distanceTo(nodes[j].position);
        if (dist < CONNECTION_DIST) {
          pairs.push([i, j]);
        }
      }
    }
    const posArray = new Float32Array(pairs.length * 6);
    return { positions: posArray, linePairs: pairs };
  }, [nodes]);

  const geometry = useMemo(() => {
    const geo = new THREE.BufferGeometry();
    geo.setAttribute('position', new THREE.BufferAttribute(positions, 3));
    return geo;
  }, [positions]);

  useFrame(({ clock }) => {
    if (!linesRef.current) return;
    const posAttr = linesRef.current.geometry.attributes.position;
    const posArray = posAttr.array as Float32Array;
    const t = clock.elapsedTime;

    for (let i = 0; i < linePairs.length; i++) {
      const [a, b] = linePairs[i];
      const offsetA = Math.sin(t * 0.5 + nodes[a].position.x) * 0.15;
      const offsetB = Math.sin(t * 0.5 + nodes[b].position.x) * 0.15;
      posArray[i * 6] = nodes[a].position.x;
      posArray[i * 6 + 1] = nodes[a].position.y + offsetA;
      posArray[i * 6 + 2] = nodes[a].position.z;
      posArray[i * 6 + 3] = nodes[b].position.x;
      posArray[i * 6 + 4] = nodes[b].position.y + offsetB;
      posArray[i * 6 + 5] = nodes[b].position.z;
    }
    posAttr.needsUpdate = true;
  });

  return (
    <lineSegments ref={linesRef} geometry={geometry}>
      <lineBasicMaterial color="#3A4148" transparent opacity={0.25} />
    </lineSegments>
  );
}

function SceneController() {
  const { gl } = useThree();
  const groupRef = useRef<THREE.Group>(null);
  const isDragging = useRef(false);
  const previousMouse = useRef({ x: 0, y: 0 });

  const nodes = useMemo(() => {
    const result: { position: THREE.Vector3; phase: number }[] = [];
    for (let i = 0; i < NODE_COUNT; i++) {
      const r = Math.cbrt(Math.random()) * SCENE_RADIUS;
      const theta = Math.random() * Math.PI * 2;
      const phi = Math.acos(2 * Math.random() - 1);
      result.push({
        position: new THREE.Vector3(
          r * Math.sin(phi) * Math.cos(theta),
          r * Math.sin(phi) * Math.sin(theta),
          r * Math.cos(phi)
        ),
        phase: Math.random() * Math.PI * 2,
      });
    }
    return result;
  }, []);

  useFrame(() => {
    if (groupRef.current && !isDragging.current) {
      groupRef.current.rotation.y += 0.001;
    }
  });

  useEffect(() => {
    const canvas = gl.domElement;

    const onMouseDown = (e: MouseEvent) => {
      isDragging.current = true;
      previousMouse.current = { x: e.clientX, y: e.clientY };
    };
    const onMouseMove = (e: MouseEvent) => {
      if (!isDragging.current || !groupRef.current) return;
      const deltaX = e.clientX - previousMouse.current.x;
      const deltaY = e.clientY - previousMouse.current.y;
      groupRef.current.rotation.y += deltaX * 0.005;
      groupRef.current.rotation.x += deltaY * 0.005;
      previousMouse.current = { x: e.clientX, y: e.clientY };
    };
    const onMouseUp = () => {
      isDragging.current = false;
    };

    canvas.addEventListener('mousedown', onMouseDown);
    window.addEventListener('mousemove', onMouseMove);
    window.addEventListener('mouseup', onMouseUp);

    return () => {
      canvas.removeEventListener('mousedown', onMouseDown);
      window.removeEventListener('mousemove', onMouseMove);
      window.removeEventListener('mouseup', onMouseUp);
    };
  }, [gl]);

  return (
    <group ref={groupRef}>
      <ambientLight intensity={0.4} />
      <directionalLight position={[10, 10, 10]} intensity={0.6} />
      <pointLight position={[0, 0, 0]} intensity={0.3} color="#5A7A6A" />
      <fog attach="fog" args={['#1A1D21', 10, 50]} />

      {nodes.map((node, i) => (
        <NodeSphere key={i} position={node.position} phase={node.phase} />
      ))}
      <ConnectionLines nodes={nodes} />
    </group>
  );
}

export function ConstellationCanvas() {
  return (
    <div
      style={{
        position: 'absolute',
        inset: 0,
        zIndex: 0,
      }}
      role="img"
      aria-label="A 3D network graph of interconnected nodes representing societal coordination systems"
    >
      <Canvas
        camera={{ position: [0, 0, 25], fov: 60, near: 0.1, far: 1000 }}
        dpr={[1, 2]}
        gl={{ antialias: true }}
        onCreated={({ gl }) => {
          gl.setClearColor('#1A1D21');
        }}
      >
        <SceneController />
      </Canvas>
    </div>
  );
}
