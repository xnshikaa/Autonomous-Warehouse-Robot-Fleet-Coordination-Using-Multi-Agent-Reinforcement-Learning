import * as THREE from 'three';
import { RobotState } from '../types/warehouse';
import { gridToWorld } from './gridConfig';

export type PathVisibilityMode = 'ALL' | 'SELECTED_ONLY' | 'OFF';

export class PathSystem {
  public group: THREE.Group;
  private pathLineMap: Map<string, THREE.Mesh> = new Map();
  private pathMatMap: Map<string, THREE.MeshBasicMaterial> = new Map();
  private pathPointsMap: Map<string, THREE.Vector3[]> = new Map();
  private pathSignatureMap: Map<string, string> = new Map();
  private pathSelectedMap: Map<string, boolean> = new Map();
  
  private visibilityMode: PathVisibilityMode = 'ALL';
  private selectedRobotId: string | null = null;

  // Distinct vibrant neon colors for robot paths (ensures 20 robots have unique path colors)
  private readonly pathColors: number[] = [
    0x38bdf8, 0x10b981, 0xf59e0b, 0xa855f7, 0xec4899,
    0x06b6d4, 0x84cc16, 0xf97316, 0x6366f1, 0xd946ef,
    0x14b8a6, 0xeab308, 0xef4444, 0x3b82f6, 0x8b5cf6,
    0x2dd4bf, 0xfacc15, 0xf87171, 0x60a5fa, 0xc084fc
  ];

  constructor(scene: THREE.Scene) {
    this.group = new THREE.Group();
    this.group.name = 'PathSystem';

    scene.add(this.group);
  }

  public setVisibilityMode(mode: PathVisibilityMode): void {
    this.visibilityMode = mode;
    this.group.visible = mode !== 'OFF';
  }

  public setSelectedRobot(robotId: string | null): void {
    const previousSelectedRobotId = this.selectedRobotId;
    this.selectedRobotId = robotId;
    [previousSelectedRobotId, robotId].forEach(id => {
      if (!id || id === previousSelectedRobotId && id === robotId) return;
      const material = this.pathMatMap.get(id);
      const line = this.pathLineMap.get(id);
      const points = this.pathPointsMap.get(id);
      if (material) this.applyStyle(id, material);
      if (line && points) {
        line.geometry.dispose();
        line.geometry = this.createPathGeometry(points, id === this.selectedRobotId);
        this.pathSelectedMap.set(id, id === this.selectedRobotId);
      }
    });
  }

  /**
   * Syncs active 3D path line ribbons with robot path telemetry
   */
  public updatePaths(robots: RobotState[]): void {
    if (this.visibilityMode === 'OFF') {
      this.group.visible = false;
      return;
    }
    this.group.visible = true;

    const activeIds = new Set<string>();

    robots.forEach((robot, idx) => {
      activeIds.add(robot.id);

      // Check visibility filter
      const isVisible =
        this.visibilityMode === 'ALL' ||
        (this.visibilityMode === 'SELECTED_ONLY' && robot.id === this.selectedRobotId);

      if (!robot.path || robot.path.length < 2 || !isVisible) {
        this.removePath(robot.id);
        return;
      }

      // Convert grid points array [(x1,y1), (x2,y2)...] to 3D Vector3 points
      const pathSignature = robot.path.map(([x, y]) => `${x},${y}`).join(';');
      const pathChanged = this.pathSignatureMap.get(robot.id) !== pathSignature;
      if (pathChanged) {
        const points: THREE.Vector3[] = robot.path.map(pt => {
          const w = gridToWorld(pt[0], pt[1]);
          return new THREE.Vector3(w.x, 0.14, w.z); // Clearly above floor and robot rings
        });
        this.pathPointsMap.set(robot.id, points);
        this.pathSignatureMap.set(robot.id, pathSignature);
      }

      let line = this.pathLineMap.get(robot.id);
      if (!line) {
        const points = this.pathPointsMap.get(robot.id)!;
        const numericId = Number(robot.id.match(/\d+/)?.[0] ?? idx);
        const colorHex = this.pathColors[Math.max(0, numericId - 1) % this.pathColors.length];
        const selected = robot.id === this.selectedRobotId;
        const geometry = this.createPathGeometry(points, selected);

        const material = new THREE.MeshBasicMaterial({
          color: colorHex,
          transparent: true,
          opacity: selected ? 1.0 : 0.78,
          depthWrite: false
        });

        line = new THREE.Mesh(geometry, material);

        this.pathLineMap.set(robot.id, line);
        this.pathMatMap.set(robot.id, material);
        this.pathSelectedMap.set(robot.id, selected);
        this.group.add(line);
      } else {
        const selected = robot.id === this.selectedRobotId;
        const wasSelected = this.pathSelectedMap.get(robot.id) ?? false;
        if (pathChanged || wasSelected !== selected) {
          const points = this.pathPointsMap.get(robot.id)!;
          line.geometry.dispose();
          line.geometry = this.createPathGeometry(points, selected);
          this.pathSelectedMap.set(robot.id, selected);
        }
        this.applyStyle(robot.id, this.pathMatMap.get(robot.id)!);
      }
    });

    // Remove paths for idle or decommissioned robots
    this.pathLineMap.forEach((_, id) => {
      if (!activeIds.has(id)) {
        this.removePath(id);
      }
    });
  }

  public update(deltaTimeSec: number): void {
    // Animate directional dash offset flow along path lines
    // Keep this loop intentionally small and allocation-free for a 20-robot fleet.
    this.pathMatMap.forEach((mat, id) => this.applyStyle(id, mat));
  }

  private applyStyle(id: string, material: THREE.MeshBasicMaterial): void {
    const selected = id === this.selectedRobotId;
    material.opacity = selected ? 1.0 : 0.78;
  }

  private removePath(id: string): void {
    const line = this.pathLineMap.get(id);
    if (line) {
      line.geometry.dispose();
      (line.material as THREE.Material).dispose();
      this.group.remove(line);
      this.pathLineMap.delete(id);
      this.pathMatMap.delete(id);
      this.pathPointsMap.delete(id);
      this.pathSignatureMap.delete(id);
      this.pathSelectedMap.delete(id);
    }
  }

  private createPathGeometry(points: THREE.Vector3[], selected: boolean): THREE.TubeGeometry {
    return new THREE.TubeGeometry(
      new THREE.CatmullRomCurve3(points),
      Math.max(1, points.length - 1),
      selected ? 0.14 : 0.07,
      6,
      false
    );
  }
}
