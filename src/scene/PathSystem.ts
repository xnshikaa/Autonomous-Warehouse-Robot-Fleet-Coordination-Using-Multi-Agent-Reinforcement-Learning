import * as THREE from 'three';
import { RobotState } from '../types/warehouse';
import { gridToWorld } from './gridConfig';

export type PathVisibilityMode = 'ALL' | 'SELECTED_ONLY' | 'OFF';

export class PathSystem {
  public group: THREE.Group;
  private pathLineMap: Map<string, THREE.Line> = new Map();
  private pathMatMap: Map<string, THREE.LineDashedMaterial> = new Map();
  
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
    this.selectedRobotId = robotId;
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
      const points: THREE.Vector3[] = robot.path.map(pt => {
        const w = gridToWorld(pt[0], pt[1]);
        return new THREE.Vector3(w.x, 0.08, w.z); // Slightly above floor to prevent z-fighting
      });

      let line = this.pathLineMap.get(robot.id);
      if (!line) {
        const colorHex = this.pathColors[idx % this.pathColors.length];
        const geometry = new THREE.BufferGeometry().setFromPoints(points);

        const material = new THREE.LineDashedMaterial({
          color: colorHex,
          linewidth: 3,
          scale: 1,
          dashSize: 0.6,
          gapSize: 0.3,
          transparent: true,
          opacity: 0.85
        });

        line = new THREE.Line(geometry, material);
        line.computeLineDistances();

        this.pathLineMap.set(robot.id, line);
        this.pathMatMap.set(robot.id, material);
        this.group.add(line);
      } else {
        // Update line geometry points
        line.geometry.dispose();
        line.geometry = new THREE.BufferGeometry().setFromPoints(points);
        line.computeLineDistances();
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
    this.pathMatMap.forEach(mat => {
      if (mat.dashSize !== undefined) {
        mat.dashSize = (mat.dashSize + deltaTimeSec * 0.5) % 2;
      }
    });
  }

  private removePath(id: string): void {
    const line = this.pathLineMap.get(id);
    if (line) {
      line.geometry.dispose();
      (line.material as THREE.Material).dispose();
      this.group.remove(line);
      this.pathLineMap.delete(id);
      this.pathMatMap.delete(id);
    }
  }
}
