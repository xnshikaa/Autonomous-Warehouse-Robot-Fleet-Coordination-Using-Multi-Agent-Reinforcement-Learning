import * as THREE from 'three';
import { CollisionEvent, SafetyZoneState } from '../types/warehouse';
import { gridToWorld } from './gridConfig';

export class SafetySystem {
  public group: THREE.Group;
  private collisionRingMap: Map<string, THREE.Mesh> = new Map();
  private overrideWarningMesh!: THREE.Mesh;
  private overrideMat!: THREE.MeshBasicMaterial;
  private pulseTime: number = 0;

  constructor(scene: THREE.Scene) {
    this.group = new THREE.Group();
    this.group.name = 'SafetySystem';

    this.createOverridePerimeterFlash();

    scene.add(this.group);
  }

  /**
   * 1. Safety Override Visual Perimeter Warning (Subtle amber pulse)
   */
  private createOverridePerimeterFlash(): void {
    const geo = new THREE.RingGeometry(19.8, 20.2, 4);
    this.overrideMat = new THREE.MeshBasicMaterial({
      color: 0xf59e0b,
      side: THREE.DoubleSide,
      transparent: true,
      opacity: 0.0
    });

    this.overrideWarningMesh = new THREE.Mesh(geo, this.overrideMat);
    this.overrideWarningMesh.rotation.x = -Math.PI / 2;
    this.overrideWarningMesh.rotation.z = Math.PI / 4;
    this.overrideWarningMesh.position.y = 0.05;
    this.group.add(this.overrideWarningMesh);
  }

  public setSafetyOverride(active: boolean): void {
    if (active) {
      this.overrideMat.opacity = 0.65;
    } else {
      this.overrideMat.opacity = 0.0;
    }
  }

  /**
   * 2. Collision Warning Pulse Rings
   */
  public updateCollisions(collisions: CollisionEvent[]): void {
    const activeIds = new Set<string>();

    collisions.forEach(c => {
      activeIds.add(c.id);
      let ring = this.collisionRingMap.get(c.id);

      if (!ring) {
        const ringGeo = new THREE.RingGeometry(0.6, 1.2, 24);
        const ringMat = new THREE.MeshBasicMaterial({
          color: 0xef4444, // Hazard Red
          side: THREE.DoubleSide,
          transparent: true,
          opacity: 0.9
        });
        ring = new THREE.Mesh(ringGeo, ringMat);
        ring.rotation.x = -Math.PI / 2;

        const wPos = gridToWorld(c.position[0], c.position[1]);
        ring.position.set(wPos.x, 0.06, wPos.z);

        this.collisionRingMap.set(c.id, ring);
        this.group.add(ring);
      }
    });

    // Remove expired collision rings
    this.collisionRingMap.forEach((ring, id) => {
      if (!activeIds.has(id)) {
        ring.geometry.dispose();
        (ring.material as THREE.Material).dispose();
        this.group.remove(ring);
        this.collisionRingMap.delete(id);
      }
    });
  }

  public update(deltaTimeSec: number, safetyOverrideActive: boolean): void {
    this.pulseTime += deltaTimeSec * 4.0;

    // Pulse safety override boundary if active
    if (safetyOverrideActive) {
      this.overrideMat.opacity = 0.4 + Math.sin(this.pulseTime) * 0.25;
    }

    // Pulse collision rings
    const scale = 1.0 + Math.sin(this.pulseTime * 1.5) * 0.2;
    this.collisionRingMap.forEach(ring => {
      ring.scale.set(scale, scale, 1);
    });
  }
}
