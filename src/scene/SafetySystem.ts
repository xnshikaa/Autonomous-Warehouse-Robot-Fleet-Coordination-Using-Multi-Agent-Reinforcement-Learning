import * as THREE from 'three';
import { CollisionEvent, SafetyZoneState } from '../types/warehouse';
import { gridToWorld } from './gridConfig';

export class SafetySystem {
  public group: THREE.Group;
  private collisionRingMap: Map<string, THREE.Mesh> = new Map();
  private collisionLabelMap: Map<string, THREE.Sprite> = new Map();
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

        const label = this.createConflictLabel(c.robots);
        label.position.set(wPos.x, 0.55, wPos.z);
        this.collisionLabelMap.set(c.id, label);
        this.group.add(label);
      }
    });

    // Remove expired collision rings
    this.collisionRingMap.forEach((ring, id) => {
      if (!activeIds.has(id)) {
        ring.geometry.dispose();
        (ring.material as THREE.Material).dispose();
        this.group.remove(ring);
        this.collisionRingMap.delete(id);

        const label = this.collisionLabelMap.get(id);
        if (label) {
          (label.material as THREE.SpriteMaterial).map?.dispose();
          (label.material as THREE.Material).dispose();
          this.group.remove(label);
          this.collisionLabelMap.delete(id);
        }
      }
    });
  }

  public update(deltaTimeSec: number, safetyOverrideActive: boolean): void {
    this.pulseTime += deltaTimeSec * 4.0;

    // Pulse safety override boundary if active
    if (safetyOverrideActive) {
      this.overrideMat.opacity = 0.4 + Math.sin(this.pulseTime) * 0.25;
    } else {
      this.overrideMat.opacity = 0.0;
    }

    // Pulse collision rings
    const scale = 1.0 + Math.sin(this.pulseTime * 1.5) * 0.2;
    this.collisionRingMap.forEach(ring => {
      ring.scale.set(scale, scale, 1);
    });
    this.collisionLabelMap.forEach(label => {
      label.material.opacity = 0.72 + Math.sin(this.pulseTime) * 0.2;
    });
  }

  private createConflictLabel(robotIds: string[]): THREE.Sprite {
    const canvas = document.createElement('canvas');
    canvas.width = 256;
    canvas.height = 64;
    const context = canvas.getContext('2d')!;
    context.fillStyle = '#7F1D1D';
    context.fillRect(0, 0, canvas.width, canvas.height);
    context.strokeStyle = '#FCA5A5';
    context.lineWidth = 4;
    context.strokeRect(2, 2, canvas.width - 4, canvas.height - 4);
    context.fillStyle = '#FFFFFF';
    context.font = 'bold 27px "JetBrains Mono", monospace';
    context.textAlign = 'center';
    context.textBaseline = 'middle';
    const participants = robotIds
      .map(id => String(Number(id.match(/\d+/)?.[0] ?? id)))
      .join('/');
    context.fillText(`! ${participants || 'CONFLICT'}`, canvas.width / 2, canvas.height / 2);

    const texture = new THREE.CanvasTexture(canvas);
    const material = new THREE.SpriteMaterial({
      map: texture,
      transparent: true,
      depthWrite: false,
      opacity: 0.9
    });
    const sprite = new THREE.Sprite(material);
    sprite.scale.set(2.8, 0.7, 1);
    return sprite;
  }
}
