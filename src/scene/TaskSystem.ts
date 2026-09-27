import * as THREE from 'three';
import { TaskState } from '../types/warehouse';
import { gridToWorld } from './gridConfig';

interface TaskBeaconGroup {
  group: THREE.Group;
  pickupRing: THREE.Mesh;
  deliveryRing: THREE.Mesh;
  pickupMat: THREE.MeshBasicMaterial;
  deliveryMat: THREE.MeshBasicMaterial;
}

export class TaskSystem {
  public group: THREE.Group;
  private taskBeaconMap: Map<string, TaskBeaconGroup> = new Map();
  private pulseTime: number = 0;

  constructor(scene: THREE.Scene) {
    this.group = new THREE.Group();
    this.group.name = 'TaskSystem';

    scene.add(this.group);
  }

  /**
   * Syncs task target beacons with active TaskState telemetry
   */
  public updateTasks(tasks: TaskState[]): void {
    const activeIds = new Set<string>();

    tasks.forEach(task => {
      if (task.status === 'COMPLETED') return;

      activeIds.add(task.id);
      let beaconGroup = this.taskBeaconMap.get(task.id);

      if (!beaconGroup) {
        beaconGroup = this.createBeaconGroup(task);
        this.taskBeaconMap.set(task.id, beaconGroup);
        this.group.add(beaconGroup.group);
      }

      // Update positions
      const pPos = gridToWorld(task.pickupPos[0], task.pickupPos[1]);
      const dPos = gridToWorld(task.deliveryPos[0], task.deliveryPos[1]);

      beaconGroup.pickupRing.position.set(pPos.x, 0.04, pPos.z);
      beaconGroup.deliveryRing.position.set(dPos.x, 0.04, dPos.z);
    });

    // Remove finished task beacons
    this.taskBeaconMap.forEach((bg, id) => {
      if (!activeIds.has(id)) {
        this.group.remove(bg.group);
        this.taskBeaconMap.delete(id);
      }
    });
  }

  public update(deltaTimeSec: number): void {
    this.pulseTime += deltaTimeSec * 3.0;

    // Pulse pulse ring scale & opacity
    const scale = 1.0 + Math.sin(this.pulseTime) * 0.15;
    const opacity = 0.6 + Math.cos(this.pulseTime) * 0.25;

    this.taskBeaconMap.forEach(bg => {
      bg.pickupRing.scale.set(scale, scale, 1);
      bg.deliveryRing.scale.set(scale, scale, 1);

      bg.pickupMat.opacity = opacity;
      bg.deliveryMat.opacity = opacity;
    });
  }

  private createBeaconGroup(task: TaskState): TaskBeaconGroup {
    const mainGroup = new THREE.Group();
    mainGroup.name = `TaskBeacon_${task.id}`;

    // 1. Pickup Target Ring (Amber)
    const ringGeo = new THREE.RingGeometry(0.5, 0.8, 24);
    const pickupMat = new THREE.MeshBasicMaterial({
      color: 0xf59e0b,
      side: THREE.DoubleSide,
      transparent: true,
      opacity: 0.85
    });

    const pickupRing = new THREE.Mesh(ringGeo, pickupMat);
    pickupRing.rotation.x = -Math.PI / 2;
    mainGroup.add(pickupRing);

    // 2. Delivery Target Ring (Emerald Green)
    const deliveryMat = new THREE.MeshBasicMaterial({
      color: 0x10b981,
      side: THREE.DoubleSide,
      transparent: true,
      opacity: 0.85
    });

    const deliveryRing = new THREE.Mesh(ringGeo, deliveryMat);
    deliveryRing.rotation.x = -Math.PI / 2;
    mainGroup.add(deliveryRing);

    // 3. Text Label Badges
    const pPos = gridToWorld(task.pickupPos[0], task.pickupPos[1]);
    const dPos = gridToWorld(task.deliveryPos[0], task.deliveryPos[1]);

    const pickupLabel = this.createBeaconLabel(`PICKUP ${task.id}`, 0xf59e0b);
    pickupLabel.position.set(pPos.x, 1.8, pPos.z);
    mainGroup.add(pickupLabel);

    const deliveryLabel = this.createBeaconLabel(`DELIVER ${task.id}`, 0x10b981);
    deliveryLabel.position.set(dPos.x, 1.8, dPos.z);
    mainGroup.add(deliveryLabel);

    return {
      group: mainGroup,
      pickupRing,
      deliveryRing,
      pickupMat,
      deliveryMat
    };
  }

  private createBeaconLabel(text: string, hexColor: number): THREE.Mesh {
    const canvas = document.createElement('canvas');
    canvas.width = 256;
    canvas.height = 64;
    const ctx = canvas.getContext('2d')!;

    const colorHexStr = `#${hexColor.toString(16).padStart(6, '0')}`;
    ctx.fillStyle = 'rgba(15, 23, 42, 0.95)';
    ctx.fillRect(0, 0, 256, 64);
    ctx.strokeStyle = colorHexStr;
    ctx.lineWidth = 4;
    ctx.strokeRect(2, 2, 252, 60);

    ctx.font = 'bold 22px "JetBrains Mono", monospace';
    ctx.fillStyle = colorHexStr;
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    ctx.fillText(text, 128, 32);

    const texture = new THREE.CanvasTexture(canvas);
    const mat = new THREE.MeshBasicMaterial({
      map: texture,
      transparent: true,
      side: THREE.DoubleSide
    });

    return new THREE.Mesh(new THREE.PlaneGeometry(2.0, 0.5), mat);
  }
}
