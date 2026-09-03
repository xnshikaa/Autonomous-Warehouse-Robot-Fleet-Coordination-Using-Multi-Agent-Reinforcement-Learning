import * as THREE from 'three';
import { GRID_SIZE, CELL_SIZE, gridToWorld } from './gridConfig';

export class ZoneSystem {
  public group: THREE.Group;

  constructor(scene: THREE.Scene) {
    this.group = new THREE.Group();
    this.group.name = 'ZoneSystem';

    this.createDispatchZone();
    this.createChargingStation();
    this.createHumanSafetyZone();

    scene.add(this.group);
  }

  /**
   * 1. Loading & Dispatch Zone (Delivery locations D01 - D04)
   */
  private createDispatchZone(): void {
    const dispatchGroup = new THREE.Group();
    dispatchGroup.name = 'DispatchZone';

    const dispatchCols = [3, 7, 11, 15];
    const dispatchRow = 18;

    dispatchCols.forEach((col, idx) => {
      const worldPos = gridToWorld(col, dispatchRow);

      // Floor target pad (Green glowing border)
      const padGeo = new THREE.PlaneGeometry(CELL_SIZE * 0.9, CELL_SIZE * 0.9);
      const padMat = new THREE.MeshStandardMaterial({
        color: 0x10b981,
        roughness: 0.3,
        metalness: 0.4,
        emissive: 0x10b981,
        emissiveIntensity: 0.25,
        transparent: true,
        opacity: 0.85
      });

      const padMesh = new THREE.Mesh(padGeo, padMat);
      padMesh.rotation.x = -Math.PI / 2;
      padMesh.position.set(worldPos.x, 0.02, worldPos.z);
      dispatchGroup.add(padMesh);

      // Delivery ID Signage D01, D02...
      this.createZoneSign(`D0${idx + 1}`, new THREE.Vector3(worldPos.x, 0.03, worldPos.z), 0x10b981);

      // Roller Conveyor Belt structure behind delivery pad
      const conveyorGeo = new THREE.BoxGeometry(CELL_SIZE * 0.85, 0.4, 0.8);
      const conveyorMat = new THREE.MeshStandardMaterial({
        color: 0x334155,
        metalness: 0.8,
        roughness: 0.2
      });
      const conveyorMesh = new THREE.Mesh(conveyorGeo, conveyorMat);
      conveyorMesh.position.set(worldPos.x, 0.2, worldPos.z + 1.2);
      conveyorMesh.castShadow = true;
      dispatchGroup.add(conveyorMesh);
    });

    // Overview Header Sign for Dispatch
    const mainPos = gridToWorld(9, 19);
    this.createHeaderSign('DISPATCH & DELIVERY BAY', new THREE.Vector3(mainPos.x, 3.5, mainPos.z + 1.5), 0x10b981);

    this.group.add(dispatchGroup);
  }

  /**
   * 2. Robot Charging Station (Charging slots C01 - C05)
   */
  private createChargingStation(): void {
    const chargingGroup = new THREE.Group();
    chargingGroup.name = 'ChargingStation';

    const chargingCols = [2, 6, 10, 14, 18];
    const chargingRow = 0;

    chargingCols.forEach((col, idx) => {
      const worldPos = gridToWorld(col, chargingRow);

      // Inductive Charging Floor Pad (Cyan/Blue status glow)
      const padGeo = new THREE.RingGeometry(0.4, 0.75, 16);
      const padMat = new THREE.MeshBasicMaterial({
        color: 0x38bdf8,
        side: THREE.DoubleSide,
        transparent: true,
        opacity: 0.9
      });

      const padMesh = new THREE.Mesh(padGeo, padMat);
      padMesh.rotation.x = -Math.PI / 2;
      padMesh.position.set(worldPos.x, 0.02, worldPos.z);
      chargingGroup.add(padMesh);

      // Vertical Charging Docking Tower
      const dockGeo = new THREE.BoxGeometry(0.4, 1.2, 0.4);
      const dockMat = new THREE.MeshStandardMaterial({
        color: 0x1e293b,
        metalness: 0.9,
        roughness: 0.2,
        emissive: 0x38bdf8,
        emissiveIntensity: 0.3
      });

      const dockMesh = new THREE.Mesh(dockGeo, dockMat);
      dockMesh.position.set(worldPos.x, 0.6, worldPos.z - 0.8);
      dockMesh.castShadow = true;
      chargingGroup.add(dockMesh);

      this.createZoneSign(`C0${idx + 1}`, new THREE.Vector3(worldPos.x, 0.03, worldPos.z), 0x38bdf8);
    });

    // Overview Header Sign for Charging Station
    const mainPos = gridToWorld(9, 0);
    this.createHeaderSign('ROBOT CHARGING STATION', new THREE.Vector3(mainPos.x, 3.5, mainPos.z - 1.5), 0x38bdf8);

    this.group.add(chargingGroup);
  }

  /**
   * 3. Human-Worker Safety Zone (Hazard stripes, translucent cage, warning signs)
   */
  private createHumanSafetyZone(): void {
    const safetyGroup = new THREE.Group();
    safetyGroup.name = 'HumanSafetyZone';

    // Occupies grid range [13-15, 8-10]
    const minPos = gridToWorld(8, 8);
    const maxPos = gridToWorld(10, 10);

    const centerX = (minPos.x + maxPos.x) / 2;
    const centerZ = (minPos.z + maxPos.z) / 2;
    const sizeX = (3 * CELL_SIZE);
    const sizeZ = (3 * CELL_SIZE);

    // Hazard Striped Floor Plane
    const canvas = document.createElement('canvas');
    canvas.width = 512;
    canvas.height = 512;
    const ctx = canvas.getContext('2d')!;

    // Yellow / Dark Grey hazard diagonal stripes
    ctx.fillStyle = '#F59E0B';
    ctx.fillRect(0, 0, 512, 512);

    ctx.fillStyle = '#1E293B';
    ctx.lineWidth = 40;
    for (let i = -512; i < 1024; i += 80) {
      ctx.beginPath();
      ctx.moveTo(i, 0);
      ctx.lineTo(i + 512, 512);
      ctx.lineTo(i + 550, 512);
      ctx.lineTo(i + 38, 0);
      ctx.fill();
    }

    const hazardTexture = new THREE.CanvasTexture(canvas);
    hazardTexture.wrapS = THREE.RepeatWrapping;
    hazardTexture.wrapT = THREE.RepeatWrapping;
    hazardTexture.repeat.set(2, 2);

    const floorMat = new THREE.MeshStandardMaterial({
      map: hazardTexture,
      roughness: 0.5,
      metalness: 0.2,
      transparent: true,
      opacity: 0.75
    });

    const floorMesh = new THREE.Mesh(new THREE.PlaneGeometry(sizeX, sizeZ), floorMat);
    floorMesh.rotation.x = -Math.PI / 2;
    floorMesh.position.set(centerX, 0.025, centerZ);
    safetyGroup.add(floorMesh);

    // Translucent Yellow Boundary Cage Wall
    const cageGeo = new THREE.BoxGeometry(sizeX, 2.2, sizeZ);
    const cageMat = new THREE.MeshStandardMaterial({
      color: 0xf59e0b,
      transparent: true,
      opacity: 0.12,
      roughness: 0.1,
      metalness: 0.8,
      side: THREE.DoubleSide
    });

    const cageMesh = new THREE.Mesh(cageGeo, cageMat);
    cageMesh.position.set(centerX, 1.1, centerZ);
    safetyGroup.add(cageMesh);

    // Cage Frame Posts
    const frameGeo = new THREE.BoxGeometry(sizeX, 0.06, sizeZ);
    const frameMat = new THREE.MeshStandardMaterial({ color: 0xf59e0b, metalness: 0.8, roughness: 0.2 });
    const topFrame = new THREE.Mesh(frameGeo, frameMat);
    topFrame.position.set(centerX, 2.2, centerZ);
    safetyGroup.add(topFrame);

    // Warning Header Sign
    this.createHeaderSign('HUMAN SAFETY ZONE • AUTHORIZED PERSONNEL ONLY', new THREE.Vector3(centerX, 2.8, centerZ), 0xf59e0b);

    this.group.add(safetyGroup);
  }

  private createZoneSign(text: string, pos: THREE.Vector3, hexColor: number): void {
    const canvas = document.createElement('canvas');
    canvas.width = 128;
    canvas.height = 128;
    const ctx = canvas.getContext('2d')!;

    ctx.fillStyle = 'rgba(15, 23, 42, 0.95)';
    ctx.beginPath();
    ctx.arc(64, 64, 58, 0, Math.PI * 2);
    ctx.fill();

    const colorHexStr = `#${hexColor.toString(16).padStart(6, '0')}`;
    ctx.strokeStyle = colorHexStr;
    ctx.lineWidth = 6;
    ctx.stroke();

    ctx.font = 'bold 36px "JetBrains Mono", monospace';
    ctx.fillStyle = '#FFFFFF';
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    ctx.fillText(text, 64, 64);

    const texture = new THREE.CanvasTexture(canvas);
    const mat = new THREE.MeshBasicMaterial({ map: texture, transparent: true });
    const mesh = new THREE.Mesh(new THREE.PlaneGeometry(1.2, 1.2), mat);
    mesh.rotation.x = -Math.PI / 2;
    mesh.position.copy(pos);
    this.group.add(mesh);
  }

  private createHeaderSign(text: string, pos: THREE.Vector3, hexColor: number): void {
    const canvas = document.createElement('canvas');
    canvas.width = 512;
    canvas.height = 64;
    const ctx = canvas.getContext('2d')!;

    const colorHexStr = `#${hexColor.toString(16).padStart(6, '0')}`;
    ctx.fillStyle = 'rgba(15, 23, 42, 0.95)';
    ctx.fillRect(0, 0, 512, 64);
    ctx.strokeStyle = colorHexStr;
    ctx.lineWidth = 4;
    ctx.strokeRect(2, 2, 508, 60);

    ctx.font = 'bold 18px "JetBrains Mono", monospace';
    ctx.fillStyle = colorHexStr;
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    ctx.fillText(text, 256, 32);

    const texture = new THREE.CanvasTexture(canvas);
    const mat = new THREE.MeshStandardMaterial({
      map: texture,
      emissive: hexColor,
      emissiveIntensity: 0.3,
      transparent: true,
      side: THREE.DoubleSide
    });

    const mesh = new THREE.Mesh(new THREE.PlaneGeometry(7.0, 0.875), mat);
    mesh.position.copy(pos);
    this.group.add(mesh);
  }
}
