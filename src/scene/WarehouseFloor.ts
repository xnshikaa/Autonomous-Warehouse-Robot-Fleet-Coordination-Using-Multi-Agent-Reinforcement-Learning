import * as THREE from 'three';
import { GRID_SIZE, CELL_SIZE, WAREHOUSE_WIDTH, WAREHOUSE_LENGTH, gridToWorld } from './gridConfig';

export class WarehouseFloor {
  public group: THREE.Group;
  private floorMesh!: THREE.Mesh;

  constructor(scene: THREE.Scene) {
    this.group = new THREE.Group();
    this.group.name = 'WarehouseFloor';

    this.createConcreteFloor();
    this.createGridLines();
    this.createAisleMarkings();
    this.createPerimeterBarriers();

    scene.add(this.group);
  }

  private createConcreteFloor(): void {
    const geometry = new THREE.PlaneGeometry(WAREHOUSE_WIDTH + 8, WAREHOUSE_LENGTH + 8);

    // Generate procedural polished concrete texture with subtle grid lines & tile texture
    const canvas = document.createElement('canvas');
    canvas.width = 1024;
    canvas.height = 1024;
    const ctx = canvas.getContext('2d')!;

    // Concrete background
    ctx.fillStyle = '#0F172A';
    ctx.fillRect(0, 0, 1024, 1024);

    // Subtle concrete noise/grain
    ctx.fillStyle = 'rgba(255, 255, 255, 0.03)';
    for (let i = 0; i < 5000; i++) {
      const x = Math.random() * 1024;
      const y = Math.random() * 1024;
      ctx.fillRect(x, y, 2, 2);
    }

    // Grid lines mapping 20x20
    ctx.strokeStyle = 'rgba(51, 65, 85, 0.4)';
    ctx.lineWidth = 2;
    const step = 1024 / GRID_SIZE;
    for (let i = 0; i <= GRID_SIZE; i++) {
      ctx.beginPath();
      ctx.moveTo(i * step, 0);
      ctx.lineTo(i * step, 1024);
      ctx.stroke();

      ctx.beginPath();
      ctx.moveTo(0, i * step);
      ctx.lineTo(1024, i * step);
      ctx.stroke();
    }

    const texture = new THREE.CanvasTexture(canvas);
    texture.wrapS = THREE.ClampToEdgeWrapping;
    texture.wrapT = THREE.ClampToEdgeWrapping;

    const material = new THREE.MeshStandardMaterial({
      map: texture,
      roughness: 0.4,
      metalness: 0.2,
      color: 0x1e293b
    });

    this.floorMesh = new THREE.Mesh(geometry, material);
    this.floorMesh.rotation.x = -Math.PI / 2;
    this.floorMesh.receiveShadow = true;
    this.group.add(this.floorMesh);
  }

  private createGridLines(): void {
    // 3D Grid helper elevated slightly above floor to prevent z-fighting
    const gridHelper = new THREE.GridHelper(WAREHOUSE_WIDTH, GRID_SIZE, 0x38bdf8, 0x334155);
    gridHelper.position.y = 0.01;
    (gridHelper.material as THREE.Material).opacity = 0.35;
    (gridHelper.material as THREE.Material).transparent = true;
    this.group.add(gridHelper);
  }

  private createAisleMarkings(): void {
    // Painted yellow hazard border line around 20x20 workspace boundary
    const borderGeo = new THREE.RingGeometry(WAREHOUSE_WIDTH / 2, WAREHOUSE_WIDTH / 2 + 0.3, 4);
    const borderMat = new THREE.MeshBasicMaterial({
      color: 0xf59e0b,
      side: THREE.DoubleSide,
      transparent: true,
      opacity: 0.8
    });
    const borderMesh = new THREE.Mesh(borderGeo, borderMat);
    borderMesh.rotation.x = -Math.PI / 2;
    borderMesh.rotation.z = Math.PI / 4;
    borderMesh.position.y = 0.02;
    this.group.add(borderMesh);

    // Painted Aisle text labels (Aisle 01 to Aisle 08)
    const aisleColumns = [3, 7, 11, 15];
    aisleColumns.forEach((colIndex, idx) => {
      const posTop = gridToWorld(colIndex, 0);
      const posBottom = gridToWorld(colIndex, GRID_SIZE - 1);

      this.createFloorTextMarker(`AISLE 0${idx + 1}`, new THREE.Vector3(posTop.x, 0.03, posTop.z - 1.2));
      this.createFloorTextMarker(`AISLE 0${idx + 1}`, new THREE.Vector3(posBottom.x, 0.03, posBottom.z + 1.2));
    });
  }

  private createFloorTextMarker(text: string, pos: THREE.Vector3): void {
    const canvas = document.createElement('canvas');
    canvas.width = 256;
    canvas.height = 64;
    const ctx = canvas.getContext('2d')!;

    ctx.fillStyle = 'rgba(15, 23, 42, 0.9)';
    ctx.fillRect(0, 0, 256, 64);
    ctx.strokeStyle = '#38BDF8';
    ctx.lineWidth = 4;
    ctx.strokeRect(2, 2, 252, 60);

    ctx.font = 'bold 24px "JetBrains Mono", monospace';
    ctx.fillStyle = '#F8FAFC';
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    ctx.fillText(text, 128, 32);

    const texture = new THREE.CanvasTexture(canvas);
    const material = new THREE.MeshBasicMaterial({
      map: texture,
      transparent: true,
      opacity: 0.9
    });

    const mesh = new THREE.Mesh(new THREE.PlaneGeometry(3, 0.75), material);
    mesh.rotation.x = -Math.PI / 2;
    mesh.position.copy(pos);
    this.group.add(mesh);
  }

  private createPerimeterBarriers(): void {
    // Low-profile industrial yellow safety railings around outer bounds
    const barrierMat = new THREE.MeshStandardMaterial({
      color: 0xf59e0b,
      metalness: 0.7,
      roughness: 0.3
    });

    const halfW = WAREHOUSE_WIDTH / 2 + 0.8;
    const height = 0.5;

    // Corner posts & horizontal rails
    const railPositions = [
      { start: [-halfW, -halfW], end: [halfW, -halfW] },
      { start: [halfW, -halfW], end: [halfW, halfW] },
      { start: [halfW, halfW], end: [-halfW, halfW] },
      { start: [-halfW, halfW], end: [-halfW, -halfW] }
    ];

    railPositions.forEach(rail => {
      const p1 = new THREE.Vector3(rail.start[0], height / 2, rail.start[1]);
      const p2 = new THREE.Vector3(rail.end[0], height / 2, rail.end[1]);
      const distance = p1.distanceTo(p2);

      const railMesh = new THREE.Mesh(
        new THREE.CylinderGeometry(0.06, 0.06, distance, 8),
        barrierMat
      );

      railMesh.position.copy(p1.clone().add(p2).multiplyScalar(0.5));
      railMesh.lookAt(p2);
      railMesh.rotateX(Math.PI / 2);
      this.group.add(railMesh);
    });
  }
}
