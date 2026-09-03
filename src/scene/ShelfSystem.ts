import * as THREE from 'three';
import { GRID_SIZE, CELL_SIZE, gridToWorld } from './gridConfig';
import { ShelfState } from '../types/warehouse';

export class ShelfSystem {
  public group: THREE.Group;
  public shelves: ShelfState[] = [];
  
  private shelfRackInstancedMesh!: THREE.InstancedMesh;
  private crateInstancedMesh!: THREE.InstancedMesh;
  private totalRacks: number = 0;

  constructor(scene: THREE.Scene) {
    this.group = new THREE.Group();
    this.group.name = 'ShelfSystem';

    this.generateLayoutData();
    this.buildInstancedRacks();
    this.createShelfSignage();

    scene.add(this.group);
  }

  /**
   * Defines logical shelf layout across 20x20 grid.
   * Shelves are grouped in double-row blocks creating 4 primary aisles & 1 cross-aisle.
   */
  private generateLayoutData(): void {
    // Columns where shelf racks are placed (leaving columns 0, 4, 9, 14, 19 for aisles/corridors)
    const rackColumns = [1, 2, 5, 6, 11, 12, 16, 17];
    // Rows where shelf racks are placed (split into Top & Bottom blocks by Cross-Aisle at rows 9-10)
    const rackRows = [2, 3, 4, 5, 6, 7, 12, 13, 14, 15, 16, 17];

    let shelfCount = 1;
    rackColumns.forEach(col => {
      rackRows.forEach(row => {
        const shelfId = `S${shelfCount < 10 ? '0' : ''}${shelfCount}`;
        const aisleId = `AISLE 0${Math.floor(col / 4) + 1}`;

        this.shelves.push({
          id: shelfId,
          gridPos: [col, row],
          levels: 3,
          aisleId
        });
        shelfCount++;
      });
    });

    this.totalRacks = this.shelves.length;
  }

  private buildInstancedRacks(): void {
    const rackWidth = CELL_SIZE * 0.85;  // X span
    const rackDepth = CELL_SIZE * 0.85;  // Z span
    const rackHeight = 2.4;              // Y height (3 tiers)
    const postRadius = 0.04;

    // 1. Instanced Mesh for Rack Framework (Metallic Dark Steel)
    const rackFrameGeo = new THREE.BoxGeometry(rackWidth, 0.06, rackDepth);
    const rackMat = new THREE.MeshStandardMaterial({
      color: 0x334155,
      metalness: 0.8,
      roughness: 0.3
    });

    // 3 tiers per rack -> total tier instances = totalRacks * 3
    const totalTierInstances = this.totalRacks * 3;
    const tierInstancedMesh = new THREE.InstancedMesh(rackFrameGeo, rackMat, totalTierInstances);
    tierInstancedMesh.castShadow = true;
    tierInstancedMesh.receiveShadow = true;

    // 2. Instanced Mesh for Storage Crates/Boxes
    const crateGeo = new THREE.BoxGeometry(0.5, 0.45, 0.5);
    const crateMat = new THREE.MeshStandardMaterial({
      color: 0xd97706, // Industrial amber cardboard
      roughness: 0.7,
      metalness: 0.1
    });

    // Up to 6 crates per rack -> total crate instances = totalRacks * 6
    const maxCrates = this.totalRacks * 6;
    this.crateInstancedMesh = new THREE.InstancedMesh(crateGeo, crateMat, maxCrates);
    this.crateInstancedMesh.castShadow = true;
    this.crateInstancedMesh.receiveShadow = true;

    const dummy = new THREE.Object3D();
    let tierIdx = 0;
    let crateIdx = 0;

    // Colors for variety in crates (Cardboard brown, Blue tote, Slate grey)
    const crateColors = [
      new THREE.Color(0xd97706),
      new THREE.Color(0x2563eb),
      new THREE.Color(0x475569)
    ];

    this.shelves.forEach(shelf => {
      const worldPos = gridToWorld(shelf.gridPos[0], shelf.gridPos[1]);

      // Place 3 horizontal shelf tiers
      for (let level = 0; level < 3; level++) {
        const tierY = 0.3 + level * 0.75;
        dummy.position.set(worldPos.x, tierY, worldPos.z);
        dummy.rotation.set(0, 0, 0);
        dummy.scale.set(1, 1, 1);
        dummy.updateMatrix();
        tierInstancedMesh.setMatrixAt(tierIdx++, dummy.matrix);

        // Place crates on tiers
        const crateOffsets = [
          [-0.3, tierY + 0.25, -0.2],
          [0.3, tierY + 0.25, 0.2],
          [0.0, tierY + 0.25, -0.1]
        ];

        crateOffsets.forEach((off, i) => {
          if (crateIdx < maxCrates && Math.random() > 0.15) { // 85% occupancy
            dummy.position.set(worldPos.x + off[0], off[1], worldPos.z + off[2]);
            dummy.rotation.set(0, (Math.random() - 0.5) * 0.1, 0);
            dummy.scale.set(1, 1, 1);
            dummy.updateMatrix();
            
            this.crateInstancedMesh.setMatrixAt(crateIdx, dummy.matrix);
            this.crateInstancedMesh.setColorAt(crateIdx, crateColors[(crateIdx + i) % crateColors.length]);
            crateIdx++;
          }
        });
      }

      // Add vertical corner posts for each rack
      this.createRackCornerPosts(worldPos, rackWidth, rackDepth, rackHeight, rackMat);
    });

    tierInstancedMesh.instanceMatrix.needsUpdate = true;
    if (this.crateInstancedMesh.instanceColor) this.crateInstancedMesh.instanceColor.needsUpdate = true;
    this.crateInstancedMesh.instanceMatrix.needsUpdate = true;

    this.group.add(tierInstancedMesh);
    this.group.add(this.crateInstancedMesh);
  }

  private createRackCornerPosts(worldPos: THREE.Vector3, width: number, depth: number, height: number, material: THREE.Material): void {
    const postGeo = new THREE.CylinderGeometry(0.03, 0.03, height, 8);
    const halfX = width / 2 - 0.04;
    const halfZ = depth / 2 - 0.04;

    const corners = [
      [-halfX, -halfZ],
      [halfX, -halfZ],
      [-halfX, halfZ],
      [halfX, halfZ]
    ];

    corners.forEach(([cx, cz]) => {
      const postMesh = new THREE.Mesh(postGeo, material);
      postMesh.position.set(worldPos.x + cx, height / 2, worldPos.z + cz);
      postMesh.castShadow = true;
      this.group.add(postMesh);
    });
  }

  private createShelfSignage(): void {
    // Attach LED shelf ID badges to end face of racks facing main aisles
    this.shelves.forEach((shelf, idx) => {
      if (idx % 2 === 0) { // Only place on end-caps for high visual clarity
        const worldPos = gridToWorld(shelf.gridPos[0], shelf.gridPos[1]);
        
        const canvas = document.createElement('canvas');
        canvas.width = 128;
        canvas.height = 64;
        const ctx = canvas.getContext('2d')!;

        ctx.fillStyle = '#0F172A';
        ctx.fillRect(0, 0, 128, 64);
        ctx.strokeStyle = '#38BDF8';
        ctx.lineWidth = 4;
        ctx.strokeRect(2, 2, 124, 60);

        ctx.font = 'bold 26px "JetBrains Mono", monospace';
        ctx.fillStyle = '#38BDF8';
        ctx.textAlign = 'center';
        ctx.textBaseline = 'middle';
        ctx.fillText(shelf.id, 64, 32);

        const texture = new THREE.CanvasTexture(canvas);
        const signMat = new THREE.MeshStandardMaterial({
          map: texture,
          emissive: 0x38bdf8,
          emissiveIntensity: 0.4,
          roughness: 0.2
        });

        const signMesh = new THREE.Mesh(new THREE.PlaneGeometry(0.7, 0.35), signMat);
        signMesh.position.set(worldPos.x, 2.1, worldPos.z + 0.9);
        this.group.add(signMesh);
      }
    });
  }
}
