import * as THREE from 'three';
import { RobotStatus } from '../types/warehouse';

export class AMRModel {
  public mesh: THREE.Group;
  public robotId: string;
  
  private statusRingMesh!: THREE.Mesh;
  private statusRingMaterial!: THREE.MeshBasicMaterial;
  private statusBandMesh!: THREE.Mesh;
  private statusBandMaterial!: THREE.MeshStandardMaterial;
  private lidarTowerMesh!: THREE.Mesh;
  private cargoPackageMesh!: THREE.Group;
  private idBadgeMesh!: THREE.Mesh;

  // Status color mappings
  private readonly statusColors: Record<RobotStatus, number> = {
    IDLE: 0x38bdf8,       // Industrial Cyan
    MOVING: 0x38bdf8,     // Industrial Cyan
    PICKUP: 0xf59e0b,     // Amber
    DELIVERING: 0x10b981, // Emerald Green
    CHARGING: 0x8b5cf6,   // Purple
    WARNING: 0xf59e0b,    // Amber Warning
    COLLISION: 0xef4444   // Hazard Red
  };

  constructor(robotId: string = 'R01') {
    this.robotId = robotId;
    this.mesh = new THREE.Group();
    this.mesh.name = `AMR_${robotId}`;

    this.buildChassis();
    this.buildWheels();
    this.buildLidarSensor();
    this.buildStatusRing();
    this.buildCargoPlatform();
    this.buildIdBadge();
  }

  /**
   * Main AMR High-Contrast Industrial Chassis Body
   */
  private buildChassis(): void {
    // 1. Lower main base chassis frame (Dark metallic charcoal)
    const baseGeo = new THREE.BoxGeometry(1.1, 0.16, 1.3);
    const baseMat = new THREE.MeshStandardMaterial({
      color: 0x0f172a,
      metalness: 0.9,
      roughness: 0.2
    });
    const baseMesh = new THREE.Mesh(baseGeo, baseMat);
    baseMesh.position.y = 0.14; // Sits right above wheels
    baseMesh.castShadow = true;
    baseMesh.receiveShadow = true;
    this.mesh.add(baseMesh);

    // 2. High-Visibility Safety Amber Upper Body Shell (Contrast against dark floor)
    const coverGeo = new THREE.BoxGeometry(0.96, 0.14, 1.15);
    const coverMat = new THREE.MeshStandardMaterial({
      color: 0xf59e0b, // Bright Industrial Safety Amber
      metalness: 0.4,
      roughness: 0.3
    });
    const coverMesh = new THREE.Mesh(coverGeo, coverMat);
    coverMesh.position.y = 0.29;
    coverMesh.castShadow = true;
    coverMesh.receiveShadow = true;
    this.mesh.add(coverMesh);

    // 3. Side Metal Guards & Trim
    const guardGeo = new THREE.BoxGeometry(1.12, 0.08, 1.0);
    const guardMat = new THREE.MeshStandardMaterial({
      color: 0x334155,
      metalness: 0.8,
      roughness: 0.3
    });
    const guardMesh = new THREE.Mesh(guardGeo, guardMat);
    guardMesh.position.y = 0.20;
    this.mesh.add(guardMesh);

    // 4. Front Bumper with Black & Yellow Hazard Stripes
    const bumperGeo = new THREE.BoxGeometry(1.08, 0.10, 0.12);
    const bumperMat = new THREE.MeshStandardMaterial({
      color: 0x1e293b,
      metalness: 0.8,
      roughness: 0.3
    });
    const bumperMesh = new THREE.Mesh(bumperGeo, bumperMat);
    bumperMesh.position.set(0, 0.14, 0.65);
    this.mesh.add(bumperMesh);

    // 5. Front Directional LED Headlights
    const lightGeo = new THREE.CylinderGeometry(0.05, 0.05, 0.06, 12);
    const lightMat = new THREE.MeshBasicMaterial({ color: 0xffffff });
    
    const leftHeadlight = new THREE.Mesh(lightGeo, lightMat);
    leftHeadlight.rotation.x = Math.PI / 2;
    leftHeadlight.position.set(-0.38, 0.16, 0.70);
    this.mesh.add(leftHeadlight);

    const rightHeadlight = new THREE.Mesh(lightGeo, lightMat);
    rightHeadlight.rotation.x = Math.PI / 2;
    rightHeadlight.position.set(0.38, 0.16, 0.70);
    this.mesh.add(rightHeadlight);

    // 6. Rear Red Tail Lights
    const tailMat = new THREE.MeshBasicMaterial({ color: 0xef4444 });
    const leftTaillight = new THREE.Mesh(lightGeo, tailMat);
    leftTaillight.rotation.x = Math.PI / 2;
    leftTaillight.position.set(-0.38, 0.16, -0.66);
    this.mesh.add(leftTaillight);

    const rightTaillight = new THREE.Mesh(lightGeo, tailMat);
    rightTaillight.rotation.x = Math.PI / 2;
    rightTaillight.position.set(0.38, 0.16, -0.66);
    this.mesh.add(rightTaillight);
  }

  /**
   * Heavy Duty Industrial Wheels
   */
  private buildWheels(): void {
    const wheelGeo = new THREE.CylinderGeometry(0.10, 0.10, 0.08, 16);
    const wheelMat = new THREE.MeshStandardMaterial({
      color: 0x020617,
      roughness: 0.9,
      metalness: 0.1
    });

    const hubGeo = new THREE.CylinderGeometry(0.05, 0.05, 0.09, 12);
    const hubMat = new THREE.MeshStandardMaterial({
      color: 0x94a3b8,
      metalness: 0.9,
      roughness: 0.1
    });

    const wheelPositions = [
      [-0.52, 0.10, 0.38],
      [0.52, 0.10, 0.38],
      [-0.52, 0.10, -0.38],
      [0.52, 0.10, -0.38]
    ];

    wheelPositions.forEach(([x, y, z]) => {
      const wheelMesh = new THREE.Mesh(wheelGeo, wheelMat);
      wheelMesh.rotation.z = Math.PI / 2;
      wheelMesh.position.set(x, y, z);
      wheelMesh.castShadow = true;
      this.mesh.add(wheelMesh);

      const hubMesh = new THREE.Mesh(hubGeo, hubMat);
      hubMesh.rotation.z = Math.PI / 2;
      hubMesh.position.set(x, y, z);
      this.mesh.add(hubMesh);
    });
  }

  /**
   * Top Rotating LiDAR Sensor Dome
   */
  private buildLidarSensor(): void {
    const towerGeo = new THREE.CylinderGeometry(0.09, 0.11, 0.12, 16);
    const towerMat = new THREE.MeshStandardMaterial({
      color: 0x0f172a,
      metalness: 0.9,
      roughness: 0.1
    });

    const towerMesh = new THREE.Mesh(towerGeo, towerMat);
    towerMesh.position.set(0, 0.42, 0.38);
    this.mesh.add(towerMesh);

    // Rotating LiDAR Lens Header
    const lensGeo = new THREE.CylinderGeometry(0.07, 0.07, 0.07, 12);
    const lensMat = new THREE.MeshStandardMaterial({
      color: 0x38bdf8,
      emissive: 0x38bdf8,
      emissiveIntensity: 0.8
    });

    this.lidarTowerMesh = new THREE.Mesh(lensGeo, lensMat);
    this.lidarTowerMesh.position.set(0, 0.50, 0.38);
    this.mesh.add(this.lidarTowerMesh);
  }

  /**
   * Status Indicator Ring & Body Glow Band
   */
  private buildStatusRing(): void {
    // 1. Ground shadow ring (resting flush on floor at y = 0.02)
    const ringGeo = new THREE.RingGeometry(0.55, 0.65, 32);
    this.statusRingMaterial = new THREE.MeshBasicMaterial({
      color: this.statusColors.IDLE,
      side: THREE.DoubleSide,
      transparent: true,
      opacity: 0.75
    });

    this.statusRingMesh = new THREE.Mesh(ringGeo, this.statusRingMaterial);
    this.statusRingMesh.rotation.x = -Math.PI / 2;
    this.statusRingMesh.position.set(0, 0.02, 0);
    this.mesh.add(this.statusRingMesh);

    // 2. Emissive status LED strip around upper chassis cover
    const bandGeo = new THREE.BoxGeometry(0.98, 0.03, 1.17);
    this.statusBandMaterial = new THREE.MeshStandardMaterial({
      color: this.statusColors.IDLE,
      emissive: this.statusColors.IDLE,
      emissiveIntensity: 0.5,
      metalness: 0.5,
      roughness: 0.2
    });

    this.statusBandMesh = new THREE.Mesh(bandGeo, this.statusBandMaterial);
    this.statusBandMesh.position.set(0, 0.36, 0);
    this.mesh.add(this.statusBandMesh);
  }

  /**
   * Cargo Bed & Box Payload
   */
  private buildCargoPlatform(): void {
    const platformGeo = new THREE.BoxGeometry(0.82, 0.03, 0.72);
    const platformMat = new THREE.MeshStandardMaterial({
      color: 0x334155,
      metalness: 0.7,
      roughness: 0.3
    });

    const platformMesh = new THREE.Mesh(platformGeo, platformMat);
    platformMesh.position.set(0, 0.37, -0.15);
    this.mesh.add(platformMesh);

    // Cargo Box Payload
    this.cargoPackageMesh = new THREE.Group();

    const boxGeo = new THREE.BoxGeometry(0.55, 0.40, 0.55);
    const boxMat = new THREE.MeshStandardMaterial({
      color: 0xd97706, // Cardboard Amber
      roughness: 0.7
    });

    const boxMesh = new THREE.Mesh(boxGeo, boxMat);
    boxMesh.position.set(0, 0.58, -0.15);
    boxMesh.castShadow = true;
    this.cargoPackageMesh.add(boxMesh);

    // Security Strapping
    const strapGeo = new THREE.BoxGeometry(0.57, 0.04, 0.57);
    const strapMat = new THREE.MeshBasicMaterial({ color: 0x38bdf8 });
    const strapMesh = new THREE.Mesh(strapGeo, strapMat);
    strapMesh.position.set(0, 0.58, -0.15);
    this.cargoPackageMesh.add(strapMesh);

    this.cargoPackageMesh.visible = false;
    this.mesh.add(this.cargoPackageMesh);
  }

  /**
   * Crisp ID Badge Tag (e.g. "R01")
   */
  private buildIdBadge(): void {
    const canvas = document.createElement('canvas');
    canvas.width = 128;
    canvas.height = 64;
    const ctx = canvas.getContext('2d')!;

    ctx.fillStyle = '#0F172A';
    ctx.fillRect(0, 0, 128, 64);
    ctx.strokeStyle = '#38BDF8';
    ctx.lineWidth = 4;
    ctx.strokeRect(2, 2, 124, 60);

    ctx.font = 'bold 32px "JetBrains Mono", monospace';
    ctx.fillStyle = '#FFFFFF';
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    ctx.fillText(this.robotId, 64, 32);

    const texture = new THREE.CanvasTexture(canvas);
    const badgeMat = new THREE.MeshStandardMaterial({
      map: texture,
      emissive: 0x38bdf8,
      emissiveIntensity: 0.4,
      side: THREE.DoubleSide
    });

    this.idBadgeMesh = new THREE.Mesh(new THREE.PlaneGeometry(0.6, 0.3), badgeMat);
    this.idBadgeMesh.position.set(0, 0.72, 0); // Hovering cleanly above top LiDAR sensor
    this.idBadgeMesh.rotation.y = 0; // Facing front/top view
    this.mesh.add(this.idBadgeMesh);
  }

  public setStatus(status: RobotStatus): void {
    const colorHex = this.statusColors[status] || this.statusColors.IDLE;
    this.statusRingMaterial.color.setHex(colorHex);
    this.statusBandMaterial.color.setHex(colorHex);
    this.statusBandMaterial.emissive.setHex(colorHex);
  }

  public setHasCargo(hasCargo: boolean): void {
    this.cargoPackageMesh.visible = hasCargo;
  }

  public update(deltaTimeSec: number): void {
    // Rotate top LiDAR scanner
    if (this.lidarTowerMesh) {
      this.lidarTowerMesh.rotation.y += deltaTimeSec * 4.0;
    }
  }
}
