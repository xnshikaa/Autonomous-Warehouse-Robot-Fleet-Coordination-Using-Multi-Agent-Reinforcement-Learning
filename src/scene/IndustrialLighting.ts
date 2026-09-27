import * as THREE from 'three';
import { WAREHOUSE_WIDTH } from './gridConfig';

export class IndustrialLighting {
  public lightGroup: THREE.Group;
  public mainSunLight: THREE.DirectionalLight;
  public ambientLight: THREE.AmbientLight;
  public hemiLight: THREE.HemisphereLight;

  constructor(scene: THREE.Scene) {
    this.lightGroup = new THREE.Group();
    this.lightGroup.name = 'IndustrialLighting';

    // 1. Soft Ambient fill (Dark slate base)
    this.ambientLight = new THREE.AmbientLight(0x1e293b, 1.2);
    this.lightGroup.add(this.ambientLight);

    // 2. Hemispheric light (Industrial cyan sky fill, ground bounce dark grey)
    this.hemiLight = new THREE.HemisphereLight(0x38bdf8, 0x0f172a, 0.8);
    this.hemiLight.position.set(0, 40, 0);
    this.lightGroup.add(this.hemiLight);

    // 3. Primary Key Directional Light (Overhead sunlight with soft shadows)
    this.mainSunLight = new THREE.DirectionalLight(0xfffbeb, 1.8);
    this.mainSunLight.position.set(WAREHOUSE_WIDTH * 0.6, 35, WAREHOUSE_WIDTH * 0.6);
    this.mainSunLight.castShadow = true;
    this.mainSunLight.shadow.mapSize.width = 2048;
    this.mainSunLight.shadow.mapSize.height = 2048;
    this.mainSunLight.shadow.camera.near = 5;
    this.mainSunLight.shadow.camera.far = 80;
    
    const d = WAREHOUSE_WIDTH * 0.7;
    this.mainSunLight.shadow.camera.left = -d;
    this.mainSunLight.shadow.camera.right = d;
    this.mainSunLight.shadow.camera.top = d;
    this.mainSunLight.shadow.camera.bottom = -d;
    this.mainSunLight.shadow.bias = -0.0005;
    this.mainSunLight.shadow.radius = 2.5;

    this.lightGroup.add(this.mainSunLight);

    // 4. Overhead Industrial High-Bay LED Lights Array
    this.createOverheadBayLights();

    scene.add(this.lightGroup);
  }

  private createOverheadBayLights(): void {
    // 4 high-bay spotlights positioned over main aisles for crisp localized highlights
    const bayPositions = [
      [-10, 18, -10],
      [10, 18, -10],
      [-10, 18, 10],
      [10, 18, 10]
    ];

    bayPositions.forEach(([x, y, z]) => {
      const bayLight = new THREE.PointLight(0xe0f2fe, 1.5, 30, 1.8);
      bayLight.position.set(x, y, z);
      this.lightGroup.add(bayLight);

      // Visual light fixture housing mesh
      const fixtureMesh = new THREE.Mesh(
        new THREE.CylinderGeometry(0.6, 0.9, 0.4, 16),
        new THREE.MeshStandardMaterial({
          color: 0x334155,
          metalness: 0.8,
          roughness: 0.2,
          emissive: 0x38bdf8,
          emissiveIntensity: 0.3
        })
      );
      fixtureMesh.position.set(x, y, z);
      this.lightGroup.add(fixtureMesh);
    });
  }
}
