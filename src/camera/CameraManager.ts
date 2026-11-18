import * as THREE from 'three';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js';
import { CameraMode } from '../types/warehouse';
import { WAREHOUSE_WIDTH } from '../scene/gridConfig';

export class CameraManager {
  public camera: THREE.PerspectiveCamera;
  public controls: OrbitControls;
  private currentMode: CameraMode = 'OVERVIEW';
  private targetObject: THREE.Object3D | null = null;
  // A little breathing room keeps the full 40m x 40m grid visible on short
  // landscape viewports while preserving enough scale for numeric markers.
  private readonly topDownPosition = new THREE.Vector3(0, WAREHOUSE_WIDTH * 1.7, 0.001);
  private readonly topDownTarget = new THREE.Vector3(0, 0, 0);

  constructor(container: HTMLElement) {
    const aspect = container.clientWidth / container.clientHeight;
    this.camera = new THREE.PerspectiveCamera(45, aspect, 0.5, 500);

    this.controls = new OrbitControls(this.camera, container);
    // The demo is intentionally a fixed presentation: orbiting and zooming make
    // grid positions, paths, and safety markers harder to compare frame to frame.
    this.controls.enabled = false;
    this.controls.enableDamping = false;

    this.setCameraMode('TOP_DOWN', false);
  }

  public resize(width: number, height: number): void {
    this.camera.aspect = width / height;
    this.camera.updateProjectionMatrix();
    this.controls.update();
  }

  public getMode(): CameraMode {
    return this.currentMode;
  }

  public setCameraMode(mode: CameraMode, animate: boolean = true): void {
    // Keep this API for existing callers, but do not expose alternate views in
    // the presentation. All requested modes resolve to the same top-down shot.
    this.currentMode = 'TOP_DOWN';
    this.targetObject = null;
    this.camera.position.copy(this.topDownPosition);
    this.controls.target.copy(this.topDownTarget);
    this.camera.lookAt(this.topDownTarget);
  }

  public focusOnTarget(targetPos: THREE.Vector3, offset: THREE.Vector3 = new THREE.Vector3(6, 7, 8), animate: boolean = true): void {
    // Retained as a no-op compatibility hook. Selection changes path emphasis,
    // never the camera framing.
    void targetPos;
    void offset;
    void animate;
  }

  public trackObject(object: THREE.Object3D): void {
    void object;
    this.targetObject = null;
    this.currentMode = 'TOP_DOWN';
  }

  public update(): void {
    this.camera.position.copy(this.topDownPosition);
    this.controls.target.copy(this.topDownTarget);
    this.camera.lookAt(this.topDownTarget);
  }
}
