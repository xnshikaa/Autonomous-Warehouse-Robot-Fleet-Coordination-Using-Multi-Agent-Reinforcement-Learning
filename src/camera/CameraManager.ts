import * as THREE from 'three';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js';
import gsap from 'gsap';
import { CameraMode } from '../types/warehouse';
import { WAREHOUSE_WIDTH } from '../scene/gridConfig';

export class CameraManager {
  public camera: THREE.PerspectiveCamera;
  public controls: OrbitControls;
  private currentMode: CameraMode = 'OVERVIEW';
  private targetObject: THREE.Object3D | null = null;
  private readonly followTarget = new THREE.Vector3();
  private readonly followDelta = new THREE.Vector3();
  private isTransitioning: boolean = false;

  // Preset camera positions & targets
  private readonly presets: Record<CameraMode, { pos: THREE.Vector3; target: THREE.Vector3 }> = {
    OVERVIEW: {
      pos: new THREE.Vector3(34, 30, 34),
      target: new THREE.Vector3(0, 0, 0)
    },
    TOP_DOWN: {
      pos: new THREE.Vector3(0, 48, 0.001), // Slight offset to prevent gimbal lock
      target: new THREE.Vector3(0, 0, 0)
    },
    ROBOT_FOLLOW: {
      pos: new THREE.Vector3(12, 14, 12),
      target: new THREE.Vector3(0, 0.5, 0)
    },
    TASK_FOLLOW: {
      pos: new THREE.Vector3(10, 12, 14),
      target: new THREE.Vector3(0, 0, 0)
    },
    FREE_ORBIT: {
      pos: new THREE.Vector3(30, 25, 30),
      target: new THREE.Vector3(0, 0, 0)
    }
  };

  constructor(container: HTMLElement) {
    const aspect = container.clientWidth / container.clientHeight;
    this.camera = new THREE.PerspectiveCamera(45, aspect, 0.5, 500);

    this.controls = new OrbitControls(this.camera, container);
    this.controls.enableDamping = true;
    this.controls.dampingFactor = 0.05;
    this.controls.maxPolarAngle = Math.PI / 2 - 0.05; // Prevent camera clipping below floor
    this.controls.minDistance = 8;
    this.controls.maxDistance = 75;

    // Apply default overview preset
    this.setCameraMode('OVERVIEW', false);
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
    this.currentMode = mode;
    this.targetObject = null;

    const preset = this.presets[mode];
    if (!preset) return;

    const position = preset.pos.clone();
    if (mode === 'OVERVIEW' || mode === 'TOP_DOWN') {
      position.multiplyScalar(Math.max(1, 1 / this.camera.aspect));
    }
    if (!animate) {
      this.camera.position.copy(position);
      this.controls.target.copy(preset.target);
      this.controls.update();
      return;
    }

    this.animateCameraTo(position, preset.target, 1.2);
  }

  public focusOnTarget(targetPos: THREE.Vector3, offset: THREE.Vector3 = new THREE.Vector3(6, 7, 8), animate: boolean = true): void {
    const desiredPos = targetPos.clone().add(offset);
    if (!animate) {
      this.camera.position.copy(desiredPos);
      this.controls.target.copy(targetPos);
      this.controls.update();
      return;
    }
    this.animateCameraTo(desiredPos, targetPos, 1.0);
  }

  public trackObject(object: THREE.Object3D, animate: boolean = true): void {
    this.targetObject = object;
    this.currentMode = 'ROBOT_FOLLOW';

    const objPos = new THREE.Vector3();
    object.getWorldPosition(objPos);
    const targetPos = new THREE.Vector3(objPos.x, 0.5, objPos.z);
    const cameraPos = new THREE.Vector3(objPos.x + 9, 12, objPos.z + 9);

    if (animate) {
      this.animateCameraTo(cameraPos, targetPos, 0.8);
    } else {
      this.camera.position.copy(cameraPos);
      this.controls.target.copy(targetPos);
      this.controls.update();
    }
  }

  public stopFollowing(): void {
    if (this.currentMode === 'ROBOT_FOLLOW') {
      this.targetObject = null;
      this.currentMode = 'FREE_ORBIT';
      this.rebaseFollowView();
    }
  }

  public rebaseFollowView(): void {
    gsap.killTweensOf(this.camera.position);
    gsap.killTweensOf(this.controls.target);
    this.isTransitioning = false;
    this.controls.enabled = true;
  }

  public resetView(): void {
    if (this.currentMode === 'ROBOT_FOLLOW' && this.targetObject?.parent) {
      this.trackObject(this.targetObject);
    } else {
      this.setCameraMode('OVERVIEW');
    }
  }

  public dispose(): void {
    gsap.killTweensOf(this.camera.position);
    gsap.killTweensOf(this.controls.target);
    this.controls.dispose();
  }

  public update(deltaSeconds = 1 / 60): void {
    if (this.currentMode === 'ROBOT_FOLLOW' && this.targetObject && !this.isTransitioning) {
      if (!this.targetObject.parent) {
        this.stopFollowing();
      } else {
        this.targetObject.getWorldPosition(this.followTarget);
        this.followTarget.y = 0.5;
        this.followDelta.copy(this.followTarget).sub(this.controls.target)
          .multiplyScalar(1 - Math.exp(-3 * Math.min(deltaSeconds, 0.1)));
        this.controls.target.add(this.followDelta);
        this.camera.position.add(this.followDelta);
      }
    }

    this.controls.update();
  }

  private animateCameraTo(targetPos: THREE.Vector3, lookAtTarget: THREE.Vector3, durationSec: number = 1.0): void {
    this.isTransitioning = true;
    this.controls.enabled = false;

    gsap.killTweensOf(this.camera.position);
    gsap.killTweensOf(this.controls.target);

    gsap.to(this.camera.position, {
      x: targetPos.x,
      y: targetPos.y,
      z: targetPos.z,
      duration: durationSec,
      ease: 'power2.inOut'
    });

    gsap.to(this.controls.target, {
      x: lookAtTarget.x,
      y: lookAtTarget.y,
      z: lookAtTarget.z,
      duration: durationSec,
      ease: 'power2.inOut',
      onUpdate: () => {
        this.controls.update();
      },
      onComplete: () => {
        this.isTransitioning = false;
        this.controls.enabled = true;
      }
    });
  }
}
