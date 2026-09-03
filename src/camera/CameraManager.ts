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
  private isTransitioning: boolean = false;

  // Preset camera positions & targets
  private readonly presets: Record<CameraMode, { pos: THREE.Vector3; target: THREE.Vector3 }> = {
    OVERVIEW: {
      pos: new THREE.Vector3(WAREHOUSE_WIDTH * 0.95, WAREHOUSE_WIDTH * 0.85, WAREHOUSE_WIDTH * 0.95),
      target: new THREE.Vector3(0, 0, 0)
    },
    TOP_DOWN: {
      pos: new THREE.Vector3(0, WAREHOUSE_WIDTH * 1.2, 0.001), // Slight offset to prevent gimbal lock
      target: new THREE.Vector3(0, 0, 0)
    },
    ROBOT_FOLLOW: {
      pos: new THREE.Vector3(5, 6, 8),
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
    this.controls.maxPolarAngle = Math.PI / 2 - 0.02; // Prevent camera clipping below floor
    this.controls.minDistance = 5;
    this.controls.maxDistance = 120;

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

    if (!animate) {
      this.camera.position.copy(preset.pos);
      this.controls.target.copy(preset.target);
      this.controls.update();
      return;
    }

    this.animateCameraTo(preset.pos, preset.target, 1.2);
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

  public trackObject(object: THREE.Object3D): void {
    this.targetObject = object;
    this.currentMode = 'ROBOT_FOLLOW';
  }

  public update(): void {
    if (this.currentMode === 'ROBOT_FOLLOW' && this.targetObject && !this.isTransitioning) {
      const objPos = this.targetObject.position;
      const currentTarget = this.controls.target;
      
      // Smoothly follow target position without jarring
      currentTarget.lerp(new THREE.Vector3(objPos.x, objPos.y + 0.6, objPos.z), 0.1);
      
      // Maintain camera relative offset
      const offset = new THREE.Vector3(6, 8, 10);
      const desiredCamPos = currentTarget.clone().add(offset);
      this.camera.position.lerp(desiredCamPos, 0.08);
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
