import * as THREE from 'three';
import { CameraManager } from '../camera/CameraManager';
import { IndustrialLighting } from './IndustrialLighting';
import { WarehouseFloor } from './WarehouseFloor';
import { ShelfSystem } from './ShelfSystem';
import { ZoneSystem } from './ZoneSystem';
import { RobotSystem } from './RobotSystem';
import { PathSystem, PathVisibilityMode } from './PathSystem';
import { TaskSystem } from './TaskSystem';
import { SafetySystem } from './SafetySystem';
import { WarehouseState } from '../types/warehouse';

export class WarehouseSceneController {
  public scene: THREE.Scene;
  public renderer: THREE.WebGLRenderer;
  public cameraManager: CameraManager;
  public lighting: IndustrialLighting;
  public floor: WarehouseFloor;
  public shelfSystem: ShelfSystem;
  public zoneSystem: ZoneSystem;
  public robotSystem: RobotSystem;
  public pathSystem: PathSystem;
  public taskSystem: TaskSystem;
  public safetySystem: SafetySystem;

  private container: HTMLElement;
  private animationFrameId: number | null = null;
  private lastTime: number = performance.now();
  private fpsCounter: number = 60;
  private frameCount: number = 0;
  private lastFpsUpdate: number = performance.now();

  private raycaster: THREE.Raycaster = new THREE.Raycaster();
  private mouse: THREE.Vector2 = new THREE.Vector2();

  public onRobotSelected?: (robotId: string | null) => void;
  public onTaskSelected?: (taskId: string | null) => void;

  constructor(container: HTMLElement) {
    this.container = container;

    // 1. Initialize Scene & Industrial Fog
    this.scene = new THREE.Scene();
    this.scene.background = new THREE.Color(0x0b0f17);
    this.scene.fog = new THREE.FogExp2(0x0b0f17, 0.012);

    // 2. Initialize Camera Manager
    this.cameraManager = new CameraManager(container);

    // 3. Initialize WebGL Renderer
    this.renderer = new THREE.WebGLRenderer({
      antialias: true,
      alpha: false,
      powerPreference: 'high-performance'
    });
    this.renderer.setSize(container.clientWidth, container.clientHeight);
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    this.renderer.shadowMap.enabled = true;
    this.renderer.shadowMap.type = THREE.PCFSoftShadowMap;
    this.renderer.toneMapping = THREE.ACESFilmicToneMapping;
    this.renderer.toneMappingExposure = 1.05;

    container.appendChild(this.renderer.domElement);

    // 4. Initialize Core Scene Systems
    this.lighting = new IndustrialLighting(this.scene);
    this.floor = new WarehouseFloor(this.scene);
    this.shelfSystem = new ShelfSystem(this.scene);
    this.zoneSystem = new ZoneSystem(this.scene);
    this.robotSystem = new RobotSystem(this.scene);
    this.pathSystem = new PathSystem(this.scene);
    this.taskSystem = new TaskSystem(this.scene);
    this.safetySystem = new SafetySystem(this.scene);

    // Event Listeners
    window.addEventListener('resize', this.onResize);
    this.renderer.domElement.addEventListener('pointerdown', this.onPointerDown);

    // Start Loop
    this.start();
  }

  public updateState(state: WarehouseState): void {
    this.robotSystem.updateRobots(state.robots, 0.85);
    this.pathSystem.updatePaths(state.robots);
    this.taskSystem.updateTasks(state.tasks);
    this.safetySystem.updateCollisions(state.recentCollisions);
    this.safetySystem.update(0.016, state.safetyOverrideActive);
  }

  public setPathVisibilityMode(mode: PathVisibilityMode): void {
    this.pathSystem.setVisibilityMode(mode);
  }

  public focusRobot(robotId: string): void {
    const mesh = this.robotSystem.getRobotMesh(robotId);
    if (mesh) {
      this.cameraManager.trackObject(mesh);
      this.pathSystem.setSelectedRobot(robotId);
    }
  }

  private onPointerDown = (event: MouseEvent): void => {
    const rect = this.renderer.domElement.getBoundingClientRect();
    this.mouse.x = ((event.clientX - rect.left) / rect.width) * 2 - 1;
    this.mouse.y = -((event.clientY - rect.top) / rect.height) * 2 + 1;

    this.raycaster.setFromCamera(this.mouse, this.cameraManager.camera);
    const intersects = this.raycaster.intersectObjects(this.robotSystem.group.children, true);

    if (intersects.length > 0) {
      let current: THREE.Object3D | null = intersects[0].object;
      while (current && !current.name.startsWith('AMR_')) {
        current = current.parent;
      }
      if (current && current.name.startsWith('AMR_')) {
        const robotId = current.name.replace('AMR_', '');
        if (this.onRobotSelected) {
          this.onRobotSelected(robotId);
        }
        return;
      }
    }

    if (this.onRobotSelected) this.onRobotSelected(null);
  };

  public onResize = (): void => {
    if (!this.container) return;
    const width = this.container.clientWidth;
    const height = this.container.clientHeight;

    this.renderer.setSize(width, height);
    this.cameraManager.resize(width, height);
  };

  public getFPS(): number {
    return this.fpsCounter;
  }

  public start(): void {
    if (this.animationFrameId !== null) return;
    this.animate();
  }

  public stop(): void {
    if (this.animationFrameId !== null) {
      cancelAnimationFrame(this.animationFrameId);
      this.animationFrameId = null;
    }
  }

  public dispose(): void {
    this.stop();
    window.removeEventListener('resize', this.onResize);
    if (this.renderer.domElement) {
      this.renderer.domElement.removeEventListener('pointerdown', this.onPointerDown);
      if (this.renderer.domElement.parentElement) {
        this.renderer.domElement.parentElement.removeChild(this.renderer.domElement);
      }
    }
    this.renderer.dispose();
  }

  private animate = (): void => {
    this.animationFrameId = requestAnimationFrame(this.animate);

    const now = performance.now();
    const delta = (now - this.lastTime) / 1000;
    this.lastTime = now;

    this.frameCount++;
    if (now - this.lastFpsUpdate >= 1000) {
      this.fpsCounter = Math.round((this.frameCount * 1000) / (now - this.lastFpsUpdate));
      this.frameCount = 0;
      this.lastFpsUpdate = now;
    }

    this.robotSystem.update(delta);
    this.pathSystem.update(delta);
    this.taskSystem.update(delta);
    this.cameraManager.update();

    this.renderer.render(this.scene, this.cameraManager.camera);
  };
}
