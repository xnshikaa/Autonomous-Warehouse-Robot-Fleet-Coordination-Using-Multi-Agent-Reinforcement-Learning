import * as THREE from 'three';
import gsap from 'gsap';
import { AMRModel } from './AMRModel';
import { RobotState } from '../types/warehouse';
import { gridToWorld } from './gridConfig';

interface RobotAnimationState {
  currentPos: THREE.Vector3;
  targetPos: THREE.Vector3;
  currentRotation: number;
  targetRotation: number;
  tweenPos?: gsap.core.Tween;
  tweenRot?: gsap.core.Tween;
}

export class RobotSystem {
  public group: THREE.Group;
  private robotMap: Map<string, AMRModel> = new Map();
  private animMap: Map<string, RobotAnimationState> = new Map();

  constructor(scene: THREE.Scene) {
    this.group = new THREE.Group();
    this.group.name = 'RobotSystem';

    scene.add(this.group);
  }

  /**
   * Syncs state array and animates positional/rotational transitions smoothly
   */
  public updateRobots(states: RobotState[], moveDurationSec: number = 0.8): void {
    const activeIds = new Set<string>();

    states.forEach(state => {
      activeIds.add(state.id);
      let amr = this.robotMap.get(state.id);
      let anim = this.animMap.get(state.id);

      const targetWorld = gridToWorld(state.position[0], state.position[1]);

      if (!amr || !anim) {
        // Instantiate model
        amr = new AMRModel(state.id);
        this.robotMap.set(state.id, amr);
        this.group.add(amr.mesh);

        amr.mesh.position.copy(targetWorld);
        amr.mesh.rotation.y = state.rotation;

        anim = {
          currentPos: targetWorld.clone(),
          targetPos: targetWorld.clone(),
          currentRotation: state.rotation,
          targetRotation: state.rotation
        };
        this.animMap.set(state.id, anim);
      } else {
        // Smooth positional transition if target changed
        if (!anim.targetPos.equals(targetWorld)) {
          // Calculate heading angle towards target cell
          const dx = targetWorld.x - anim.currentPos.x;
          const dz = targetWorld.z - anim.currentPos.z;

          if (Math.abs(dx) > 0.01 || Math.abs(dz) > 0.01) {
            // Three.js angle facing forward along Z
            let headingAngle = Math.atan2(dx, dz);
            
            // Shortest rotational path (prevent 360 spin bug when wrapping PI)
            let diff = headingAngle - anim.currentRotation;
            while (diff < -Math.PI) diff += Math.PI * 2;
            while (diff > Math.PI) diff -= Math.PI * 2;
            const finalRot = anim.currentRotation + diff;

            anim.targetRotation = finalRot;

            // Kill active rotation tween & animate turning
            if (anim.tweenRot) anim.tweenRot.kill();
            anim.tweenRot = gsap.to(amr.mesh.rotation, {
              y: finalRot,
              duration: moveDurationSec * 0.35,
              ease: 'power2.inOut',
              onUpdate: () => {
                anim!.currentRotation = amr!.mesh.rotation.y;
              }
            });
          }

          anim.targetPos.copy(targetWorld);

          // Kill active position tween & animate movement
          if (anim.tweenPos) anim.tweenPos.kill();
          anim.tweenPos = gsap.to(amr.mesh.position, {
            x: targetWorld.x,
            z: targetWorld.z,
            duration: moveDurationSec,
            ease: 'power1.inOut',
            onUpdate: () => {
              anim!.currentPos.copy(amr!.mesh.position);
            }
          });
        }
      }

      amr.setStatus(state.status);
      amr.setHasCargo(state.hasCargo);
    });

    // Clean up decommissioned robots
    this.robotMap.forEach((amr, id) => {
      if (!activeIds.has(id)) {
        const anim = this.animMap.get(id);
        if (anim?.tweenPos) anim.tweenPos.kill();
        if (anim?.tweenRot) anim.tweenRot.kill();
        
        this.group.remove(amr.mesh);
        this.robotMap.delete(id);
        this.animMap.delete(id);
      }
    });
  }

  public getRobotMesh(id: string): THREE.Group | null {
    const amr = this.robotMap.get(id);
    return amr ? amr.mesh : null;
  }

  public update(deltaTimeSec: number): void {
    this.robotMap.forEach(amr => amr.update(deltaTimeSec));
  }
}
