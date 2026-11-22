import * as THREE from 'three';
import { disposeSceneObjects } from './disposeSceneObjects';
import gsap from 'gsap';
import { AMRModel } from './AMRModel';
import { RobotState } from '../types/warehouse';
import { gridToWorld } from './gridConfig';

interface RobotAnimationState {
  targetPos: THREE.Vector3;
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
  public updateRobots(states: RobotState[], moveDurationSec: number = 0.8, resync = false): void {
    const activeIds = new Set<string>();

    states.forEach(state => {
      activeIds.add(state.id);
      let amr = this.robotMap.get(state.id);
      let anim = this.animMap.get(state.id);

      const targetWorld = gridToWorld(state.position[0], state.position[1]);
      const renderRotation = state.rotation ?? 0;

      if (!amr || !anim) {
        // Instantiate model
        amr = new AMRModel(state.id);
        this.robotMap.set(state.id, amr);
        this.group.add(amr.mesh);

        amr.mesh.position.copy(targetWorld);
        amr.mesh.rotation.y = renderRotation;

        anim = {
          targetPos: targetWorld.clone(),
        };
        this.animMap.set(state.id, anim);
      } else if (!anim.targetPos.equals(targetWorld)) {
        const dx = targetWorld.x - anim.targetPos.x;
        const dz = targetWorld.z - anim.targetPos.z;
        anim.tweenPos?.kill();
        anim.tweenRot?.kill();
        if (resync || (dx !== 0 && dz !== 0)) {
          // A missed multi-agent frame cannot safely interpolate a shortcut.
          amr.mesh.position.copy(targetWorld);
          const movement = state.lastSuccessfulMove;
          if (movement) {
            amr.mesh.rotation.y = Math.atan2(movement.to[0] - movement.from[0], movement.to[1] - movement.from[1]);
          }
        } else {
          // Tween directly from current mesh position to targetWorld to prevent backward snapping jitter
          const heading = Math.atan2(dx, dz);
          const delta = heading - amr.mesh.rotation.y;
          const shortestTurn = Math.atan2(Math.sin(delta), Math.cos(delta));
          anim.tweenRot = gsap.to(amr.mesh.rotation, {
            y: amr.mesh.rotation.y + shortestTurn,
            duration: Math.max(0.12, moveDurationSec * 0.35),
            ease: 'power2.inOut',
          });
          anim.tweenPos = gsap.to(amr.mesh.position, {
            x: targetWorld.x,
            z: targetWorld.z,
            duration: Math.max(0.2, moveDurationSec),
            ease: 'power1.inOut',
          });
        }
        anim.targetPos.copy(targetWorld);
      }

      amr.setStatus(state.status);
      amr.setHasCargo(state.hasCargo);
      amr.setSafetyState(Boolean(state.safetyAlert), Boolean(state.actionOverridden));
    });

    // Clean up decommissioned robots
    this.robotMap.forEach((amr, id) => {
      if (!activeIds.has(id)) {
        const anim = this.animMap.get(id);
        if (anim?.tweenPos) anim.tweenPos.kill();
        if (anim?.tweenRot) anim.tweenRot.kill();
        
        this.group.remove(amr.mesh);
        disposeSceneObjects(amr.mesh);
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
