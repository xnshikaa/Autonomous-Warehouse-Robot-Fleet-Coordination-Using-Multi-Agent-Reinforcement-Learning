import { IController } from './BaseController';
import { WarehouseState, ActionDict, Action, ControllerType } from '../types/warehouse';

export class FutureRLController implements IController {
  public readonly type: ControllerType = 'RL_BACKEND';
  private currentActions: ActionDict = {};

  /**
   * Called when external QMIX / EPyMARL policy outputs Discrete(4) action dictionary:
   * 0 = UP, 1 = DOWN, 2 = LEFT, 3 = RIGHT
   */
  public setRLActions(actions: ActionDict): void {
    this.currentActions = { ...actions };
  }

  public getActions(state: WarehouseState): ActionDict {
    const actions: ActionDict = {};
    state.robots.forEach(r => {
      actions[r.id] = this.currentActions[r.id] !== undefined 
        ? this.currentActions[r.id] 
        : Action.UP;
    });
    return actions;
  }
}
