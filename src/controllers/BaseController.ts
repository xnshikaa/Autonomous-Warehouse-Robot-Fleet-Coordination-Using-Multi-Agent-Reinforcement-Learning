import { WarehouseState, ActionDict, ControllerType } from '../types/warehouse';

export interface IController {
  readonly type: ControllerType;
  getActions(state: WarehouseState): ActionDict;
}
