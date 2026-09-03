import { WarehouseState, Action, RobotState } from '../types/warehouse';

export interface IObservationBuilder {
  buildObservations(state: WarehouseState): Record<string, number[]>;
  getAvailableActions(robotId: string, state: WarehouseState): Action[];
}

/**
 * Standard 50-Dimensional Observation Vector Builder (EPyMARL / QMIX Compatible)
 * 
 * Shape per robot: (50,)
 * - 45 values: 3x3 local occupancy grid (9 cells x 5 one-hot categories)
 *   Categories: 0=EMPTY_FLOOR, 1=WALL_OR_BOUNDARY, 2=SHELF, 3=ROBOT, 4=HUMAN_CORRIDOR
 *   (Center cell is represented as EMPTY_FLOOR; no SELF category)
 * - 3 values: normalized delta_row, normalized delta_col, normalized distance
 * - 2 values: has_task (1.0/0.0), carrying_item (1.0/0.0)
 */
export class DefaultObservationBuilder implements IObservationBuilder {
  public buildObservations(state: WarehouseState): Record<string, number[]> {
    const observations: Record<string, number[]> = {};
    const { gridWidth, gridHeight } = state.config;

    const shelfSet = new Set<string>(
      state.shelves.map(s => `${s.gridPos[0]},${s.gridPos[1]}`)
    );

    // Map current robot positions for fast lookup
    const robotPosMap = new Map<string, string>();
    state.robots.forEach(r => {
      robotPosMap.set(`${r.position[0]},${r.position[1]}`, r.id);
    });

    state.robots.forEach(robot => {
      const [rx, ry] = robot.position;
      const obsVector: number[] = [];

      // 1. 3x3 Occupancy Window (9 cells x 5 one-hot categories = 45 values)
      for (let dy = -1; dy <= 1; dy++) {
        for (let dx = -1; dx <= 1; dx++) {
          const cx = rx + dx;
          const cy = ry + dy;
          const key = `${cx},${cy}`;

          let category = 0; // Default: EMPTY_FLOOR

          if (dx === 0 && dy === 0) {
            // Robot's own center cell represented as EMPTY_FLOOR
            category = 0;
          } else if (cx < 0 || cx >= gridWidth || cy < 0 || cy >= gridHeight) {
            category = 1; // WALL_OR_BOUNDARY
          } else if (shelfSet.has(key)) {
            category = 2; // SHELF
          } else if (robotPosMap.has(key)) {
            category = 3; // ROBOT
          } else if (cx >= 8 && cx <= 10 && cy >= 8 && cy <= 10) {
            category = 4; // HUMAN_CORRIDOR
          } else {
            category = 0; // EMPTY_FLOOR
          }

          // One-hot encode category (5 dimensions)
          const oneHot = [0, 0, 0, 0, 0];
          oneHot[category] = 1;
          obsVector.push(...oneHot);
        }
      }

      // 2. Task Bearing & Distance (3 values)
      let normDeltaRow = 0.0;
      let normDeltaCol = 0.0;
      let normDistance = 0.0;

      const task = state.tasks.find(t => t.id === robot.taskId);
      if (task && task.status !== 'COMPLETED') {
        const [tx, ty] = robot.hasCargo ? task.deliveryPos : task.pickupPos;
        normDeltaRow = (ty - ry) / Math.max(1, gridHeight - 1);
        normDeltaCol = (tx - rx) / Math.max(1, gridWidth - 1);
        normDistance = Math.sqrt(normDeltaRow * normDeltaRow + normDeltaCol * normDeltaCol);
      }

      obsVector.push(normDeltaRow, normDeltaCol, normDistance);

      // 3. Task Status Flags (2 values)
      const hasTaskFlag = task && task.status !== 'COMPLETED' ? 1.0 : 0.0;
      const carryingItemFlag = robot.hasCargo ? 1.0 : 0.0;

      obsVector.push(hasTaskFlag, carryingItemFlag);

      // Verify total shape is exactly 50
      observations[robot.id] = obsVector;
    });

    return observations;
  }

  public getAvailableActions(robotId: string, state: WarehouseState): Action[] {
    const robot = state.robots.find(r => r.id === robotId);
    if (!robot) return [Action.UP];

    // Discrete(4) policy action space: UP(0), DOWN(1), LEFT(2), RIGHT(3)
    return [Action.UP, Action.DOWN, Action.LEFT, Action.RIGHT];
  }
}
