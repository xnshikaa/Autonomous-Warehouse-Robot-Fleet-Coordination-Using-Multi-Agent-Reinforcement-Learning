import { IController } from './BaseController';
import { WarehouseState, ActionDict, Action, ControllerType } from '../types/warehouse';

export class DemoController implements IController {
  public readonly type: ControllerType = 'DEMO_CONTROLLER';

  public getActions(state: WarehouseState): ActionDict {
    const actions: ActionDict = {};
    const { gridWidth, gridHeight } = state.config;

    const shelfSet = new Set<string>(
      state.shelves.map(s => `${s.gridPos[0]},${s.gridPos[1]}`)
    );

    state.robots.forEach(robot => {
      const [x, y] = robot.position;
      let targetX = x;
      let targetY = y;

      const task = state.tasks.find(t => t.id === robot.taskId);
      if (task) {
        if (!robot.hasCargo && task.status !== 'COMPLETED') {
          [targetX, targetY] = task.pickupPos;
        } else if (robot.hasCargo) {
          [targetX, targetY] = task.deliveryPos;
        }
      } else if (robot.targetPosition) {
        [targetX, targetY] = robot.targetPosition;
      }

      if ((x === targetX && y === targetY) || shelfSet.has(`${targetX},${targetY}`)) {
        const openAisleCols = [0, 4, 9, 14, 19].filter(c => c < gridWidth);
        const randCol = openAisleCols[Math.floor(Math.random() * openAisleCols.length)];
        const randRow = Math.floor(Math.random() * (gridHeight - 2)) + 1;
        targetX = randCol;
        targetY = randRow;
        robot.targetPosition = [targetX, targetY];
      }

      const path = this.findBFSPath([x, y], [targetX, targetY], gridWidth, gridHeight, shelfSet);

      if (path && path.length > 1) {
        const nextCell = path[1];
        const [nx, ny] = nextCell;

        if (ny === y - 1 && nx === x) actions[robot.id] = Action.UP; // 0
        else if (ny === y + 1 && nx === x) actions[robot.id] = Action.DOWN; // 1
        else if (nx === x - 1 && ny === y) actions[robot.id] = Action.LEFT; // 2
        else if (nx === x + 1 && ny === y) actions[robot.id] = Action.RIGHT; // 3
        else actions[robot.id] = Action.UP;
      } else {
        // Default policy action choice from Discrete(4) if path not available
        actions[robot.id] = Action.UP;
      }
    });

    return actions;
  }

  private findBFSPath(
    start: [number, number],
    goal: [number, number],
    width: number,
    height: number,
    shelves: Set<string>
  ): [number, number][] | null {
    const startKey = `${start[0]},${start[1]}`;
    const goalKey = `${goal[0]},${goal[1]}`;

    if (startKey === goalKey) return [start];

    const queue: [number, number][] = [start];
    const parent = new Map<string, [number, number] | null>();
    const visited = new Set<string>([startKey]);
    parent.set(startKey, null);

    const neighbors = [
      [0, -1], // UP
      [0, 1],  // DOWN
      [-1, 0], // LEFT
      [1, 0]   // RIGHT
    ];

    while (queue.length > 0) {
      const [cx, cy] = queue.shift()!;
      const currentKey = `${cx},${cy}`;

      if (currentKey === goalKey) {
        const path: [number, number][] = [];
        let curr: [number, number] | null = [cx, cy];
        while (curr) {
          path.unshift(curr);
          const k: string = `${curr[0]},${curr[1]}`;
          curr = parent.get(k) || null;
        }
        return path;
      }

      for (const [dx, dy] of neighbors) {
        const nx = cx + dx;
        const ny = cy + dy;
        const nKey = `${nx},${ny}`;

        if (
          nx >= 0 &&
          nx < width &&
          ny >= 0 &&
          ny < height &&
          !visited.has(nKey) &&
          (!shelves.has(nKey) || nKey === goalKey)
        ) {
          visited.add(nKey);
          parent.set(nKey, [cx, cy]);
          queue.push([nx, ny]);
        }
      }
    }

    return null;
  }
}
