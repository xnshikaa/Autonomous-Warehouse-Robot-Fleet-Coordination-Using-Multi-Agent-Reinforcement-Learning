import {
  WarehouseState,
  WarehouseConfig,
  RobotState,
  TaskState,
  ShelfState,
  Action,
  ExecutedAction,
  ActionDict,
  ConflictEvent,
  ConflictEventType,
  ActionEventTrace,
  WarehouseMetrics,
  AlgorithmType,
  ControllerType
} from '../types/warehouse';
import { IObservationBuilder, DefaultObservationBuilder } from './ObservationBuilder';
import { TaskQueue } from '../taskManagement/TaskQueue';
import { RobotAssignment } from '../taskManagement/RobotAssignment';

export class WarehouseEnvironment {
  public readonly action_space = {
    n: 4,
    shape: [4],
    type: 'Discrete'
  };

  public readonly observation_space = {
    shape: [50],
    dtype: 'float32'
  };

  private config: WarehouseConfig;
  private state: WarehouseState;
  private observationBuilder: IObservationBuilder;
  private shelfMap: Set<string> = new Set();
  private taskQueue: TaskQueue = new TaskQueue();
  private robotAssignment: RobotAssignment = new RobotAssignment(this.taskQueue);

  constructor(
    config: WarehouseConfig = { gridWidth: 20, gridHeight: 20, cellSize: 2.0 },
    initialFleetSize: number = 10,
    observationBuilder?: IObservationBuilder
  ) {
    this.config = config;
    this.observationBuilder = observationBuilder || new DefaultObservationBuilder();

    const shelves = this.createDefaultShelves(config);
    shelves.forEach(s => this.shelfMap.add(`${s.gridPos[0]},${s.gridPos[1]}`));

    this.state = {
      timestep: 0,
      episode: 1,
      algorithm: 'QMIX',
      controllerType: 'DEMO_CONTROLLER',
      config: { ...this.config },
      robots: [],
      shelves,
      tasks: [],
      safetyZones: [
        {
          id: 'HZ01',
          name: 'HUMAN WORK ZONE',
          bounds: { minX: 8, maxX: 10, minY: 8, maxY: 10 },
          activeWarning: false
        }
      ],
      recentCollisions: [],
      lastActionTraces: [],
      metrics: {
        activeRobots: initialFleetSize,
        totalRobots: initialFleetSize,
        activeTasks: 0,
        completedTasks: 0,
        throughputPerHour: 0,
        averageDeliveryTimeSec: 0,
        collisionAttempts: 0,
        conflictingProposals: 0,
        safetyOverrides: 0,
        physicalCollisions: 0,
        totalCollisions: 0,
        collisionCount: 0,
        blockedActionCount: 0,
        overrideCount: 0,
        movementSteps: 0,
        idleRobotSteps: 0,
        timestep: 0,
        episode: 1,
        reward: 0,
        fps: 60
      },
      isDemoMode: true,
      safetyOverrideActive: false,
      isConnectedToBackend: false
    };

    this.initFleet(initialFleetSize);
    this.initTasks();
  }

  public reset(): WarehouseState {
    this.state.timestep = 0;
    this.state.episode++;
    this.state.recentCollisions = [];
    this.state.lastActionTraces = [];
    this.state.metrics.timestep = 0;
    this.state.metrics.episode = this.state.episode;
    this.state.metrics.completedTasks = 0;
    this.state.metrics.collisionAttempts = 0;
    this.state.metrics.conflictingProposals = 0;
    this.state.metrics.safetyOverrides = 0;
    this.state.metrics.physicalCollisions = 0;
    this.state.metrics.totalCollisions = 0;
    this.state.metrics.collisionCount = 0;
    this.state.metrics.blockedActionCount = 0;
    this.state.metrics.overrideCount = 0;
    this.state.metrics.movementSteps = 0;
    this.state.metrics.idleRobotSteps = 0;

    this.initFleet(this.state.robots.length || 10);
    this.initTasks();
    return this.getState();
  }

  public getState(): WarehouseState {
    return JSON.parse(JSON.stringify(this.state));
  }

  public getObservations(): Record<string, number[]> {
    return this.observationBuilder.buildObservations(this.state);
  }

  public getAvailableActions(robotId: string): Action[] {
    return this.observationBuilder.getAvailableActions(robotId, this.state);
  }

  public setControllerType(controllerType: ControllerType): void {
    this.state.controllerType = controllerType;
  }

  public setAlgorithm(algorithm: AlgorithmType): void {
    this.state.algorithm = algorithm;
  }

  public setFleetSize(size: number): void {
    this.state.metrics.totalRobots = size;
    this.state.metrics.activeRobots = size;
    this.initFleet(size);
    this.initTasks();
  }

  public setRobots(robots: RobotState[]): void {
    this.state.robots = robots.map(r => ({ ...r }));
    this.state.metrics.totalRobots = robots.length;
    this.state.metrics.activeRobots = robots.length;
  }

  public setTasks(tasks: TaskState[]): void {
    this.state.tasks = tasks.map(t => ({ ...t }));
    this.state.metrics.activeTasks = tasks.filter(t => t.status !== 'COMPLETED').length;
  }

  /**
   * Core Step Execution Pipeline:
   * 1. currentState snapshot
   * 2. proposedActions for ALL robots (Discrete 4: UP=0, DOWN=1, LEFT=2, RIGHT=3)
   * 3. proposedPositions for ALL robots simultaneously from currentState
   * 4. Resolution Pipeline (Boundary -> Shelf -> Swap -> Same-cell target occupancy)
   * 5. executedActions for ALL robots (Action or NO_OP)
   * 6. Apply ALL valid movements simultaneously
   * 7. newState & raw metrics update
   */
  public step(requestedActions: ActionDict): WarehouseState {
    this.state.timestep++;
    this.state.metrics.timestep = this.state.timestep;

    const robots = this.state.robots;
    const n = robots.length;
    const { gridWidth, gridHeight } = this.config;

    // 1 & 2. Proposed policy actions for ALL robots (Default UP=0 if unassigned)
    const proposedActions: Record<string, Action> = {};
    robots.forEach(r => {
      proposedActions[r.id] = requestedActions[r.id] !== undefined ? requestedActions[r.id] : Action.UP;
    });

    // 3. Compute proposed positions from CURRENT state simultaneously
    const proposedPositions: Record<string, [number, number]> = {};
    robots.forEach(r => {
      const [x, y] = r.position;
      const act = proposedActions[r.id];
      let nx = x;
      let ny = y;

      if (act === Action.UP) ny = y - 1;
      else if (act === Action.DOWN) ny = y + 1;
      else if (act === Action.LEFT) nx = x - 1;
      else if (act === Action.RIGHT) nx = x + 1;

      proposedPositions[r.id] = [nx, ny];
    });

    // 4. Resolution Pipeline
    const executedActions: Record<string, ExecutedAction> = { ...proposedActions };
    const conflictEvents: ConflictEvent[] = [];
    const traces: ActionEventTrace[] = [];

    let stepCollisionAttempts = 0;
    let stepConflictingProposals = 0;
    let stepSafetyOverrides = 0;

    // Step A: Boundary Check
    robots.forEach(r => {
      const [nx, ny] = proposedPositions[r.id];
      if (nx < 0 || nx >= gridWidth || ny < 0 || ny >= gridHeight) {
        executedActions[r.id] = 'NO_OP';
        stepCollisionAttempts++;
        stepSafetyOverrides++;
        conflictEvents.push({
          id: `EV_BOUND_${this.state.timestep}_${r.id}`,
          type: 'BOUNDARY_BLOCK',
          robots: [r.id],
          position: [r.position[0], r.position[1]],
          timestamp: Date.now()
        });
      }
    });

    // Step B: Shelf / Obstacle Check
    robots.forEach(r => {
      if (executedActions[r.id] === 'NO_OP') return;
      const [nx, ny] = proposedPositions[r.id];
      if (this.shelfMap.has(`${nx},${ny}`)) {
        executedActions[r.id] = 'NO_OP';
        stepCollisionAttempts++;
        stepSafetyOverrides++;
        conflictEvents.push({
          id: `EV_OBS_${this.state.timestep}_${r.id}`,
          type: 'OBSTACLE_BLOCK',
          robots: [r.id],
          position: [nx, ny],
          timestamp: Date.now()
        });
      }
    });

    // Step C: Position Swap Conflict Check (R1 -> R2.curr && R2 -> R1.curr)
    for (let i = 0; i < n; i++) {
      for (let j = i + 1; j < n; j++) {
        const r1 = robots[i];
        const r2 = robots[j];

        const r1Curr = `${r1.position[0]},${r1.position[1]}`;
        const r2Curr = `${r2.position[0]},${r2.position[1]}`;

        const r1Prop = `${proposedPositions[r1.id][0]},${proposedPositions[r1.id][1]}`;
        const r2Prop = `${proposedPositions[r2.id][0]},${proposedPositions[r2.id][1]}`;

        if (r1Prop === r2Curr && r2Prop === r1Curr && r1Curr !== r2Curr) {
          executedActions[r1.id] = 'NO_OP';
          executedActions[r2.id] = 'NO_OP';

          stepCollisionAttempts += 2;
          stepConflictingProposals += 1;
          stepSafetyOverrides += 2;

          conflictEvents.push({
            id: `EV_SWAP_${this.state.timestep}_${r1.id}_${r2.id}`,
            type: 'ROBOT_ROBOT_SWAP',
            robots: [r1.id, r2.id],
            position: [r1.position[0], r1.position[1]],
            timestamp: Date.now()
          });
        }
      }
    }

    // Step D: Same-Cell & Target Occupancy Check (Deterministic Convergence Loop)
    let changed = true;
    let iterations = 0;

    while (changed && iterations < 10) {
      changed = false;
      iterations++;

      const targetPosMap: Record<string, [number, number]> = {};
      robots.forEach(r => {
        const [x, y] = r.position;
        const act = executedActions[r.id];
        let tx = x;
        let ty = y;
        if (act === Action.UP) ty = y - 1;
        else if (act === Action.DOWN) ty = y + 1;
        else if (act === Action.LEFT) tx = x - 1;
        else if (act === Action.RIGHT) tx = x + 1;
        targetPosMap[r.id] = [tx, ty];
      });

      const cellOccupancy = new Map<string, string[]>();
      robots.forEach(r => {
        const key = `${targetPosMap[r.id][0]},${targetPosMap[r.id][1]}`;
        if (!cellOccupancy.has(key)) cellOccupancy.set(key, []);
        cellOccupancy.get(key)!.push(r.id);
      });

      const sortedCellKeys = Array.from(cellOccupancy.keys()).sort();
      sortedCellKeys.forEach(cellKey => {
        const requestingRobots = cellOccupancy.get(cellKey)!.sort();
        if (requestingRobots.length > 1) {
          const movingRobots = requestingRobots.filter(id => executedActions[id] !== 'NO_OP');
          if (movingRobots.length > 0) {
            movingRobots.forEach(id => {
              executedActions[id] = 'NO_OP';
              changed = true;
              stepCollisionAttempts++;
              stepSafetyOverrides++;
            });

            stepConflictingProposals++;

            const [cx, cy] = cellKey.split(',').map(Number);
            const conflictType: ConflictEventType = requestingRobots.length > 2 
              ? 'MULTI_ROBOT_CONFLICT' 
              : 'ROBOT_ROBOT_CONFLICT';

            conflictEvents.push({
              id: `EV_CELL_${this.state.timestep}_${requestingRobots.join('_')}`,
              type: conflictType,
              robots: requestingRobots,
              position: [cx, cy],
              timestamp: Date.now()
            });
          }
        }
      });
    }

    // 5 & 6. Apply ALL executed actions simultaneously & update robot states
    let stepMovementCount = 0;
    let stepIdleCount = 0;
    let stepBlockedCount = 0;

    this.state.robots = robots.map(r => {
      const reqAct = proposedActions[r.id];
      const execAct = executedActions[r.id];
      const wasOverridden = execAct === 'NO_OP';

      if (wasOverridden) stepBlockedCount++;

      let [x, y] = r.position;
      let rotation = r.rotation;

      if (execAct === Action.UP) {
        y -= 1;
        rotation = Math.PI;
        stepMovementCount++;
      } else if (execAct === Action.DOWN) {
        y += 1;
        rotation = 0;
        stepMovementCount++;
      } else if (execAct === Action.LEFT) {
        x -= 1;
        rotation = -Math.PI / 2;
        stepMovementCount++;
      } else if (execAct === Action.RIGHT) {
        x += 1;
        rotation = Math.PI / 2;
        stepMovementCount++;
      } else {
        stepIdleCount++;
      }

      const inSafetyZone = x >= 8 && x <= 10 && y >= 8 && y <= 10;

      let status = r.status;
      let hasCargo = r.hasCargo;

      if (r.taskId) {
        const task = this.state.tasks.find(t => t.id === r.taskId);
        if (task) {
          if (x === task.pickupPos[0] && y === task.pickupPos[1] && !hasCargo) {
            hasCargo = true;
            status = 'PICKUP';
            task.status = 'IN_PROGRESS';
          }else if (
           x === task.deliveryPos[0] &&
           y === task.deliveryPos[1] &&
           hasCargo
          ) {
            this.robotAssignment.completeTask(
              r,
              task,
              this.state.timestep
            );

  hasCargo = false;
  status = 'IDLE';

  this.state.metrics.completedTasks++;
} else {
            status = hasCargo ? 'DELIVERING' : 'MOVING';
          }
        }
      } else {
        status = 'IDLE';
      }

      traces.push({
        robotId: r.id,
        timestep: this.state.timestep,
        requestedAction: reqAct,
        executedAction: execAct,
        wasOverridden,
        reason: wasOverridden ? 'SAFETY_OVERRIDE' : undefined
      });

      return {
        ...r,
        position: [x, y],
        rotation,
        status,
        hasCargo,
        safetyAlert: inSafetyZone,
        lastRequestedAction: reqAct,
        lastExecutedAction: execAct,
        actionOverridden: wasOverridden
      };
    });

    // 7. Update Raw Telemetry Metrics once per timestep
    this.state.recentCollisions = conflictEvents;
    this.state.lastActionTraces = traces;

    this.state.metrics.collisionAttempts += stepCollisionAttempts;
    this.state.metrics.conflictingProposals += stepConflictingProposals;
    this.state.metrics.safetyOverrides += stepSafetyOverrides;
    this.state.metrics.physicalCollisions = 0; // Safety layer prevents physical overlaps

    this.state.metrics.blockedActionCount += stepBlockedCount;
    this.state.metrics.overrideCount += stepBlockedCount;
    this.state.metrics.totalCollisions = this.state.metrics.conflictingProposals;
    this.state.metrics.collisionCount = this.state.metrics.conflictingProposals;
    this.state.metrics.movementSteps += stepMovementCount;
    this.state.metrics.idleRobotSteps += stepIdleCount;
    this.state.metrics.activeTasks = this.state.tasks.filter(t => t.status !== 'COMPLETED').length;
    this.state.safetyOverrideActive = stepBlockedCount > 0;

    return this.getState();
  }

  private createDefaultShelves(config: WarehouseConfig): ShelfState[] {
    const shelves: ShelfState[] = [];
    const rackColumns = [1, 2, 5, 6, 11, 12, 16, 17].filter(c => c < config.gridWidth);
    const rackRows = [2, 3, 4, 5, 6, 7, 12, 13, 14, 15, 16, 17].filter(r => r < config.gridHeight);

    let count = 1;
    rackColumns.forEach(col => {
      rackRows.forEach(row => {
        shelves.push({
          id: `S${count < 10 ? '0' : ''}${count}`,
          gridPos: [col, row],
          levels: 3,
          aisleId: `AISLE 0${Math.floor(col / 4) + 1}`
        });
        count++;
      });
    });

    return shelves;
  }

  private initFleet(size: number): void {
    const robots: RobotState[] = [];
    const mainAisleCols = [0, 4, 9, 14, 19].filter(c => c < this.config.gridWidth);

    for (let i = 1; i <= size; i++) {
      const id = `R${i < 10 ? '0' : ''}${i}`;
      const col = mainAisleCols[(i - 1) % mainAisleCols.length];
      const row = (Math.floor((i - 1) / mainAisleCols.length) * 3 + 2) % (this.config.gridHeight - 1);

      robots.push({
        id,
        position: [col, row],
        rotation: 0,
        status: 'IDLE',
        hasCargo: false,
        battery: 85 + (i * 3) % 15,
        path: [],
        safetyAlert: false,
        lastRequestedAction: Action.UP,
        lastExecutedAction: Action.UP,
        actionOverridden: false
      });
    }

    this.state.robots = robots;
  }

  private initTasks(): void {
  const tasks: TaskState[] = [];

  // Clear any tasks left in the queue from the previous episode
  this.taskQueue.clearQueue();

  // Create tasks based on the current fleet size
  const taskCount = Math.max(
    4,
    Math.floor(this.state.robots.length * 0.7)
  );

  for (let i = 1; i <= taskCount; i++) {
    const id = `T${i < 10 ? '0' : ''}${i}`;

    const pX =
      (i * 3 + 1) % (this.config.gridWidth - 2);

    const pY =
      (i * 2 + 2) % (this.config.gridHeight - 4);

    const dX = [0, 4, 9, 14][i % 4];

    const dY = this.config.gridHeight - 1;

    // Create the task as UNASSIGNED
    const task: TaskState = {
      id,
      pickupPos: [pX, pY],
      pickupShelfId: `S${pX + 1}`,
      deliveryPos: [dX, dY],
      deliveryZoneId: `D0${(i % 4) + 1}`,
      status: 'UNASSIGNED',
      createdTimestep: 0
    };

    // Add task to the warehouse task list
    tasks.push(task);

    // Add the same task to the FIFO waiting queue
    this.taskQueue.addTask(task);
  }

  // Store all generated tasks in the warehouse state
  this.state.tasks = tasks;

  // Assign waiting tasks to available robots
  while (this.taskQueue.hasWaitingTasks()) {
    const result =
      this.robotAssignment.assignNextTask(
        this.state.robots
      );

    // Stop if there are no available robots
    if (!result) {
      break;
    }
  }
}
}