/**
 * WAREHOUSE STATE & TELEMETRY BACKEND CONTRACT (MARL / QMIX READY)
 * 
 * Policy Action Space: Discrete(4)
 * 0 = UP    (dx = 0, dy = -1)
 * 1 = DOWN  (dx = 0, dy = +1)
 * 2 = LEFT  (dx = -1, dy = 0)
 * 3 = RIGHT (dx = +1, dy = 0)
 * 
 * NO_OP is an environment execution result when an action is overridden,
 * NOT a 5th neural-network policy action.
 */

export type AlgorithmType = 'QMIX' | 'IQL' | 'GREEDY_ASTAR';

export type ControllerType = 'TEST_CONTROLLER' | 'DEMO_CONTROLLER' | 'RL_BACKEND';

export enum Action {
  UP = 0,
  DOWN = 1,
  LEFT = 2,
  RIGHT = 3,
}

export type ExecutedAction = Action | 'NO_OP';

export type ActionDict = Record<string, Action>;

export interface WarehouseConfig {
  gridWidth: number;
  gridHeight: number;
  cellSize: number;
}

export type RobotStatus = 
  | 'IDLE' 
  | 'MOVING' 
  | 'PICKUP' 
  | 'DELIVERING' 
  | 'CHARGING' 
  | 'WARNING' 
  | 'COLLISION';

export type TaskStatus = 
  | 'UNASSIGNED' 
  | 'ASSIGNED' 
  | 'IN_PROGRESS' 
  | 'COMPLETED';

export type CameraMode = 
  | 'OVERVIEW' 
  | 'TOP_DOWN' 
  | 'ROBOT_FOLLOW' 
  | 'TASK_FOLLOW' 
  | 'FREE_ORBIT';

export interface RobotState {
  id: string; // e.g. "R01", "R02"
  position: [number, number]; // [gridX, gridY]
  targetPosition?: [number, number]; // [gridX, gridY]
  rotation: number; // angle in radians
  status: RobotStatus;
  taskId?: string;
  hasCargo: boolean;
  battery: number; // 0-100 visual indicator
  path: [number, number][]; // array of grid coordinates
  safetyAlert: boolean;
  lastRequestedAction?: Action;
  lastExecutedAction?: ExecutedAction;
  actionOverridden?: boolean;
}

export interface TaskState {
  id: string;
  pickupPos: [number, number];
  pickupShelfId?: string;
  deliveryPos: [number, number];
  deliveryZoneId?: string;
  assignedRobotId?: string;
  status: TaskStatus;
  createdTimestep?: number;
  completedTimestep?: number;
}

export interface SafetyZoneState {
  id: string;
  name: string;
  bounds: { minX: number; maxX: number; minY: number; maxY: number };
  activeWarning: boolean;
}

export type ConflictEventType = 
  | 'ROBOT_ROBOT_CONFLICT'
  | 'ROBOT_ROBOT_SWAP'
  | 'MULTI_ROBOT_CONFLICT'
  | 'OBSTACLE_BLOCK'
  | 'BOUNDARY_BLOCK';

export interface ConflictEvent {
  id: string;
  type: ConflictEventType;
  robots: string[];
  position: [number, number];
  timestamp: number;
}

export type CollisionEvent = ConflictEvent;

export interface ActionEventTrace {
  robotId: string;
  timestep: number;
  requestedAction: Action;
  executedAction: ExecutedAction;
  wasOverridden: boolean;
  reason?: string;
}

export interface WarehouseMetrics {
  activeRobots: number;
  totalRobots: number;
  activeTasks: number;
  completedTasks: number;
  throughputPerHour: number;
  averageDeliveryTimeSec: number;
  collisionAttempts: number;
  conflictingProposals: number;
  safetyOverrides: number;
  physicalCollisions: number;
  totalCollisions: number;
  collisionCount: number;
  blockedActionCount: number;
  overrideCount: number;
  movementSteps: number;
  idleRobotSteps: number;
  timestep: number;
  episode: number;
  reward?: number;
  fps: number;
}

export interface ShelfState {
  id: string;
  gridPos: [number, number];
  levels: number;
  aisleId: string;
}

export interface WarehouseState {
  timestep: number;
  episode: number;
  algorithm: AlgorithmType;
  controllerType: ControllerType;
  config: WarehouseConfig;
  robots: RobotState[];
  shelves: ShelfState[];
  tasks: TaskState[];
  safetyZones: SafetyZoneState[];
  recentCollisions: ConflictEvent[];
  lastActionTraces?: ActionEventTrace[];
  metrics: WarehouseMetrics;
  isDemoMode: boolean;
  safetyOverrideActive: boolean;
  isConnectedToBackend: boolean;
}

export interface StateProvider {
  getState(): WarehouseState;
  subscribe(callback: (state: WarehouseState) => void): () => void;
  setSpeed(multiplier: number): void;
  pause(): void;
  resume(): void;
  reset(): void;
  selectRobot(robotId: string | null): void;
  selectTask(taskId: string | null): void;
}
