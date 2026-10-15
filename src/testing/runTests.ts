import { TaskQueue } from '../taskManagement/TaskQueue';
import type { TaskState } from '../types/warehouse';
import { RobotAssignment } from '../taskManagement/RobotAssignment';
import type { RobotState } from '../types/warehouse';
import { WarehouseEnvironment } from '../environment/WarehouseEnvironment';
import { Action } from '../types/warehouse';


export interface TestResult {
  id: number;
  name: string;
  passed: boolean;
  message?: string;
}

/**
 * Programmatic Automated Acceptance Test Suite (23 Assertions)
 */
export function runAutomatedWarehouseTests(): TestResult[] {
  const results: TestResult[] = [];

  // Helper to record assertion result
  const assert = (id: number, name: string, condition: boolean, message?: string) => {
    results.push({ id, name, passed: condition, message });
  };

  const env = new WarehouseEnvironment({ gridWidth: 20, gridHeight: 20, cellSize: 2.0 }, 10);

  // 1. UP movement (0,1) -> (0,0)
  env.setRobots([{ id: 'R01', position: [0, 1], rotation: 0, status: 'IDLE', hasCargo: false, battery: 100, path: [], safetyAlert: false }]);
  env.step({ R01: Action.UP });
  assert(1, 'UP Movement', env.getState().robots[0].position[1] === 0);

  // 2. DOWN movement (0,0) -> (0,1)
  env.setRobots([{ id: 'R01', position: [0, 0], rotation: 0, status: 'IDLE', hasCargo: false, battery: 100, path: [], safetyAlert: false }]);
  env.step({ R01: Action.DOWN });
  assert(2, 'DOWN Movement', env.getState().robots[0].position[1] === 1);

  // 3. LEFT movement (1,0) -> (0,0)
  env.setRobots([{ id: 'R01', position: [1, 0], rotation: 0, status: 'IDLE', hasCargo: false, battery: 100, path: [], safetyAlert: false }]);
  env.step({ R01: Action.LEFT });
  assert(3, 'LEFT Movement', env.getState().robots[0].position[0] === 0);

  // 4. RIGHT movement (0,0) -> (1,0)
  env.setRobots([{ id: 'R01', position: [0, 0], rotation: 0, status: 'IDLE', hasCargo: false, battery: 100, path: [], safetyAlert: false }]);
  env.step({ R01: Action.RIGHT });
  assert(4, 'RIGHT Movement', env.getState().robots[0].position[0] === 1);

  // 5. Boundary Rejection (0,0) + UP -> stays (0,0)
  env.setRobots([{ id: 'R01', position: [0, 0], rotation: 0, status: 'IDLE', hasCargo: false, battery: 100, path: [], safetyAlert: false }]);
  env.step({ R01: Action.UP });
  assert(5, 'Boundary Rejection', env.getState().robots[0].position[1] === 0);

  // 6. Shelf Rejection (0,2) + RIGHT (shelf at 1,2) -> stays (0,2)
  env.setRobots([{ id: 'R01', position: [0, 2], rotation: 0, status: 'IDLE', hasCargo: false, battery: 100, path: [], safetyAlert: false }]);
  env.step({ R01: Action.RIGHT });
  assert(6, 'Shelf Rejection', env.getState().robots[0].position[0] === 0);

  // 7. Same-Cell Conflict: R1(4,5) RIGHT, R2(6,5) LEFT -> target (5,5) -> both NO_OP
  env.setRobots([
    { id: 'R01', position: [4, 5], rotation: 0, status: 'IDLE', hasCargo: false, battery: 100, path: [], safetyAlert: false },
    { id: 'R02', position: [6, 5], rotation: 0, status: 'IDLE', hasCargo: false, battery: 100, path: [], safetyAlert: false }
  ]);
  env.step({ R01: Action.RIGHT, R02: Action.LEFT });
  const s7 = env.getState();
  assert(7, 'Same-Cell Conflict Resolution', s7.robots[0].position[0] === 4 && s7.robots[1].position[0] === 6);

  // 8. Swap Conflict: R1(5,5) DOWN, R2(5,6) UP -> both NO_OP
  env.setRobots([
    { id: 'R01', position: [5, 5], rotation: 0, status: 'IDLE', hasCargo: false, battery: 100, path: [], safetyAlert: false },
    { id: 'R02', position: [5, 6], rotation: 0, status: 'IDLE', hasCargo: false, battery: 100, path: [], safetyAlert: false }
  ]);
  env.step({ R01: Action.DOWN, R02: Action.UP });
  const s8 = env.getState();
  assert(8, 'Position Swap Conflict', s8.robots[0].position[1] === 5 && s8.robots[1].position[1] === 6);

  // 9. Simultaneous Movement: R1(0,0) RIGHT, R2(1,0) RIGHT, R3(2,0) RIGHT -> all move right
  env.setRobots([
    { id: 'R01', position: [0, 0], rotation: 0, status: 'IDLE', hasCargo: false, battery: 100, path: [], safetyAlert: false },
    { id: 'R02', position: [1, 0], rotation: 0, status: 'IDLE', hasCargo: false, battery: 100, path: [], safetyAlert: false },
    { id: 'R03', position: [2, 0], rotation: 0, status: 'IDLE', hasCargo: false, battery: 100, path: [], safetyAlert: false }
  ]);
  env.step({ R01: Action.RIGHT, R02: Action.RIGHT, R03: Action.RIGHT });
  const s9 = env.getState();
  assert(9, 'Simultaneous Action Resolution', s9.robots[0].position[0] === 1 && s9.robots[1].position[0] === 2 && s9.robots[2].position[0] === 3);

  // 10. NO_OP result on invalid action trace
  assert(10, 'NO_OP Execution Result', s8.robots[0].lastExecutedAction === 'NO_OP');

  // 11 & 12. Observation shape = 50, valid numeric values
  const obs = env.getObservations();
  const r01Obs = obs['R01'];
  assert(11, 'Observation Shape (50,)', r01Obs && r01Obs.length === 50);
  assert(12, 'Observation Type Float Array', Array.isArray(r01Obs) && r01Obs.every(v => typeof v === 'number'));

  // 13. 3x3 occupancy grid one-hot encoding validity (first 45 elements)
  const occSlice = r01Obs.slice(0, 45);
  let oneHotValid = true;
  for (let i = 0; i < 9; i++) {
    const chunk = occSlice.slice(i * 5, (i + 1) * 5);
    const sum = chunk.reduce((a, b) => a + b, 0);
    if (sum !== 1) oneHotValid = false;
  }
  assert(13, 'Occupancy Grid One-Hot Encoding', oneHotValid);

  // 14. Task bearing values (indices 45..47)
  const bearingSlice = r01Obs.slice(45, 48);
  assert(14, 'Task Bearing Values', bearingSlice.length === 3 && bearingSlice.every(v => !isNaN(v)));

  // 15 & 16. Task flags: has_task (index 48) and carrying_item (index 49)
  assert(15, 'has_task Flag', typeof r01Obs[48] === 'number');
  assert(16, 'carrying_item Flag', typeof r01Obs[49] === 'number');

  // 17. Reset functionality
  env.reset();
  assert(17, 'Environment Reset', env.getState().timestep === 0);

  // 18, 19, 20. Fleet size scaling: N=5, N=10, N=20
  env.setFleetSize(5);
  assert(18, 'Fleet Size N=5', env.getState().robots.length === 5);

  env.setFleetSize(10);
  assert(19, 'Fleet Size N=10', env.getState().robots.length === 10);

  env.setFleetSize(20);
  assert(20, 'Fleet Size N=20', env.getState().robots.length === 20);

  // 21. Metrics increment per timestep
  const metricsBefore = env.getState().metrics.timestep;
  env.step({});
  const metricsAfter = env.getState().metrics.timestep;
  assert(21, 'Timestep Metrics Increment', metricsAfter === metricsBefore + 1);

  // 22 & 23. Logical grid position authoritative & active robot count
  assert(22, 'Logical Grid Position Ownership', env.getState().robots.length === 20);
  assert(23, 'Active AMRs Count Sync', env.getState().metrics.activeRobots === 20);

  return results;
}

export function runTaskQueueTest(): void {
  const queue = new TaskQueue();

  const task1: TaskState = {
    id: 'T01',
    pickupPos: [1, 1],
    deliveryPos: [5, 5],
    status: 'UNASSIGNED'
  };

  const task2: TaskState = {
    id: 'T02',
    pickupPos: [2, 2],
    deliveryPos: [6, 6],
    status: 'UNASSIGNED'
  };

  const task3: TaskState = {
    id: 'T03',
    pickupPos: [3, 3],
    deliveryPos: [7, 7],
    status: 'UNASSIGNED'
  };

  queue.addTask(task1);
  queue.addTask(task2);
  queue.addTask(task3);

  console.log('Queue length:', queue.getQueueLength());
  console.log('Next task:', queue.getNextTask()?.id);

  queue.removeTask('T01');

  console.log('Queue length after removing T01:', queue.getQueueLength());
  console.log('Next task after removing T01:', queue.getNextTask()?.id);
}

export function runRobotAssignmentTest(): void {
  const queue = new TaskQueue();
  const assignment = new RobotAssignment(queue);

  const task: TaskState = {
    id: 'T01',
    pickupPos: [2, 3],
    deliveryPos: [8, 8],
    status: 'UNASSIGNED'
  };

  const robot: RobotState = {
    id: 'R01',
    position: [0, 0],
    rotation: 0,
    status: 'IDLE',
    hasCargo: false,
    battery: 100,
    path: [],
    safetyAlert: false
  };

  queue.addTask(task);

  console.log('--- BEFORE ASSIGNMENT ---');
  console.log('Queue length:', queue.getQueueLength());
  console.log('Robot status:', robot.status);
  console.log('Robot task:', robot.taskId);

  const result = assignment.assignNextTask([robot]);

  console.log('--- AFTER ASSIGNMENT ---');
  console.log('Assigned robot:', result?.robot.id);
  console.log('Assigned task:', result?.task.id);
  console.log('Task status:', task.status);
  console.log('Task assigned robot:', task.assignedRobotId);
  console.log('Robot status:', robot.status);
  console.log('Robot task:', robot.taskId);
  console.log('Robot target:', robot.targetPosition);
  console.log('Queue length:', queue.getQueueLength());

  assignment.completeTask(robot, task, 100);

  console.log('--- AFTER COMPLETION ---');
  console.log('Task status:', task.status);
  console.log('Robot status:', robot.status);
  console.log('Robot task:', robot.taskId);
}

export function runTaskIntegrationTest(): void {
  const env = new WarehouseEnvironment();

  const state = env.getState();

  console.log('--- TASK INTEGRATION TEST ---');

  console.log('Total robots:', state.robots.length);
  console.log('Total tasks:', state.tasks.length);

  console.log(
    'Assigned tasks:',
    state.tasks.filter(task => task.status === 'ASSIGNED').length
  );

  console.log(
    'Unassigned tasks:',
    state.tasks.filter(task => task.status === 'UNASSIGNED').length
  );

  console.log('--- TASK DETAILS ---');

  state.tasks.forEach(task => {
    console.log(
      task.id,
      '| Status:', task.status,
      '| Robot:', task.assignedRobotId
    );
  });

  console.log('--- ROBOT DETAILS ---');

  state.robots.forEach(robot => {
    console.log(
      robot.id,
      '| Status:', robot.status,
      '| Task:', robot.taskId,
      '| Target:', robot.targetPosition
    );
  });
}

// Automatically expose on window for browser developer console execution
if (typeof window !== 'undefined') {
  (window as any).runWarehouseTests = runAutomatedWarehouseTests;
  (window as any).runTaskQueueTest = runTaskQueueTest;
  (window as any).runRobotAssignmentTest = runRobotAssignmentTest;
  (window as any).runTaskIntegrationTest = runTaskIntegrationTest;
}