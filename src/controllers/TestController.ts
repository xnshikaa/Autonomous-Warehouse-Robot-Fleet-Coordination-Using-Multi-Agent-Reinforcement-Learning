import { IController } from './BaseController';
import { WarehouseState, ActionDict, Action, ControllerType } from '../types/warehouse';
import { WarehouseEnvironment } from '../environment/WarehouseEnvironment';

export interface TestSuiteDefinition {
  id: string;
  name: string;
  description: string;
  setupFleet: (env: WarehouseEnvironment) => void;
  getActionsForStep: (stepIdx: number) => ActionDict;
  expectedSteps: number;
}

export class TestController implements IController {
  public readonly type: ControllerType = 'DEMO_CONTROLLER';
  private currentTestId: string = 'TEST_1_SINGLE_ROBOT';
  private currentStepIdx: number = 0;

  public readonly testSuites: Record<string, TestSuiteDefinition> = {
    TEST_1_SINGLE_ROBOT: {
      id: 'TEST_1_SINGLE_ROBOT',
      name: 'Test 1: Single Robot Movement Sequence',
      description: 'Verifies sequential discrete grid movement (DOWN, RIGHT, RIGHT) for a single AMR.',
      expectedSteps: 3,
      setupFleet: (env) => {
        env.setRobots([
          {
            id: 'R01',
            position: [0, 0],
            rotation: 0,
            status: 'IDLE',
            hasCargo: false,
            battery: 100,
            path: [],
            safetyAlert: false
          }
        ]);
      },
      getActionsForStep: (step) => {
        const script: ActionDict[] = [
          { R01: Action.DOWN },  // (0,1)
          { R01: Action.RIGHT }, // (1,1)
          { R01: Action.RIGHT }  // (2,1)
        ];
        return script[Math.min(step, script.length - 1)];
      }
    },

    TEST_2_SAME_CELL: {
      id: 'TEST_2_SAME_CELL',
      name: 'Test 2: Same-Cell Target Conflict',
      description: 'Two robots request entry into the same grid cell (5,5). Verifies BOTH executed actions become NO_OP.',
      expectedSteps: 1,
      setupFleet: (env) => {
        env.setRobots([
          { id: 'R01', position: [4, 5], rotation: 0, status: 'IDLE', hasCargo: false, battery: 100, path: [], safetyAlert: false },
          { id: 'R02', position: [6, 5], rotation: 0, status: 'IDLE', hasCargo: false, battery: 100, path: [], safetyAlert: false }
        ]);
      },
      getActionsForStep: (step) => {
        return {
          R01: Action.RIGHT, // Targets (5,5)
          R02: Action.LEFT   // Targets (5,5)
        };
      }
    },

    TEST_3_SWAP_CONFLICT: {
      id: 'TEST_3_SWAP_CONFLICT',
      name: 'Test 3: Position Swap Conflict',
      description: 'R1 at (5,5) targets (5,6) while R2 at (5,6) targets (5,5). Verifies ROBOT_ROBOT_SWAP detection and dual NO_OP override.',
      expectedSteps: 1,
      setupFleet: (env) => {
        env.setRobots([
          { id: 'R01', position: [5, 5], rotation: 0, status: 'IDLE', hasCargo: false, battery: 100, path: [], safetyAlert: false },
          { id: 'R02', position: [5, 6], rotation: 0, status: 'IDLE', hasCargo: false, battery: 100, path: [], safetyAlert: false }
        ]);
      },
      getActionsForStep: (step) => {
        return {
          R01: Action.DOWN, // Targets (5,6)
          R02: Action.UP    // Targets (5,5)
        };
      }
    },

    TEST_4_SHELF_COLLISION: {
      id: 'TEST_4_SHELF_COLLISION',
      name: 'Test 4: Shelf Obstacle Entry Attempt',
      description: 'Robot at (0,2) attempts to enter shelf cell at (1,2). Verifies OBSTACLE_BLOCK override to NO_OP.',
      expectedSteps: 1,
      setupFleet: (env) => {
        env.setRobots([
          { id: 'R01', position: [0, 2], rotation: 0, status: 'IDLE', hasCargo: false, battery: 100, path: [], safetyAlert: false }
        ]);
      },
      getActionsForStep: (step) => {
        return { R01: Action.RIGHT }; // Cell (1,2) is a shelf
      }
    },

    TEST_5_BOUNDARY_COLLISION: {
      id: 'TEST_5_BOUNDARY_COLLISION',
      name: 'Test 5: Warehouse Boundary Exit Attempt',
      description: 'Robot at boundary (0,0) attempts UP and LEFT actions. Verifies BOUNDARY_BLOCK override to NO_OP.',
      expectedSteps: 2,
      setupFleet: (env) => {
        env.setRobots([
          { id: 'R01', position: [0, 0], rotation: 0, status: 'IDLE', hasCargo: false, battery: 100, path: [], safetyAlert: false }
        ]);
      },
      getActionsForStep: (step) => {
        return step === 0 ? { R01: Action.UP } : { R01: Action.LEFT };
      }
    },

    TEST_8_SIMULTANEOUS_EVALUATION: {
      id: 'TEST_8_SIMULTANEOUS_EVALUATION',
      name: 'Test 8: Simultaneous Action Pipeline Evaluation',
      description: 'Chain of 3 robots (R1 at 0,0; R2 at 1,0; R3 at 2,0) move RIGHT simultaneously. Verifies proposed positions are calculated from the SAME previous state.',
      expectedSteps: 1,
      setupFleet: (env) => {
        env.setRobots([
          { id: 'R01', position: [0, 0], rotation: 0, status: 'IDLE', hasCargo: false, battery: 100, path: [], safetyAlert: false },
          { id: 'R02', position: [1, 0], rotation: 0, status: 'IDLE', hasCargo: false, battery: 100, path: [], safetyAlert: false },
          { id: 'R03', position: [2, 0], rotation: 0, status: 'IDLE', hasCargo: false, battery: 100, path: [], safetyAlert: false }
        ]);
      },
      getActionsForStep: (step) => {
        return {
          R01: Action.RIGHT, // Targets (1,0)
          R02: Action.RIGHT, // Targets (2,0)
          R03: Action.RIGHT  // Targets (3,0)
        };
      }
    }
  };

  public selectTest(testId: string, env?: WarehouseEnvironment): void {
    if (this.testSuites[testId]) {
      this.currentTestId = testId;
      this.currentStepIdx = 0;
      if (env) {
        this.testSuites[testId].setupFleet(env);
      }
    }
  }

  public getCurrentTest(): TestSuiteDefinition {
    return this.testSuites[this.currentTestId] || this.testSuites.TEST_1_SINGLE_ROBOT;
  }

  public getStepIdx(): number {
    return this.currentStepIdx;
  }

  public getActions(state: WarehouseState): ActionDict {
    const suite = this.getCurrentTest();
    const actions = suite.getActionsForStep(this.currentStepIdx);
    this.currentStepIdx++;
    return actions;
  }
}
