import { 
  WarehouseState, 
  StateProvider, 
  AlgorithmType, 
  ControllerType,
  ActionDict 
} from '../types/warehouse';
import { WarehouseEnvironment } from '../environment/WarehouseEnvironment';
import { IController } from '../controllers/BaseController';
import { DemoController } from '../controllers/DemoController';
import { TestController } from '../controllers/TestController';
import { FutureRLController } from '../controllers/FutureRLController';

export class DemoStateProvider implements StateProvider {
  private environment: WarehouseEnvironment;
  private subscribers: Set<(state: WarehouseState) => void> = new Set();
  
  private isPaused: boolean = false;
  private speedMultiplier: number = 1.0;
  private timerId: any = null;
  private fleetSize: number = 10;

  public demoController: DemoController;
  public testController: TestController;
  public futureRLController: FutureRLController;
  private activeController: IController;

  constructor(initialFleetSize: number = 10, initialAlgorithm: AlgorithmType = 'QMIX') {
    this.fleetSize = initialFleetSize;
    this.environment = new WarehouseEnvironment(
      { gridWidth: 20, gridHeight: 20, cellSize: 2.0 },
      initialFleetSize
    );
    this.environment.setAlgorithm(initialAlgorithm);

    this.demoController = new DemoController();
    this.testController = new TestController();
    this.futureRLController = new FutureRLController();
    this.activeController = this.demoController;

    this.startLoop();
  }

  public getState(): WarehouseState {
    return this.environment.getState();
  }

  public getEnvironment(): WarehouseEnvironment {
    return this.environment;
  }

  public subscribe(callback: (state: WarehouseState) => void): () => void {
    this.subscribers.add(callback);
    callback(this.getState());
    return () => {
      this.subscribers.delete(callback);
    };
  }

  public setSpeed(multiplier: number): void {
    this.speedMultiplier = multiplier;
    this.restartLoop();
  }

  public pause(): void {
    this.isPaused = true;
  }

  public resume(): void {
    this.isPaused = false;
  }

  public reset(): void {
    this.environment.reset();
    if (this.activeController.type === 'TEST_CONTROLLER') {
      const currentTestId = this.testController.getCurrentTest().id;
      this.testController.selectTest(currentTestId, this.environment);
    }
    this.notifySubscribers();
  }

  public setFleetSize(size: number): void {
    this.fleetSize = size;
    this.environment.setFleetSize(size);
    this.notifySubscribers();
  }

  public setAlgorithm(algo: AlgorithmType): void {
    this.environment.setAlgorithm(algo);
    this.notifySubscribers();
  }

  public setControllerType(type: ControllerType): void {
    if (type === 'TEST_CONTROLLER') {
      this.activeController = this.testController;
      this.testController.selectTest('TEST_1_SINGLE_ROBOT', this.environment);
    } else if (type === 'RL_BACKEND') {
      this.activeController = this.futureRLController;
    } else {
      this.activeController = this.demoController;
    }
    this.environment.setControllerType(type);
    this.notifySubscribers();
  }

  public getActiveController(): IController {
    return this.activeController;
  }

  public selectTestSuite(testId: string): void {
    this.activeController = this.testController;
    this.environment.setControllerType('TEST_CONTROLLER');
    this.testController.selectTest(testId, this.environment);
    this.notifySubscribers();
  }

  public stepOnce(): WarehouseState {
    const actions: ActionDict = this.activeController.getActions(this.getState());
    const nextState = this.environment.step(actions);
    this.notifySubscribers();
    return nextState;
  }

  public selectRobot(robotId: string | null): void {
    // Selection state maintained locally for inspector UI
  }

  public selectTask(taskId: string | null): void {
    // Selection state maintained locally for inspector UI
  }

  private startLoop(): void {
    const intervalMs = Math.max(200, 1000 / (1.5 * this.speedMultiplier));
    this.timerId = setInterval(() => this.stepSimulation(), intervalMs);
  }

  private restartLoop(): void {
    if (this.timerId) clearInterval(this.timerId);
    this.startLoop();
  }

  private stepSimulation(): void {
    if (this.isPaused) return;
    this.stepOnce();
  }

  private notifySubscribers(): void {
    const currentState = this.getState();
    this.subscribers.forEach(cb => cb(currentState));
  }
}
