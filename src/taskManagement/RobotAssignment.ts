import type { RobotState, TaskState } from '../types/warehouse';
import { TaskQueue } from './TaskQueue';

export class RobotAssignment {
  constructor(private taskQueue: TaskQueue) {}

  /**
   * Find the first robot that is currently available.
   */
  private findAvailableRobot(
    robots: RobotState[]
  ): RobotState | undefined {
    return robots.find(
      robot =>
        robot.status === 'IDLE' &&
        robot.taskId === undefined
    );
  }

  /**
   * Assign the next waiting task to an available robot.
   */
  public assignNextTask(
    robots: RobotState[]
  ): { robot: RobotState; task: TaskState } | undefined {
    const task = this.taskQueue.getNextTask();

    if (!task) {
      return undefined;
    }

    const robot = this.findAvailableRobot(robots);

    if (!robot) {
      return undefined;
    }

    task.status = 'ASSIGNED';
    task.assignedRobotId = robot.id;

    robot.taskId = task.id;
    robot.status = 'MOVING';
    robot.targetPosition = task.pickupPos;

    this.taskQueue.removeTask(task.id);

    return {
      robot,
      task
    };
  }

  /**
   * Mark the robot's current task as completed.
   */
  public completeTask(
    robot: RobotState,
    task: TaskState,
    completedTimestep?: number
  ): void {
    task.status = 'COMPLETED';
    task.completedTimestep = completedTimestep;

    robot.taskId = undefined;
    robot.status = 'IDLE';
    robot.hasCargo = false;
    robot.targetPosition = undefined;
    robot.path = [];
  }
}