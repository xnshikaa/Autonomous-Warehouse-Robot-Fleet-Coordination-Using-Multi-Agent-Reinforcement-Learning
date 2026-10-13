import type { TaskState } from '../types/warehouse';

export class TaskQueue {
  private queue: TaskState[] = [];

  /**
   * Add a new task to the waiting queue.
   * Tasks enter the queue as UNASSIGNED.
   */
  public addTask(task: TaskState): void {
    task.status = 'UNASSIGNED';
    task.assignedRobotId = undefined;

    this.queue.push(task);
  }

  /**
   * Return the next waiting task.
   * FIFO: First In, First Out.
   */
  public getNextTask(): TaskState | undefined {
    return this.queue.find(task => task.status === 'UNASSIGNED');
  }

  /**
   * Remove a task from the waiting queue.
   */
  public removeTask(taskId: string): TaskState | undefined {
    const index = this.queue.findIndex(task => task.id === taskId);

    if (index === -1) {
      return undefined;
    }

    const removedTasks = this.queue.splice(index, 1);
    return removedTasks[0];
  }

  /**
   * Find a task using its ID.
   */
  public getTaskById(taskId: string): TaskState | undefined {
    return this.queue.find(task => task.id === taskId);
  }

  /**
   * Return all tasks currently waiting in the queue.
   */
  public getWaitingTasks(): TaskState[] {
    return this.queue.filter(task => task.status === 'UNASSIGNED');
  }

  /**
   * Check whether there are waiting tasks.
   */
  public hasWaitingTasks(): boolean {
    return this.queue.some(task => task.status === 'UNASSIGNED');
  }

  /**
   * Return the number of waiting tasks.
   */
  public getQueueLength(): number {
    return this.getWaitingTasks().length;
  }

  /**
   * Clear all tasks from the queue.
   */
  public clearQueue(): void {
    this.queue = [];
  }
}