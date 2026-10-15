class EpisodeManager:
    """
    Manages episode termination logic for the warehouse RL environment.

    An episode terminates when:
    1. All tasks have been completed, or
    2. The maximum number of steps has been reached.
    """

    def __init__(self, max_steps):
        if max_steps <= 0:
            raise ValueError("max_steps must be greater than zero")

        self.max_steps = max_steps
        self.current_step = 0

    def reset(self):
        """Reset the episode step counter."""
        self.current_step = 0

    def step(self):
        """Advance the episode by one step."""
        self.current_step += 1

    def is_terminated(self, tasks_remaining):
        """
        Determine whether the episode should terminate.

        Args:
            tasks_remaining: Number of incomplete tasks.

        Returns:
            True if the episode should terminate, otherwise False.
        """

        if tasks_remaining < 0:
            raise ValueError("tasks_remaining cannot be negative")

        all_tasks_completed = tasks_remaining == 0
        max_steps_reached = self.current_step >= self.max_steps

        return all_tasks_completed or max_steps_reached

    def termination_reason(self, tasks_remaining):
        """
        Return the reason why the episode terminated.

        Returns:
            "all_tasks_completed",
            "max_steps_reached",
            or None if the episode is still running.
        """

        if tasks_remaining < 0:
            raise ValueError("tasks_remaining cannot be negative")

        if tasks_remaining == 0:
            return "all_tasks_completed"

        if self.current_step >= self.max_steps:
            return "max_steps_reached"

        return None