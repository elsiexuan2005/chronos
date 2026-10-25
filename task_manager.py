import time

class TaskManager:
    def __init__(self, total_tasks, deadline_seconds):
        self.total_tasks = total_tasks
        self.deadline_seconds = deadline_seconds
        self.completed_tasks = 0
        self.start_time = time.perf_counter()
        self.progress_history = []

    def finish_task(self):
        if self.completed_tasks >= self.total_tasks: 
            return
        self.completed_tasks += 1
        current_time = time.perf_counter()
        elapsed = current_time - self.start_time
        completed_task = {
            "task": self.completed_tasks,
            "completed_at": elapsed 
        }
        self.progress_history.append(completed_task)


    def get_status(self):
        current_time = time.perf_counter()
        elapsed = current_time - self.start_time
        remaining_time = max(0, self.deadline_seconds - elapsed)
        progress = self.completed_tasks / self.total_tasks
        status = {
            "elapsed": elapsed,
            "progress": progress,
            "remaining": remaining_time,
            "completed": self.completed_tasks,
            "total_tasks": self.total_tasks,
            "deadline": self.deadline_seconds
        }
        return status

