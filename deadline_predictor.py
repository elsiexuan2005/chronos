def predict_deadline(status):
    completed = status["completed"]
    total = status["total_tasks"]
    elapsed = status["elapsed"]
    remaining = status["remaining"]
    deadline = status["deadline"]

    # All questions completed
    if completed >= total:
        return {
            "estimated_remaining": 0.0,
            "predicted_on_time": elapsed <= deadline,
            "eta_seconds": 0.0,
            "safety_margin": remaining,
            "future_progress": total,
        }

    # We cannot estimate pace without completed questions
    if completed == 0:
        return {
            "estimated_remaining": None,
            "predicted_on_time": None,
            "eta_seconds": None,
            "safety_margin": None,
            "future_progress": None,
        }

    # Average completion time per question
    avg_task_time = elapsed / completed

    questions_left = total - completed

    # Estimated time needed for remaining questions
    eta = avg_task_time * questions_left

    # How much time is left after estimated completion?
    safety_margin = remaining - eta

    predicted_on_time = eta <= remaining

    # Estimated questions completed in the next 15 seconds
    horizon = min(15.0, remaining)
    expected_new_questions = horizon / avg_task_time

    future_progress = min(
        total,
        completed + expected_new_questions
    )

    return {
        "estimated_remaining": eta,
        "predicted_on_time": predicted_on_time,
        "eta_seconds": eta,
        "safety_margin": safety_margin,
        "future_progress": future_progress,
    }