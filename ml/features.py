def extract_features(task_manager, presence_tracker):
    status = task_manager.get_status()
    summary = presence_tracker.get_summary()

    elapsed = status["elapsed"]
    progress = status["progress"]
    remaining = status["remaining"]

    present_time = summary["present"]
    absent_time = summary["absent"]

    # count events where state == absent

    absence_count = sum(
        1 for event in presence_tracker.events
        if event["state"] == "ABSENT"
    )

    # calculate the presence ratio

    if present_time + absent_time > 0:

        presence_ratio = present_time / (present_time + absent_time) 
    else:
        presence_ratio = 0.0

    return {
        "elapsed": elapsed,
        "progress": progress,
        "remaining": remaining,
        "present_time": present_time,
        "absent_time": absent_time,
        "absence count": absence_count,
        "presence ratio": presence_ratio
        
    }
