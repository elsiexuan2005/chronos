import cv2 as cv
import time

from face_tracker import FaceTracker
from presence_tracker import PresenceTracker
from task_manager import TaskManager
from deadline_predictor import predict_deadline
from ml.features import extract_features
from ml.collect_data import DataCollector

import csv
import uuid
import statistics
from pathlib import Path

# Initialize webcam
cap = cv.VideoCapture(0, cv.CAP_AVFOUNDATION)

if not cap.isOpened():
    raise RuntimeError("Cannot open camera")


tracker = None
presence_tracker = PresenceTracker()

task_manager = TaskManager(
    total_tasks=5,
    deadline_seconds=180
)


collector = DataCollector()

next_snapshot = 5.0
session_finished = False

# Activity classification experiment
TRIAL_DURATION = 20.0
DATA_PATH = Path("data/activity_training.csv")

trial_activity = None
trial_start = None
trial_samples = []

try:
    tracker = FaceTracker()

    start_time = time.perf_counter()
    prev_frame_time = start_time
    last_timestamp_ms = -1

    while True:
        ret, frame = cap.read()

        if not ret:
            print("Cannot receive frame")
            break

        frame = cv.flip(frame, 1)

        # Generate increasing timestamps for MediaPipe
        current_time = time.perf_counter()

        timestamp_ms = max(
            int((current_time - start_time) * 1000),
            last_timestamp_ms + 1
        )
        last_timestamp_ms = timestamp_ms

        # Face detection
        frame, face_present = tracker.detect(
            frame, timestamp_ms
        )
        if trial_activity is not None:
            pitch_difference = None

            if (
                tracker.head_pitch is not None
                and tracker.neutral_pitch is not None
            ):
                pitch_difference = (
                    tracker.head_pitch - tracker.neutral_pitch
                )

            trial_samples.append({
                "visible": int(face_present),
                "pitch_difference": pitch_difference,
            })
            elapsed_trial = time.perf_counter() - trial_start

            cv.putText(
                frame,
                f"Recording: {trial_activity} ({elapsed_trial:.1f}s)",
                (20, 490),
                cv.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 255),
                2
            )

            if elapsed_trial >= TRIAL_DURATION:
                pitch_values = [
                    abs(s["pitch_difference"])
                    for s in trial_samples
                    if s["pitch_difference"] is not None
                ]

                row = {
                    "trial_id": str(uuid.uuid4()),
                    "activity": trial_activity,
                    "duration": elapsed_trial,
                    "face_visibility_ratio": (
                        sum(s["visible"] for s in trial_samples)
                        / len(trial_samples)
                    ),
                    "mean_pitch_deviation": (
                        statistics.mean(pitch_values)
                        if pitch_values else 0
                    ),
                    "pitch_variability": (
                        statistics.pstdev(pitch_values)
                        if len(pitch_values) > 1 else 0
                    ),
                    "head_tilt_ratio": (
                        sum(v > 15 for v in pitch_values)
                        / len(pitch_values)
                        if pitch_values else 0
                    ),
                }

                DATA_PATH.parent.mkdir(parents=True, exist_ok=True)

                with open(DATA_PATH, "a", newline="") as f:
                    writer = csv.DictWriter(f, fieldnames=row.keys())

                    if f.tell() == 0:
                        writer.writeheader()

                    writer.writerow(row)

                print("Saved trial:", row)

                trial_activity = None
                trial_start = None
                trial_samples = []




        # Head orientation information
        head_direction = tracker.head_direction
        head_pitch = tracker.head_pitch
        neutral_pitch = tracker.neutral_pitch

        if head_pitch is not None and neutral_pitch is not None:
            pitch_difference = head_pitch - neutral_pitch
            pitch_text = f"Pitch difference: {pitch_difference:+.1f} deg"
        else:
            pitch_text = "Pitch: Press C to calibrate"

        # Update presence tracker
        
        presence_tracker.update(face_present)

        task_status = task_manager.get_status()
        prediction = predict_deadline(task_status)

        if prediction["estimated_remaining"] is None:
            forecast = "WAITING FOR PROGRESS"
        elif prediction["predicted_on_time"]:
            forecast = "ON TRACK"
        else:
            forecast = "AT RISK"

        # FPS
        new_frame_time = time.perf_counter()
        delta = new_frame_time - prev_frame_time
        fps = 1 / delta if delta > 0 else 0
        prev_frame_time = new_frame_time

        # Presence summary
        summary = presence_tracker.get_summary()

        present_time = summary["present"]
        absent_time = summary["absent"]
        total = summary["total"]

        if (
            task_status["elapsed"] >= next_snapshot
            and task_status["remaining"] > 0
            and task_status["completed"] < task_status["total_tasks"]
        ):
            features = extract_features(
                task_manager,
                presence_tracker
            )
    

            collector.record(features)
            next_snapshot += 5.0

            print("Snapshot recorded:", features)


        # Display FPS
        cv.putText(
            frame,
            f"FPS: {fps:.1f}",
            (20, 40),
            cv.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2
        )

        # Face status
        face_status = "FACE DETECTED" if face_present else "NO FACE"
        color = (0, 255, 0) if face_present else (0, 0, 255)

        cv.putText(
            frame,
            face_status,
            (20, 80),
            cv.FONT_HERSHEY_SIMPLEX,
            1,
            color,
            2
        )

        # Presence information
        cv.putText(
            frame,
            f"TOTAL: {total:.1f}s",
            (20, 120),
            cv.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2
        )

        cv.putText(
            frame,
            f"ABSENCE: {absent_time:.1f}s",
            (20, 160),
            cv.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2
        )




        cv.putText(
            frame,
            f"PRESENT: {present_time:.1f}s",
            (20, 200),
            cv.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2
        )

        # Task information
        cv.putText(
            frame,
            f"TASKS: {task_status['completed']}/{task_status['total_tasks']}",
            (20, 250),
            cv.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2
        )

        cv.putText(
            frame,
            f"TIME LEFT: {task_status['remaining']:.1f}s",
            (20, 290),
            cv.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2
        )

        cv.putText(
            frame,
            f"PREDICTION: {forecast}",
            (20, 330),
            cv.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2
        )

        cv.putText(
            frame,
            "N = Complete task | Q = Quit",
            (20, 380),
            cv.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2
        )

        cv.putText(
            frame,
            f"Head: {head_direction}",
            (20, 420),
            cv.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 255),
            2
        )

        cv.putText(
            frame,
            pitch_text,
            (20, 455),
            cv.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 255),
            2
        )

        cv.imshow("CHRONOS - Live Challenge", frame)

        # Keyboard controls
        key = cv.waitKey(1) & 0xFF

        if key == ord("q"):
            print("Challenge cancelled.")
            break

        elif key == ord("c"):
            tracker.calibrate()
            print("Neutral pitch:", tracker.neutral_pitch)
        

        elif key in (ord("1"), ord("2"), ord("3")):
            if trial_activity is None:
                if tracker.neutral_pitch is None:
                    print("Press C to calibrate first.")
                    continue

                activities = {
                    ord("1"): "mental",
                    ord("2"): "paper",
                    ord("3"): "calculator",
                }

                trial_activity = activities[key]
                trial_start = time.perf_counter()
                trial_samples = []

                print(f"Started {trial_activity} trial")
                
        elif key == ord("n"):
            status = task_manager.get_status()

            if (
                status["remaining"] > 0
                and status["completed"] < status["total_tasks"]
            ):
                task_manager.finish_task()
                print("Task completed!")

        # Check session outcome
        task_status = task_manager.get_status()

        if (
            task_status["completed"] >= task_status["total_tasks"]
            or task_status["remaining"] <= 0
        ):
            session_finished = True
            break

finally:
    if tracker is not None:
        tracker.close()

    cap.release()
    cv.destroyAllWindows()


# Save completed session
if session_finished:
    final_status = task_manager.get_status()

    all_completed = (
        final_status["completed"] == final_status["total_tasks"]
    )

    completion_time = (
        task_manager.progress_history[-1]["completed_at"]
        if all_completed else float("inf")
    )

    finished_on_time = (
        all_completed
        and completion_time <= final_status["deadline"]
    )

    saved_count = len(collector.snapshots)
    collector.save(finished_on_time)

    print("\n=== CHRONOS SESSION RESULTS ===")
    print("Tasks completed:", final_status["completed"])
    print("Finished on time:", finished_on_time)
    print("Snapshots saved:", saved_count)