"""Chronos: maths + webcam activity-label collection.

Place this file in the Chronos project as main.py. Uses existing
face_tracker.py and math_challenge.py. Does NOT infer cheating.
"""
import csv
import math
import time
import uuid
from pathlib import Path
from statistics import mean, pstdev
import tkinter as tk
from tkinter import ttk, messagebox

import cv2 as cv
from PIL import Image, ImageTk

from face_tracker import FaceTracker
from math_challenge import MathChallenge
DATA_PATH = Path(__file__).resolve().parent / "data" / "activity_training.csv"
FIELDNAMES = [
    "participant_id", "session_id", "question_index", "category", "difficulty",
    "activity", "duration_seconds", "wrong_attempts", "frames",
    "face_visibility_ratio", "pitch_coverage_ratio", "mean_pitch_deviation",
    "pitch_variability", "head_tilt_ratio", "calibrated",
]


class ChronosApp:
    def __init__(self, root):
        self.root = root
        root.title("CHRONOS — Maths & Camera Experiment")
        root.geometry("1250x850")
        root.minsize(1100, 770)

        self.face_tracker = None
        self.cap = None
        self.after_id = None
        self.running = False
        self.challenge = None
        self.session_id = None
        self.mode = None
        self.question_idx = 0
        self.question_start = None
        self.forecast_start = None  # Begins after head calibration
        self.completed_durations = []
        self.forecast_target_seconds = 90.0  # Comparison target, not a hard deadline
        self.samples = []
        self.recording = False
        self.timeline = []
        self.session_results = []
        self.models = {}
        self.load_models()
        self.wrong_attempts = 0
        self.start_time = None
        self.last_timestamp_ms = -1
        self.participant = tk.StringVar(value="P01")
        self.activity = tk.StringVar(value="mental")
        self.build_ui()
        root.bind("<c>", self.calibration_shortcut)
        root.protocol("WM_DELETE_WINDOW", self.close)

    def update_time_forecast(self):
        """Pace-based forecasts for the existing activity experiment.

        This experiment has no enforced deadline. The safety margin is relative
        to a 90-second target, not an enforced session end.
        """
        if not self.running or self.forecast_start is None:
            return

        elapsed = max(0.0, time.perf_counter() - self.forecast_start)
        completed = self.question_idx
        total = len(self.challenge.questions)
        remaining_target = self.forecast_target_seconds - elapsed

        if completed == 0:
            self.eta_label.config(text="Estimated time remaining: Waiting for first answer")
            self.margin_label.config(
                text=f"90s target: {remaining_target:+.1f}s left (not enforced)"
            )
            self.future_label.config(text="Expected progress in 15s: Unknown")
            return

        # Average observed session time per completed question, including
        # pauses and ongoing work between questions.
        avg_seconds = elapsed / completed
        eta = avg_seconds * (total - completed)
        margin = remaining_target - eta
        future = min(total, completed + 15.0 / avg_seconds)

        self.eta_label.config(text=f"Estimated time remaining: {eta:.1f}s")
        self.margin_label.config(text=f"90s target safety margin: {margin:+.1f}s")
        self.future_label.config(
            text=f"Expected progress in 15s: {future:.1f}/{total}"
        )

    def build_ui(self):
        ttk.Label(self.root, text="CHRONOS — ACTIVITY EXPERIMENT",
                  font=("Arial", 22, "bold")).pack(pady=12)
        panel = ttk.Frame(self.root)
        panel.pack(fill="both", expand=True, padx=20)

        left = ttk.Frame(panel)
        left.pack(side="left", fill="both", expand=True, padx=(0, 15))

        ttk.Label(left, text="Participant ID:").pack(anchor="w")
        self.participant_entry = ttk.Entry(left, textvariable=self.participant, width=22)
        self.participant_entry.pack(anchor="w", pady=(0, 8))
        ttk.Label(left, text="Choose the method the participant WILL use:").pack(anchor="w")
        self.mode_box = ttk.Combobox(left, textvariable=self.activity,
                                     values=("mental", "paper", "calculator"),
                                     state="readonly", width=20)
        self.mode_box.pack(anchor="w", pady=(3, 8))
        ttk.Label(left, text="For this whole 5-question session, use only the selected method.",
                  wraplength=480).pack(anchor="w", pady=(0, 8))

        self.start_button = ttk.Button(left, text="START NEW SESSION", command=self.start_session)
        self.start_button.pack(anchor="w", pady=5)
        self.question_label = ttk.Label(left, text="Press START NEW SESSION",
                                        font=("Arial", 17, "bold"), wraplength=480)
        self.question_label.pack(anchor="w", pady=(18, 10))
        self.answer_entry = ttk.Entry(left, font=("Arial", 16), state="disabled")
        self.answer_entry.pack(fill="x", pady=6)
        self.answer_entry.bind("<Return>", lambda event: self.submit_answer())
        self.submit_button = ttk.Button(left, text="SUBMIT ANSWER", command=self.submit_answer,
                                        state="disabled")
        self.submit_button.pack(anchor="w", pady=8)
        self.feedback_label = ttk.Label(left, text="")
        self.feedback_label.pack(anchor="w", pady=5)
        self.progress_label = ttk.Label(left, text="Progress: 0/5", font=("Arial", 13))
        self.progress_label.pack(anchor="w", pady=5)
        self.elapsed_label = ttk.Label(left, text="Elapsed: 0.0s")
        self.elapsed_label.pack(anchor="w", pady=4)
        # Deadline and future progress predictions
        forecast_box = ttk.LabelFrame(left, text="TIME FORECAST")

        forecast_box.pack(fill="x", pady=(10, 5))

        self.eta_label = ttk.Label(
            forecast_box,
            text="Estimated time remaining: Waiting",
            font=("Arial", 11)
        )
        self.eta_label.pack(anchor="w", padx=10, pady=(8, 3))

        self.margin_label = ttk.Label(
            forecast_box,
            text="Safety margin: Waiting",
            font=("Arial", 11)
        )
        self.margin_label.pack(anchor="w", padx=10, pady=3)

        self.future_label = ttk.Label(
            forecast_box,
            text="Expected progress in 15s: Waiting",
            font=("Arial", 11)
        )
        self.future_label.pack(anchor="w", padx=10, pady=(3, 8))

        result_box = ttk.LabelFrame(left, text="AI ACTIVITY ANALYSIS (after each answer)")
        result_box.pack(fill="x", pady=(15, 4))
        self.model_a_label = ttk.Label(result_box, text="Model A — Gameplay: waiting",
                                       font=("Arial", 12), wraplength=470)
        self.model_a_label.pack(anchor="w", padx=10, pady=(10, 4))
        self.model_b_label = ttk.Label(result_box, text="Model B — Gameplay + Vision: waiting",
                                       font=("Arial", 12), wraplength=470)
        self.model_b_label.pack(anchor="w", padx=10, pady=4)
        self.actual_label = ttk.Label(result_box, text="Selected activity: —")
        self.actual_label.pack(anchor="w", padx=10, pady=(4, 10))
        self.session_summary_label = ttk.Label(left, text="", wraplength=470,
                                                font=("Arial", 11))
        self.session_summary_label.pack(anchor="w", pady=8)

        right = ttk.Frame(panel)
        right.pack(side="right", fill="both", expand=True)
        self.camera_label = ttk.Label(right, text="Camera starts with session")
        self.camera_label.pack(pady=5)
        self.presence_label = ttk.Label(right, text="Face: Waiting", wraplength=500)
        self.presence_label.pack(pady=6)
        self.calibrate_button = ttk.Button(right, text="CALIBRATE HEAD (C)",
                                           command=self.calibrate_head, state="disabled")
        self.calibrate_button.pack(pady=6)
        ttk.Label(right, text="HEAD MOVEMENT OVER QUESTION TIME",
                  font=("Arial", 12, "bold")).pack(pady=(9, 3))
        self.timeline_canvas = tk.Canvas(right, width=470, height=170, bg="#121d2c",
                                         highlightthickness=0)
        self.timeline_canvas.pack(pady=3)
        self.timeline_info = ttk.Label(right, text="Horizontal: seconds | Vertical: absolute pitch deviation (°)")
        self.timeline_info.pack(pady=(0, 2))
        self.draw_timeline()
        ttk.Label(right, text="Calibrate facing forward before each session.\n"
                              "Looking down does NOT prove calculator use or cheating.",
                  wraplength=460, justify="center").pack(pady=6)

    def start_session(self):
        if self.running:
            return
        person = self.participant.get().strip()
        if not person:
            messagebox.showerror("Missing participant", "Enter a participant ID first.")
            return
        self.mode = self.activity.get()
        if self.mode not in ("mental", "paper", "calculator"):
            return
        try:
            self.challenge = MathChallenge(number_of_questions=5)
        except Exception as exc:
            messagebox.showerror("Maths question error", str(exc))
            return

        cap = cv.VideoCapture(0, cv.CAP_AVFOUNDATION)
        if not cap.isOpened():
            cap.release()
            messagebox.showerror("Camera error", "Cannot open webcam")
            return
        try:
            tracker = FaceTracker()
        except Exception as exc:
            cap.release()
            messagebox.showerror("FaceTracker error", str(exc))
            return
        self.cap = cap
        self.face_tracker = tracker
        self.session_id = str(uuid.uuid4())
        self.running = True
        self.recording = False
        self.session_results = []
        self.model_a_label.config(text="Model A — Gameplay: waiting")
        self.model_b_label.config(text="Model B — Gameplay + Vision: waiting")
        self.actual_label.config(text="Selected activity: —")
        self.session_summary_label.config(text="")
        self.question_idx = 0
        self.completed_durations = []
        self.forecast_start = None
        self.eta_label.config(text="Estimated time remaining: Waiting")
        self.margin_label.config(text="90s target safety margin: Waiting")
        self.future_label.config(text="Expected progress in 15s: Waiting")
        self.start_time = time.perf_counter()
        self.last_timestamp_ms = -1
        self.participant_entry.config(state="disabled")
        self.mode_box.config(state="disabled")
        self.start_button.config(state="disabled")
        self.calibrate_button.config(state="normal")

        self.answer_entry.config(state="disabled")
        self.submit_button.config(state="disabled")
        self.feedback_label.config(text="Face camera and click CALIBRATE, then solve normally.")
        self.show_question()
        self.update_frame()

    def show_question(self):
        question = self.challenge.questions[self.question_idx]
        self.question_label.config(
            text=f"Question {self.question_idx + 1}/5 — {question.get('topic', 'Maths')}\n"
                 f"{question['question']}"
        )
        self.progress_label.config(text=f"Progress: {self.question_idx}/5")
        self.samples = []
        self.timeline = []
        self.draw_timeline()
        self.wrong_attempts = 0
        self.question_start = time.perf_counter()
        self.answer_entry.delete(0, tk.END)
        self.answer_entry.focus_set()

    def calibration_shortcut(self, event):
        if (
            self.running
            and self.question_idx == 0
            and self.calibrate_button["state"] == "normal"
            and event.widget is not self.answer_entry
        ):
            self.calibrate_head()

    def calibrate_head(self):
        if not self.running or self.face_tracker is None:
            return

        if self.face_tracker.head_pitch is None:
            self.feedback_label.config(
                text="Face not detected. Look at the camera."
            )
            return

        # Calibrate neutral head orientation
        self.face_tracker.calibrate()
        print(
            "CALIBRATION CHECK:",
            "pitch =", self.face_tracker.head_pitch,
            "neutral =", self.face_tracker.neutral_pitch,
            flush=True
        )

        if self.face_tracker.neutral_pitch is None:
            return

        # Enable answering
        self.answer_entry.config(state="normal")
        self.submit_button.config(state="normal")

        # Remove measurements collected before calibration
        self.samples = []
        self.timeline = []
        self.recording = True
        self.draw_timeline()

        # Start question and forecast clocks after calibration
        self.question_start = time.perf_counter()
        self.forecast_start = self.question_start
        self.update_time_forecast()

        # Prevent recalibration during the trial
        self.calibrate_button.config(state="disabled")

        self.feedback_label.config(
            text="Calibrated! Start solving your maths questions."
        )

        self.answer_entry.focus_set()

    def update_frame(self):
        if not self.running:
            return
        ret, frame = self.cap.read()
        if not ret:
            self.feedback_label.config(text="Camera disconnected; session cancelled.")
            self.end_session(completed=False)
            return
        frame = cv.flip(frame, 1)
        now = time.perf_counter()
        timestamp_ms = max(int((now - self.start_time) * 1000), self.last_timestamp_ms + 1)
        self.last_timestamp_ms = timestamp_ms
        try:
            frame, face_present = self.face_tracker.detect(frame, timestamp_ms)
        except Exception as exc:
            self.feedback_label.config(text=f"Tracking error: {exc}")
            self.end_session(completed=False)
            return

        pitch = getattr(self.face_tracker, "head_pitch", None)
        neutral = getattr(self.face_tracker, "neutral_pitch", None)
        pitch_difference = float(pitch - neutral) if pitch is not None and neutral is not None else None
        if self.recording:
            self.samples.append((int(face_present), pitch_difference))
            if pitch_difference is not None and face_present:
                self.timeline.append((now - self.question_start, abs(pitch_difference)))
                # Redraw every ~5 samples to avoid slowing the camera loop.
                if len(self.timeline) % 5 == 0:
                    self.draw_timeline()
        info = (f"Pitch deviation: {pitch_difference:+.1f}°" if pitch_difference is not None
                else "Pitch: press CALIBRATE while facing forward")
        self.presence_label.config(text=f"Face: {'VISIBLE' if face_present else 'ABSENT'} | {info}")
        self.elapsed_label.config(text=f"Elapsed: {now - self.start_time:.1f}s")
        self.update_time_forecast()

        frame = cv.resize(frame, (470, 340))
        frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        photo = ImageTk.PhotoImage(Image.fromarray(frame))
        self.camera_label.config(image=photo, text="")
        self.camera_label.image = photo
        self.after_id = self.root.after(40, self.update_frame)

    def submit_answer(self):
        if not self.running:
            return
        answer = self.answer_entry.get()
        if not self.challenge.check_answer(self.question_idx, answer):
            self.wrong_attempts += 1
            self.feedback_label.config(text="Incorrect — try again.")
            return
        duration = time.perf_counter() - self.question_start
        row = self.save_question(duration)
        self.show_predictions(row)
        self.completed_durations.append(duration)
        self.question_idx += 1
        self.update_time_forecast()
        if self.question_idx >= len(self.challenge.questions):
            self.end_session(completed=True)
        else:
            self.feedback_label.config(text="Correct! Next question.")
            self.show_question()

    def save_question(self, duration):
        print("=== SAVING QUESTION ===")
        print("Samples:", len(self.samples))
        print("First samples:", self.samples[:5])
        print("Neutral:", self.face_tracker.neutral_pitch)
        pitches = [abs(diff) for visible, diff in self.samples if visible and diff is not None]
        frames = len(self.samples)
        question = self.challenge.questions[self.question_idx]
        row = {
            "participant_id": self.participant.get().strip(),
            "session_id": self.session_id,
            "question_index": self.question_idx + 1,
            "category": question.get("category", self.question_idx + 1),
            "difficulty": question.get("difficulty", self.question_idx + 1),
            "activity": self.mode,
            "duration_seconds": round(duration, 3),
            "wrong_attempts": self.wrong_attempts,
            "frames": frames,
            "face_visibility_ratio": round(sum(v for v, _ in self.samples) / frames, 4) if frames else 0.0,
            "pitch_coverage_ratio": round(len(pitches) / frames, 4) if frames else 0.0,
            "mean_pitch_deviation": round(mean(pitches), 3) if pitches else "",
            "pitch_variability": round(pstdev(pitches), 3) if len(pitches) > 1 else (0.0 if pitches else ""),
            "head_tilt_ratio": round(sum(p > 15 for p in pitches) / len(pitches), 4) if pitches else "",
            "calibrated": int(bool(pitches)),
        }
        DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
        with DATA_PATH.open("a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
            if f.tell() == 0:
                writer.writeheader()
            writer.writerow(row)
        print("Saved question:", row)
        return row

    def load_models(self):
        """Load the two trusted local model files; older filenames are also supported."""
        try:
            import joblib
        except ImportError:
            print("Model predictions disabled: pip install joblib pandas scikit-learn")
            return
        folder = Path(__file__).resolve().parent / "ml"
        names = {
            "A": ["model_a.joblib", "model_a_gameplay.joblib"],
            "B": ["model_b.joblib", "model_b_vision.joblib"],
        }
        default_features = {
            "A": ["difficulty", "duration_seconds", "wrong_attempts"],
            "B": ["difficulty", "duration_seconds", "wrong_attempts",
                  "face_visibility_ratio", "pitch_coverage_ratio", "mean_pitch_deviation",
                  "pitch_variability", "head_tilt_ratio"],
        }
        for key, candidates in names.items():
            path = next((folder / name for name in candidates
                         if (folder / name).exists()), None)
            if path is None:
                print(f"Model {key} unavailable: no saved .joblib in {folder}")
                continue
            try:
                # Only load joblib files YOU generated locally: joblib uses pickle.
                bundle = joblib.load(path)
                model = bundle["model"] if isinstance(bundle, dict) else bundle
                features = bundle.get("features", default_features[key]) if isinstance(bundle, dict) else default_features[key]
                self.models[key] = (model, features)
                print(f"Loaded Model {key}: {path.name} ({features})")
            except Exception as exc:
                print(f"Cannot load Model {key} from {path.name}: {exc}")

    def show_predictions(self, row):
        """Predict once a question is finished (not before)."""
        results = {"actual": row["activity"], "A": None, "B": None}
        for key, label in (("A", self.model_a_label), ("B", self.model_b_label)):
            name = "Gameplay" if key == "A" else "Gameplay + Vision"
            if key not in self.models:
                label.config(text=f"Model {key} — {name}: not loaded (run train_chronos.py)")
                continue
            model, features = self.models[key]
            try:
                import pandas as pd
                # Preserve feature names/order used at training time.
                features_row = {feature: pd.to_numeric(row.get(feature), errors="coerce")
                                for feature in features}
                X = pd.DataFrame([features_row], columns=features)
                # Some saved estimators use their own imputer; older ones do not.
                if not hasattr(model, "steps"):
                    X = X.fillna(-1)
                predicted = str(model.predict(X)[0])
                results[key] = predicted
                label.config(text=f"Model {key} — {name}: {predicted.upper()}")
            except Exception as exc:
                label.config(text=f"Model {key} — {name}: prediction error ({exc})")
        self.actual_label.config(text=f"Selected activity: {row['activity'].upper()}"
                                       f"  |  Q{row['question_index']}: {row['duration_seconds']:.1f}s")
        self.session_results.append(results)

    def draw_timeline(self):
        """Draw raw pitch deviations recorded in this question on a Tk canvas."""
        if not hasattr(self, "timeline_canvas"):
            return
        canvas = self.timeline_canvas
        canvas.delete("all")
        w, h = 470, 170
        left, right, top, bottom = 40, 12, 12, 27
        canvas.create_line(left, top, left, h-bottom, fill="#8192a8")
        canvas.create_line(left, h-bottom, w-right, h-bottom, fill="#8192a8")
        points = getattr(self, "timeline", [])
        if not points:
            canvas.create_text(w//2, 78, text="Start and calibrate to record movement",
                               fill="#bccbe0", font=("Arial", 11))
            return
        max_t = max(5.0, points[-1][0])
        max_p = max(15.0, max(deg for _, deg in points) * 1.15)
        coords = []
        for t, deg in points:
            x = left + (w-left-right) * t / max_t
            y = h-bottom - (h-top-bottom) * deg / max_p
            coords.extend([x, y])
        if len(coords) >= 4:
            canvas.create_line(*coords, fill="#52d8c5", width=2, smooth=False)
        else:
            x, y = coords
            canvas.create_oval(x-2, y-2, x+2, y+2, fill="#52d8c5", outline="")
        canvas.create_text(7, top, anchor="w", text=f"{max_p:.0f}°", fill="#bccbe0")
        canvas.create_text(7, h-bottom, anchor="w", text="0°", fill="#bccbe0")
        canvas.create_text(w-right, h-8, anchor="e", text=f"{max_t:.0f}s", fill="#bccbe0")

    def end_session(self, completed):
        if not self.running:
            return
        self.running = False
        self.recording = False
        if self.after_id is not None:
            self.root.after_cancel(self.after_id)
            self.after_id = None
        self.answer_entry.config(state="disabled")
        self.submit_button.config(state="disabled")
        self.calibrate_button.config(state="disabled")
        self.participant_entry.config(state="normal")
        self.mode_box.config(state="readonly")
        self.start_button.config(state="normal")
        self.cleanup_camera()
        if completed:
            self.progress_label.config(text="Progress: 5/5")
            self.question_label.config(text="Session completed!")
            # The session has ended: show actual results, not a live forecast.
            finished_elapsed = (
                time.perf_counter() - self.forecast_start
                if self.forecast_start is not None else None
            )
            self.eta_label.config(text="Estimated time remaining: 0.0s (completed)")
            self.future_label.config(text="Expected progress in 15s: 5/5 (completed)")
            if finished_elapsed is not None:
                margin = self.forecast_target_seconds - finished_elapsed
                self.margin_label.config(
                    text=f"90s target: {margin:+.1f}s (actual; not enforced)"
                )
            self.feedback_label.config(text=f"Saved 5 question records to {DATA_PATH}")
            if self.session_results:
                parts = []
                for key, name in (("A", "Gameplay"), ("B", "Gameplay + Vision")):
                    valid = [r for r in self.session_results if r.get(key) is not None]
                    if valid:
                        hits = sum(r[key] == r["actual"] for r in valid)
                        parts.append(f"{name}: {hits}/{len(valid)} correct")
                self.session_summary_label.config(
                    text="SESSION SUMMARY | " + "  •  ".join(parts)
                    if parts else "Session saved (models unavailable)."
                )
        else:
            self.eta_label.config(text="Estimated time remaining: Session stopped")
            self.future_label.config(text="Expected progress in 15s: Session stopped")
            self.feedback_label.config(text="Session stopped; completed questions remain saved.")

    def cleanup_camera(self):
        if self.face_tracker is not None:
            self.face_tracker.close()
            self.face_tracker = None
        if self.cap is not None:
            self.cap.release()
            self.cap = None

    def close(self):
        if self.running:
            self.end_session(completed=False)
        elif self.after_id is not None:
            self.root.after_cancel(self.after_id)
        self.root.destroy()


if __name__ == "__main__":
    root = tk.Tk()
    ChronosApp(root)
    root.mainloop()
