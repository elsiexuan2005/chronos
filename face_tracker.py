import cv2 as cv
import mediapipe as mp
import numpy as np
from pathlib import Path



class FaceTracker:
    def __init__(self):
        base_dir = Path(__file__).resolve().parent

        detector_path = base_dir / "face_detector.task"
        landmarker_path = base_dir / "face_landmarker.task"

        # Original face detector
        detector_options = mp.tasks.vision.FaceDetectorOptions(
            base_options=mp.tasks.BaseOptions(
                model_asset_path=str(detector_path)
            ),
            running_mode=mp.tasks.vision.RunningMode.VIDEO,
            min_detection_confidence=0.5
        )

        self.detector = mp.tasks.vision.FaceDetector.create_from_options(
            detector_options
        )

        # New face landmarker
        landmarker_options = mp.tasks.vision.FaceLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(
                model_asset_path=str(landmarker_path)
            ),
            running_mode=mp.tasks.vision.RunningMode.VIDEO,
            num_faces=1,
            output_facial_transformation_matrixes=True
        )

        self.landmarker = mp.tasks.vision.FaceLandmarker.create_from_options(
            landmarker_options
        )

        # Head tracking variables
        self.head_direction = "Uncalibrated"
        self.head_pitch = None
        self.neutral_pitch = None
        self.looking_down = False

    def calibrate(self):
        """Save the head orientation when looking straight ahead."""
        if self.head_pitch is not None:
            self.neutral_pitch = self.head_pitch
            print(f"Calibrated! Neutral pitch: {self.neutral_pitch:.2f}")
        else:
            print("Cannot calibrate: no face landmarks detected.")

    def detect(self, frame, timestamp_ms):
        rgb_frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)

        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=np.ascontiguousarray(rgb_frame)
        )

        # Detect face presence
        detections = self.detector.detect_for_video(
            mp_image, timestamp_ms
        )

        face_present = len(detections.detections) > 0

        for detection in detections.detections:
            bbox = detection.bounding_box

            x = bbox.origin_x
            y = bbox.origin_y
            w = bbox.width
            h = bbox.height

            cv.rectangle(
                frame,
                (x, y),
                (x + w, y + h),
                (0, 255, 0),
                2
            )

        # Estimate head orientation
        result = self.landmarker.detect_for_video(
            mp_image, timestamp_ms
        )

        # Debug once every approximately 2 seconds
        if timestamp_ms // 2000 != getattr(self, "_last_debug_interval", -1):
            self._last_debug_interval = timestamp_ms // 2000

            print(
                "FACE TRACKER:",
                "Detector faces =", len(detections.detections),
                "Landmarker faces =", len(result.face_landmarks),
                "Matrices =", len(result.facial_transformation_matrixes),
                flush=True
            )

        self.head_pitch = None
        self.head_direction = "Unknown"
        self.looking_down = False

        if result.facial_transformation_matrixes:
            matrix = np.asarray(
                result.facial_transformation_matrixes[0]
            )

            pitch = np.degrees(
                np.arctan2(
                    -matrix[2, 1],
                    matrix[2, 2]
                )
            )

            self.head_pitch = float(pitch)

            if self.neutral_pitch is None:
                self.head_direction = "Uncalibrated"
            else:
                difference = self.head_pitch - self.neutral_pitch

                if abs(difference) > 15:
                    self.head_direction = "Tilted"
                else:
                    self.head_direction = "Forward"

        return frame, face_present

    def close(self):
        self.detector.close()
        self.landmarker.close()