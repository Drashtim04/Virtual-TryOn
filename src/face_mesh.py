import cv2
import mediapipe as mp
import numpy as np
import time
import os

from mediapipe.tasks import python
from mediapipe.tasks.python import vision

class FaceMeshDetector:
    def __init__(self, model_path='assets/face_landmarker.task', max_faces=1):
        self.model_path = model_path
        
        # Verify model exists
        if not os.path.exists(self.model_path):
            # Fallback for relative paths if run from src or main
            if os.path.exists(os.path.join("..", self.model_path)):
                self.model_path = os.path.join("..", self.model_path)
            # Check if it is in an assets folder next to the script
            elif os.path.exists(os.path.join(os.path.dirname(__file__), "..", "assets", "face_landmarker.task")):
                 self.model_path = os.path.join(os.path.dirname(__file__), "..", "assets", "face_landmarker.task")
            else:
                raise FileNotFoundError(f"Model not found at {self.model_path}. Please download it.")

        base_options = python.BaseOptions(model_asset_path=self.model_path)
        options = vision.FaceLandmarkerOptions(
            base_options=base_options,
            output_face_blendshapes=False,
            output_facial_transformation_matrixes=False,
            num_faces=max_faces,
            running_mode=vision.RunningMode.LIVE_STREAM,
            result_callback=self.print_result
        )
        self.detector = vision.FaceLandmarker.create_from_options(options)
        self.results = None
        self.last_timestamp_ms = 0

    def print_result(self, result: vision.FaceLandmarkerResult, output_image: mp.Image, timestamp_ms: int):
        self.results = result

    def find_face_mesh(self, img, draw=True):
        # Convert to MediaPipe Image
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        
        # Calculate timestamp (must be monotonically increasing)
        timestamp_ms = int(time.time() * 1000)
        if timestamp_ms <= self.last_timestamp_ms:
            timestamp_ms = self.last_timestamp_ms + 1
        self.last_timestamp_ms = timestamp_ms

        # Detect asynchronously
        self.detector.detect_async(mp_image, timestamp_ms)
        
        # Note: Since it's async, self.results might be from the previous frame.
        # Ideally we wait or sync, but for visualization async is fine.
        
        faces = []
        if self.results and self.results.face_landmarks:
            for face_landmarks in self.results.face_landmarks:
                face = []
                ih, iw, ic = img.shape
                for lm in face_landmarks:
                    x, y = int(lm.x * iw), int(lm.y * ih)
                    face.append([x, y])
                faces.append(face)
            
            # Draw (simple dot drawing as mp.solutions.drawing_utils is not available)
            if draw and faces:
                for face in faces:
                    for (x, y) in face:
                        cv2.circle(img, (x, y), 1, (255, 255, 255), -1)

        return img, faces
