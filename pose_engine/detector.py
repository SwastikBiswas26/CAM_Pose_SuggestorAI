import cv2
import numpy as np
import os
import urllib.request
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# Connection mapping for skeleton rendering
POSE_CONNECTIONS = [
    # Face outline / eyes (simplified)
    (0, 1), (1, 2), (2, 3), (0, 4), (4, 5), (5, 6),
    # Torso
    (11, 12), (11, 23), (12, 24), (23, 24),
    # Left Arm
    (11, 13), (13, 15),
    # Right Arm
    (12, 14), (14, 16),
    # Left Leg
    (23, 25), (25, 27), (27, 31), (27, 29), (29, 31),
    # Right Leg
    (24, 26), (26, 28), (28, 32), (28, 30), (30, 32)
]

class PoseDetector:
    """
    Pose detector class utilizing the modern MediaPipe Tasks API.
    Handles automatic model download, pose tracking, and custom rendering.
    """
    def __init__(self, model_name="pose_landmarker_full.task", detection_con=0.5, tracking_con=0.5):
        # Determine model paths
        self.model_path = os.path.join(os.path.dirname(__file__), model_name)
        self.download_model_if_missing(model_name)

        # Configure MediaPipe Tasks
        base_options = python.BaseOptions(model_asset_path=self.model_path)
        options = vision.PoseLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.IMAGE,
            output_segmentation_masks=False,
            min_pose_detection_confidence=detection_con,
            min_pose_presence_confidence=tracking_con
        )
        self.landmarker = vision.PoseLandmarker.create_from_options(options)
        self.results = None

    def download_model_if_missing(self, model_name):
        """
        Downloads the specified pre-trained MediaPipe task model if not present locally.
        """
        if not os.path.exists(self.model_path):
            print(f"Downloading {model_name} from Google APIs...")
            # Ensure the folder directory exists
            os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
            url = f"https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/1/{model_name}"
            try:
                urllib.request.urlretrieve(url, self.model_path)
                print("Model downloaded successfully.")
            except Exception as e:
                print(f"Error downloading model: {e}")
                # Fallback to local working directory if package path fails
                self.model_path = model_name
                if not os.path.exists(self.model_path):
                    print("Attempting local download fallback...")
                    urllib.request.urlretrieve(url, self.model_path)

    def process_frame(self, img):
        """
        Processes a BGR image frame and saves results.
        """
        # Convert BGR to RGB (MediaPipe expects RGB)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Convert numpy array to MediaPipe Image
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)
        
        # Perform detection
        self.results = self.landmarker.detect(mp_image)
        return self.results

    def find_landmarks(self, img, draw=False, draw_connections=True, color=(0, 255, 0)):
        """
        Extracts detected landmarks and optionally draws custom lines/points on img.
        """
        lm_list = []
        if not self.results or not self.results.pose_landmarks:
            return lm_list

        h, w, c = img.shape
        # Get first person detected landmarks
        landmarks = self.results.pose_landmarks[0]

        # Populate landmarker list
        for idx, lm in enumerate(landmarks):
            cx, cy = int(lm.x * w), int(lm.y * h)
            lm_list.append([idx, cx, cy, lm.x, lm.y, lm.z, lm.visibility])

        if draw:
            # Draw Connections first (so points sit on top)
            if draw_connections:
                for connection in POSE_CONNECTIONS:
                    start_idx, end_idx = connection
                    if start_idx < len(lm_list) and end_idx < len(lm_list):
                        # Ensure landmarks are visible enough
                        if lm_list[start_idx][6] > 0.4 and lm_list[end_idx][6] > 0.4:
                            p1 = (lm_list[start_idx][1], lm_list[start_idx][2])
                            p2 = (lm_list[end_idx][1], lm_list[end_idx][2])
                            cv2.line(img, p1, p2, color, 3)

            # Draw keypoint circles
            for item in lm_list:
                idx, cx, cy, _, _, _, vis = item
                if vis > 0.4:
                    # Draw a nice glowing circle
                    cv2.circle(img, (cx, cy), 5, (255, 255, 255), cv2.FILLED)
                    cv2.circle(img, (cx, cy), 7, color, 2)

        return lm_list

    def draw_guide_silhouette(self, img, landmarks_template, color=(180, 180, 180), thickness=2):
        """
        Draws a reference silhouette skeleton on the image based on a landmarks template.
        landmarks_template: list of {"id": idx, "x": norm_x, "y": norm_y}
        """
        if not landmarks_template:
            return
        h, w, c = img.shape
        pts = {}
        for lm in landmarks_template:
            idx = lm["id"]
            pts[idx] = (int(lm["x"] * w), int(lm["y"] * h))

        # Draw connections
        for connection in POSE_CONNECTIONS:
            start_idx, end_idx = connection
            if start_idx in pts and end_idx in pts:
                p1 = pts[start_idx]
                p2 = pts[end_idx]
                cv2.line(img, p1, p2, color, thickness, lineType=cv2.LINE_AA)

        # Draw joints
        for idx, pt in pts.items():
            cv2.circle(img, pt, 4, color, -1, lineType=cv2.LINE_AA)

