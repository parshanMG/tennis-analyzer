import cv2, mediapipe as mp
import numpy as np
from collections import namedtuple

Keypoints = namedtuple("Keypoints", ["landmarks", "visibility"])

class PoseEstimator:
    def __init__(self, static_image_mode=False, model_complexity=1, smooth_landmarks=True):
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=static_image_mode,
            model_complexity=model_complexity,
            smooth_landmarks=smooth_landmarks)
        self.drawer = mp.solutions.drawing_utils

    def process_frame(self, frame_bgr):
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        res = self.pose.process(rgb)
        if not res.pose_landmarks:
            return None
        lm = res.pose_landmarks.landmark
        pts = np.array([[p.x, p.y, p.z] for p in lm], dtype=np.float32)
        vis = np.array([p.visibility for p in lm], dtype=np.float32)
        return Keypoints(landmarks=pts, visibility=vis)

    def draw(self, frame_bgr, keypoints):
        if keypoints is None: return frame_bgr
        # Fake a landmark list object to reuse drawing utils
        class _L: pass
        l = _L(); l.landmark = []
        for (x, y, z), v in zip(keypoints.landmarks, keypoints.visibility):
            p = self.mp_pose.PoseLandmark.NOSE  # dummy init
            class _P: pass
            q = _P(); q.x, q.y, q.z, q.visibility = float(x), float(y), float(z), float(v)
            l.landmark.append(q)
        self.drawer.draw_landmarks(frame_bgr, l, self.mp_pose.POSE_CONNECTIONS)
        return frame_bgr
