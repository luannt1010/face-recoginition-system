import numpy as np
from retinaface import RetinaFace

class FaceDetector:
    def __init__(self):
        self.model = RetinaFace.build_model()

    def detect(self, frame):
        obj = RetinaFace.detect_faces(frame, model=self.model)
        if not obj:
            return None
        max_score_obj = max(obj.values(), key=lambda x: x["score"])
        bbox = max_score_obj["facial_area"]
        landmarks = max_score_obj["landmarks"]

        right_eye = tuple(np.asarray(landmarks["right_eye"]).astype(int))
        left_eye = tuple(np.asarray(landmarks["left_eye"]).astype(int))
        nose = tuple(np.asarray(landmarks["nose"]).astype(int))
        mouth_right = tuple(np.asarray(landmarks["mouth_right"]).astype(int))
        mouth_left = tuple(np.asarray(landmarks["mouth_left"]).astype(int))

        return {"bbox": bbox, "right_eye": right_eye, "left_eye": left_eye, "nose": nose, "mouth_right": mouth_right, "mouth_left": mouth_left}
