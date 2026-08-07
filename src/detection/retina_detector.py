import numpy as np

class FaceDetector:
    def __init__(self):
        self._retina = None
        self.model = None

    def _load_model(self):
        if self.model is not None:
            return
        from retinaface import RetinaFace
        self._retina = RetinaFace
        self.model = self._retina.build_model()

    def warmup(self):
        self._load_model()

    def detect(self, frame):
        self._load_model()
        obj = self._retina.detect_faces(frame, model=self.model)
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
