import numpy as np

class RetinaDetector:
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
        dummy = np.zeros((640, 640, 3), dtype=np.uint8)
        self._retina.detect_faces(dummy, model=self.model)

    def detect_many_faces(self, frame):
        self._load_model()
        obj = self._retina.detect_faces(frame, model=self.model)
        if not obj:
            return None
        results = []
        for face in obj.keys():
            f = obj[face]
            score = f["score"]
            bbox = f["facial_area"]
            landmarks = f["landmarks"]
            kps = np.array([landmarks["right_eye"], landmarks["left_eye"], landmarks["nose"], 
                            landmarks["mouth_right"], landmarks["mouth_left"]], dtype=np.float32)
            results.append({"bbox": bbox, "kps": kps, "det_score": score.astype(np.float32)})
        return results

    def detect_one_face(self, frame):
        self._load_model()
        obj = self._retina.detect_faces(frame, model=self.model)
        if not obj:
            return None
        max_score_obj = max(obj.values(), key=lambda x: x["score"])
        score = max_score_obj["score"]
        bbox = max_score_obj["facial_area"]
        landmarks = max_score_obj["landmarks"]

        right_eye = landmarks["right_eye"]
        left_eye = landmarks["left_eye"]
        nose = landmarks["nose"]
        mouth_right = landmarks["mouth_right"]
        mouth_left = landmarks["mouth_left"]
        landmarks = np.array([right_eye, left_eye, nose, mouth_right, mouth_left], dtype=np.float32)

        return [{"bbox": bbox,  "kps": landmarks, "det_score": score.asype(np.float32)}]

if __name__ == "__main__":
    import cv2
    img_path = r"C:\Users\luann\Downloads\images (1).jpg"
    img_arr = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
    detector = RetinaDetector()
    detector.warmup()
    results = detector.detect_many_faces(img_arr)
    print(results)

