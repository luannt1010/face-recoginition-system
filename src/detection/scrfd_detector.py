import numpy as np
from insightface.app import FaceAnalysis
from insightface.utils import face_align

class SCRFDDetector:
    def __init__(self):
        self.model = FaceAnalysis(name="buffalo_s", allowed_modules=["detection"], 
                                  providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
        self.det_size = (640, 640)
        self.model.prepare(ctx_id=0, det_thresh=0.5, det_size=self.det_size)

    def warmup(self):
        dummy = np.zeros((self.det_size[1], self.det_size[0], 3), dtype=np.uint8)
        self.model.get(dummy)

    def detect(self, frame):
        faces = self.model.get(frame)
        if not faces:
            return None
        return faces

    def align(self, frame, landmarks):
        return face_align.norm_crop(frame, landmarks, image_size=112)

# if __name__ == "__main__":
#     import cv2
#     from PIL import Image
#     img_path = r"C:\Users\luann\OneDrive\Pictures\1785774446035_207233635869047333_4293366375369274960_5c7cccc5553cedc273ca9cbc5a504fd3.jpg"
#     img_arr = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
#     detector = SCRFDDetector()
#     detector.warmup()
#     results = detector.detect(img_arr)
#     landmarks = results[0]["kps"]
#     res = detector.align(img_arr, landmarks)
#     img = Image.fromarray(res).convert("RGB")
#     img.show()
#     print(results)


    

