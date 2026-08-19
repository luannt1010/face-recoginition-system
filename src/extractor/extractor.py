import onnx
import cv2
import numpy as np
import onnxruntime as ort
from src.metrics import l2_normalize

class Extractor:
    def __init__(self, wp):
        self.wp = wp
        self.session = None
        self.input_name = None
        self.output_name = None

    def _load_model(self):
        if self.session is not None:
            return
        try:
            model = onnx.load(self.wp)
            onnx.checker.check_model(model)
            self.session = ort.InferenceSession(self.wp, providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
            self.input_name = self.session.get_inputs()[0].name
            self.output_name = self.session.get_outputs()[0].name
        except Exception as e:
            raise RuntimeError(f"Load ONNX model error: {e}")

    def warmup(self):
        self._load_model()
        dumppy_input = np.random.randn(1, 3, 112, 112).astype(np.float32)
        self.session.run([self.output_name], {self.input_name: dumppy_input})
        

    def _preprocess(self, frame):
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (112, 112))
        image = image.astype(np.float32) / 255.0
        mean, std = np.asarray([0.485, 0.456, 0.406], dtype=np.float32), np.asarray([0.229, 0.224, 0.225], dtype=np.float32)
        image = (image - mean) / std
        image = np.transpose(image, (2, 0, 1))
        return np.expand_dims(image, axis=0)
    
    def extract(self, frame):
        self._load_model()
        img = self._preprocess(frame)
        embedding = self.session.run([self.output_name], {self.input_name: img})[0]
        return l2_normalize(embedding).squeeze(0)


if __name__ == "__main__":
    import time
    wp = r"D:\private\face-recognition-system\checkpoints\final5\checkpoints\face_embedding_model.onnx"
    sd = r"D:\private\face-recognition-system\checkpoints\final5\checkpoints\best.pth"
    extractor = Extractor(wp)
    img_path = r"D:\private\dataset4embed_model\test\s18\1.bmp"
    frame = cv2.imread(img_path)
    start = time.perf_counter()
    embedding = extractor.extract(frame)
    end = time.perf_counter()
    print(embedding.shape)
    print((end-start)/60)   
    print(embedding.shape)
    print(embedding.dtype)


