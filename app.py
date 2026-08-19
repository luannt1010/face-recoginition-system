import sys
import cv2
import requests
import numpy as np

from PyQt6.QtCore import QTimer, Qt
from PyQt6.QtGui import QImage, QPixmap, QColor
from PyQt6.QtWidgets import (QApplication, QLabel, QPushButton, QVBoxLayout,
                             QWidget, QHBoxLayout, QInputDialog, QMessageBox, QGraphicsDropShadowEffect)

from src.tracker import FaceTracker
from src.alignment import (calculate_area, calculate_center_dist, validate_face_pose, align_img)


WINDOW_SIZE = (1280, 720)
CAMERA_SIZE = (640, 480)
THRESHOLD = 0.7
MIN_AREA = 0.1
MAX_AREA = 0.3
DIST2CENTER_THRESHOLD = 30
POSE_THRESHOLD = 7
BLUE = (255, 0, 0)
GREEN = (0, 255, 0)
RED = (0, 0, 255)

center_point = (CAMERA_SIZE[0]//2, CAMERA_SIZE[1]//2)
REGISTER_RECT_PT1 = (center_point[0]-100, center_point[1]-140)
REGISTER_RECT_PT2 = (center_point[0]+100, center_point[1]+140)

WP = r".\checkpoints\test_SGD\checkpoints\face_embedding_model.onnx"

class APICilent:
    def __init__(self, base_url="http://127.0.0.1:8000"):
        self.base_url = base_url
        self.session = requests.Session()

    def _encode_img(self, frame):
        success, encoded = cv2.imencode(".jpg", frame)
        if not success:
            raise RuntimeError("Cannot encode frame.")
        return encoded.tobytes()

    def detect_many_faces(self, frame):
        frame = self._encode_img(frame)
        files = {"file": frame}
        response = self.session.post(url=f"{self.base_url}/detection/detect_many_faces", files=files)
        response.raise_for_status()
        faces = response.json()
        results = []
        if not faces or len(faces) == 0:
            return results
        print("STATUS:", response.status_code)
        print("RESPONSE:", response.text)
        for face in faces:
            face["bbox"] = np.asarray(face["bbox"], dtype=np.float32)
            face["kps"] = np.asarray(face["kps"], dtype=np.float32)
            results.append(face)
        return results

    def detect_one_face(self, frame):
        frame = self._encode_img(frame)
        files = {"file": frame}
        response = self.session.post(url=f"{self.base_url}/detection/detect_one_face", files=files)
        response.raise_for_status()
        face = response.json()
        result = []
        if not face or len(face) == 0:
            return result
        print("STATUS:", response.status_code)
        print("RESPONSE:", response.text)
        for f in face:
            f["bbox"] = np.asarray(f["bbox"], dtype=np.float32)
            f["kps"] = np.asarray(f["kps"], dtype=np.float32)
            result.append(f)
        return result

    def search_embedding(self, embedding):
        response = self.session.post(url=f"{self.base_url}/database/search", json={"embedding": embedding.tolist()})
        print("STATUS:", response.status_code)
        print("RESPONSE:", response.text)
        response.raise_for_status()
        return response.json()

    def insert_embedding(self, embedding, name):
        response = self.session.post(url=f"{self.base_url}/database/insert", json={"embedding": embedding.tolist(),
                                                                                   "name": str(name)})
        print("STATUS:", response.status_code)
        print("BODY:", response.text)
        response.raise_for_status()
        return response.json()

    def extract(self, frame):
        frame = self._encode_img(frame)
        files = {"file": frame}
        response = self.session.post(url=f"{self.base_url}/extraction/extract", files=files)
        print("STATUS:", response.status_code)
        print("RESPONSE:", response.text)
        response.raise_for_status()
        result = response.json()
        return np.asarray(result["embedding"], dtype=np.float32)
    
class FaceRecognitionSys(QWidget):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Face Recognition System")
        self.resize(WINDOW_SIZE[0], WINDOW_SIZE[1])

        self.cap = None
        self.mode = None
        self.register_name = None

        self.tracker = FaceTracker(fps=30)
        self.api_cilent = APICilent()
        self.identity_cache = {}


        self.camera_label = QLabel()
        self.camera_label.setFixedSize(CAMERA_SIZE[0], CAMERA_SIZE[1])

        self.camera_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.black_pixmap = QPixmap(CAMERA_SIZE[0], CAMERA_SIZE[1])
        self.black_pixmap.fill(Qt.GlobalColor.black)
        self.camera_label.setPixmap(self.black_pixmap)

        self.recognize_button = QPushButton("Run Recognize")
        self.register_button = QPushButton("Register Face")
        self.stop_button = QPushButton("Stop Camera")
        self.quit_button = QPushButton("Quit App")

        button_style = """QPushButton {background-color: #343A40; color: white; font-size: 18px; font-weight: bold;
        min-height: 20px; padding: 8px 22px; border: 2px solid #4B5259; border-radius: 10px;}
        QPushButton:hover {background-color: #495057; border: 2px solid #6C757D;}
        QPushButton:pressed {background-color: #212529;}
        QPushButton:disabled {background-color: #2A2A2A; color: #888888; border: 2px solid #333333;}"""
        quit_button_style = """QPushButton {background-color: rgb(255, 0, 0); color: white; font-size: 18px; font-weight: bold;
        min-height: 20px; padding: 8px 22px; border: 2px solid #4B5259; border-radius: 10px;}
        QPushButton:hover {background-color: #495057; border: 2px solid #6C757D;}
        QPushButton:pressed {background-color: #212529;}
        QPushButton:disabled {background-color: #2A2A2A; color: #888888; border: 2px solid #333333;}"""
        self.add_shadow(self.recognize_button)
        self.add_shadow(self.register_button)
        self.add_shadow(self.stop_button)
        self.add_shadow(self.quit_button)
        self.recognize_button.setStyleSheet(button_style)
        self.register_button.setStyleSheet(button_style)
        self.stop_button.setStyleSheet(button_style)
        self.quit_button.setStyleSheet(quit_button_style)
        self.stop_button.setEnabled(False)


        self.main_layout = QVBoxLayout()
        camera_layout = QVBoxLayout()
        camera_layout.addWidget(self.camera_label, alignment=Qt.AlignmentFlag.AlignCenter)
        button_layout = QHBoxLayout()
        button_layout.addWidget(self.recognize_button)
        button_layout.addWidget(self.register_button)
        button_layout.addWidget(self.stop_button)
        button_layout.addWidget(self.quit_button)
        self.main_layout.addLayout(camera_layout)
        self.main_layout.addLayout(button_layout)
        self.setLayout(self.main_layout)

        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)

        self.recognize_button.clicked.connect(self.start_recognize)
        self.register_button.clicked.connect(self.start_register)
        self.stop_button.clicked.connect(self.stop_camera)
        self.quit_button.clicked.connect(self.close)

    def add_shadow(self, button):
        shadow = QGraphicsDropShadowEffect(self)

        shadow.setBlurRadius(18)
        shadow.setOffset(0, 5)
        shadow.setColor(QColor(0, 0, 0, 200))

        button.setGraphicsEffect(shadow)

    def open_camera(self):
        if self.cap is not None:
            return True
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_SIZE[0])
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_SIZE[1])
        if not self.cap.isOpened():
            QMessageBox.critical(self, "Camera Error", "Cannot open camera.")
            self.cap.release()
            self.cap = None
            return False
        return True

    def start_recognize(self):
        if not self.open_camera():
            return
        self.mode = "recognize"
        self.register_name = None
        self.identity_cache.clear()
        # Button state
        self.recognize_button.setEnabled(False)
        self.register_button.setEnabled(False)
        self.stop_button.setEnabled(True)

        self.timer.start(30)

    def start_register(self):
        name, ok = QInputDialog.getText(self, "Register Face", "Enter your name:")
        if not ok:
            return
        name = name.strip()
        if not name:
            QMessageBox.warning(self, "Invalid Name", "Name cannot be empty.")
            return

        if not self.open_camera():
            return
        self.register_name = name
        self.mode = "register"
        self.identity_cache.clear()
        # Button state
        self.recognize_button.setEnabled(False)
        self.register_button.setEnabled(False)
        self.stop_button.setEnabled(True)
        self.timer.start(30)

    def stop_camera(self):
        self.timer.stop()
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        self.mode = None
        self.register_name = None
        self.identity_cache.clear()
        self.camera_label.setPixmap(self.black_pixmap)
        # Button state
        self.stop_button.setEnabled(False)
        self.recognize_button.setEnabled(True)
        self.register_button.setEnabled(True)

    def validate_face(self, frame, bbox, landmarks):
        distance_to_center = calculate_center_dist(frame, bbox)
        if distance_to_center > DIST2CENTER_THRESHOLD:
            return (False, "Pls, center your face in the frame.")

        face_size = calculate_area(frame, bbox)
        if face_size <= MIN_AREA:
            return (False, "Pls, bring your face closer.")
        if face_size >= MAX_AREA:
            return (False, "Pls, move your face further away.")

        nose = landmarks[2]
        right_eye = landmarks[0]
        left_eye = landmarks[1]

        nose_to_right, nose_to_left = validate_face_pose(nose, right_eye, left_eye)
        pose_difference = abs(nose_to_left - nose_to_right)
        if pose_difference > POSE_THRESHOLD:
            return (False, "Pls, look straight ahead.")
        return True, None

    def draw_warning(self, frame, message):
        cv2.putText(img=frame, text=message, org=(20, 75),fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=1.0, color=RED, thickness=2)

    def process_recognize(self, frame):
        faces = self.api_cilent.detect_many_faces(frame)
        tracks = self.tracker.update(faces)

        for track in tracks:
            track_id = track["track_id"]
            bbox = track["bbox"]
            landmarks = track["landmarks"]

            if track_id not in self.identity_cache:
                valid, message = self.validate_face(frame, bbox, landmarks)
                if not valid:
                    self.draw_warning(frame, message)
                    continue

                aligned_frame = align_img(frame, landmarks)
                embedding = self.api_cilent.extract(aligned_frame)
                results = self.api_cilent.search_embedding(embedding)
                similarity = results["similarity"]
                if similarity < THRESHOLD:
                    results["name"] = "Unknown"
                self.identity_cache[track_id] = results
            else:
                results = self.identity_cache[track_id]
            name, similarity = list(results.values())
            x1, y1, x2, y2 = bbox
            cv2.rectangle(frame, (x1, y1), (x2, y2), GREEN, 2)
            text = (
                f"ID:{track_id} "
                f"{name} "
                f"{similarity:.2f}"
            )
            cv2.putText(img=frame, text=text, org=(x1, y1 - 10), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=0.6, color=GREEN, thickness=2)
        return frame

    def process_register(self, frame):
        # copy only for display guide, main frame for insert
        frame_display = frame.copy()
        cv2.rectangle(frame_display, pt1=REGISTER_RECT_PT1, pt2=REGISTER_RECT_PT2, color=GREEN, thickness=3)
        faces = self.api_cilent.detect_one_face(frame)
        if faces is None or len(faces) == 0:
            self.draw_warning(frame_display, "No face detected.")
            return frame_display
        face = faces[0]
        bbox = face["bbox"]
        landmarks = face["kps"]

        valid, message = self.validate_face(frame, bbox, landmarks)
        if not valid:
            self.draw_warning(frame_display, message)
            return frame_display

        aligned_frame = align_img(frame, landmarks)
        embedding = self.api_cilent.extract(aligned_frame)
        try:
            self.api_cilent.insert_embedding(embedding, self.register_name)
        except Exception as e:
            QMessageBox.critical(self, "Register Error", f"Cannot register face:\n{e}")
            return frame_display
        # Insert successfully and keep aligned face on screen
        self.finish_register(aligned_frame)
        # None for not display live frame
        return None

    def finish_register(self, aligned_frame):

        # save name before reset
        registered_name = self.register_name
        self.timer.stop()
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        self.mode = None
        self.register_name = None
        self.identity_cache.clear()
        # displayt frame face cut instead live face
        self.display_frame(aligned_frame)

        self.stop_button.setEnabled(False)
        self.recognize_button.setEnabled(True)
        self.register_button.setEnabled(True)

        QMessageBox.information(self, "Register Success",(f"Face registered successfully for {registered_name}."))

    # main camera loop
    def update_frame(self):
        if self.cap is None:
            return

        ret, frame = self.cap.read()
        if not ret:
            return
        
        if self.mode == "recognize":
            frame = self.process_recognize(frame)
        elif self.mode == "register":
            frame = self.process_register(frame)

            # Register successfully
            # finish_register() self displayed aligned face 
            if frame is None:
                return
        # display live frame for recognize mode
        self.display_frame(frame)

    def display_frame(self, frame):
        if frame is None:
            return
        
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        height, width, channel = (frame_rgb.shape)
        bytes_per_line = (channel * width)
        # NUMPY -> QIMAGE
        q_image = QImage(frame_rgb.data, width, height, bytes_per_line, QImage.Format.Format_RGB888)
        # QIMAGE -> QPIXMAP
        pixmap = QPixmap.fromImage(q_image )

        self.camera_label.setPixmap(pixmap.scaled(self.camera_label.size(),
                                                  Qt.AspectRatioMode.KeepAspectRatio,
                                                  Qt.TransformationMode.SmoothTransformation))

    def closeEvent(self, event):
        self.timer.stop()
        if self.cap is not None:
            self.cap.release()
            self.cap = None
            self.identity_cache.clear()
        cv2.destroyAllWindows()
        event.accept()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = FaceRecognitionSys()
    window.show()
    sys.exit(app.exec())