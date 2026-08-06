from __future__ import annotations

import sys
import threading
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Literal

import cv2
import numpy as np
import torch
from PyQt6.QtCore import QThread, Qt, pyqtSignal
from PyQt6.QtGui import QCloseEvent, QImage, QPixmap
from PyQt6.QtWidgets import (QApplication, QHBoxLayout, QLabel, QLineEdit,
                             QMainWindow, QMessageBox, QPushButton, QVBoxLayout, QWidget)
from src import (FaceRepository, calculate_area, calculate_center_dist, extract_embedding,
                 FaceDetector, load_model, validate_face_pose, crop_face)

MIN_AREA = 0.01
MAX_AREA = 0.35
DIST2CENTER_THRESHOLD = 150
POSE_THRESHOLD = 7

OperationMode = Literal["register", "identify"]

PROJECT_ROOT = Path(__file__).resolve().parent
CHECKPOINT_PATH = (PROJECT_ROOT / "checkpoints" / "final5" / "checkpoints" / "best.pth")

MODEL_TYPE = "mobile"
MODEL_SIZE = 18
EMBEDDING_DIM = 512
DROPOUT_RATE = 0.3

SIMILARITY_THRESHOLD = 0.6
CAPTURE_CONFIRMATION_SECONDS = 0.5

if not CHECKPOINT_PATH.is_file():
    raise FileNotFoundError(f"Not found checkpoint: {CHECKPOINT_PATH}")

DETECTOR = FaceDetector()
EXTRACTOR = load_model(model_type=MODEL_TYPE, model_size=MODEL_SIZE,
                       embedding_dim=EMBEDDING_DIM, dropout_rate=DROPOUT_RATE, sd_path=str(CHECKPOINT_PATH))


def cv_to_qimage(frame_bgr: np.ndarray) -> QImage:
    """Chuyển ảnh OpenCV BGR thành QImage."""
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    height, width, channels = frame_rgb.shape
    bytes_per_line = channels * width
    image = QImage(frame_rgb.data, width, height, bytes_per_line, QImage.Format.Format_RGB888)
    return image.copy()


def create_face_embedding(model: torch.nn.Module, face_bgr: np.ndarray) -> np.ndarray:
    if face_bgr is None or face_bgr.size == 0:
        raise ValueError("The image shows an empty face.")
    embedding = extract_embedding(model, face_bgr, show=False, crop=False)
    return embedding


class CameraWorker(QThread):
    frame_ready = pyqtSignal(QImage)
    status_changed = pyqtSignal(str)

    registered = pyqtSignal(str, QImage)
    recognized = pyqtSignal(str, float, QImage)
    unknown = pyqtSignal(float, QImage)

    error_occurred = pyqtSignal(str)

    def __init__(
        self,
        model: torch.nn.Module,
        detector: FaceDetector,
        mode: OperationMode,
        name: str | None = None,
        camera_index: int = 0,
    ) -> None:
        super().__init__()

        self.model = model
        self.detector = detector
        self.mode = mode
        self.name = name
        self.camera_index = camera_index

        self._stop_event = threading.Event()
        self._last_status: str | None = None

    def stop(self) -> None:
        self._stop_event.set()

    def emit_status(self, message: str) -> None:
        if message != self._last_status:
            self._last_status = message
            self.status_changed.emit(message)

    def process_captured_face(self, best_frame: np.ndarray) -> None:
        """Đăng ký hoặc nhận diện ảnh khuôn mặt đã crop."""

        embedding = create_face_embedding(model=self.model, face_bgr=best_frame)
        repository = FaceRepository()
        try:
            captured_image = cv_to_qimage(best_frame)
            if self.mode == "register":
                if not self.name:
                    raise ValueError("The username cannot be left blank.")

                repository.insert_embedding(embedding=embedding, name=self.name)
                self.registered.emit(self.name, captured_image,)
                return

            result = repository.search(embedding)
            if result is None:
                self.unknown.emit(0.0, captured_image)
                return

            # connection.py cấu hình dict_row, nhưng vẫn hỗ trợ tuple để
            # FaceRepository có thể đổi row factory mà app không bị lỗi.
            if isinstance(result, Mapping):
                name = result["name"]
                similarity = result["similarity"]
            else:
                name, similarity = result
            similarity = float(similarity)
            if similarity < SIMILARITY_THRESHOLD:
                self.unknown.emit(similarity, captured_image)
                return
            self.recognized.emit(str(name), similarity, captured_image)
        finally:
            repository.close()

    def run(self) -> None:
        self._stop_event.clear()
        self._last_status = None

        camera = cv2.VideoCapture(self.camera_index)

        if not camera.isOpened():
            self.error_occurred.emit("The camera can't turn on.")
            return

        camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

        lock = threading.Lock()
        detector_stop_event = threading.Event()

        latest_frame: np.ndarray | None = None
        latest_frame_id = 0

        latest_result = None
        latest_result_frame: np.ndarray | None = None

        best_frame: np.ndarray | None = None

        def detection_worker() -> None:
            nonlocal latest_result
            nonlocal latest_result_frame

            last_processed_frame_id = -1

            while (not self._stop_event.is_set() and not detector_stop_event.is_set()):
                frame_copy = None
                current_frame_id = -1
                with lock:
                    current_frame_id = latest_frame_id
                    if (latest_frame is not None and current_frame_id != last_processed_frame_id):
                        frame_copy = latest_frame.copy()

                if frame_copy is None:
                    time.sleep(0.01)
                    continue

                try:
                    result = self.detector.detect(frame_copy)

                except Exception as exc:
                    self.error_occurred.emit(f"Face detect error: {exc}")
                    self._stop_event.set()
                    return

                with lock:
                    latest_result = result
                    latest_result_frame = frame_copy
                last_processed_frame_id = current_frame_id
        detector_thread = threading.Thread(target=detection_worker, daemon=True)
        detector_thread.start()

        try:
            while not self._stop_event.is_set():
                ret, frame_orig = camera.read()

                if not ret:
                    self.error_occurred.emit("Can't read frame from camera.")
                    break

                frame_display = frame_orig.copy()
                cv2.rectangle(frame_display, pt1=[437, 116], pt2=[850, 689], color=(0, 255, 0), thickness=3)

                with lock:
                    latest_frame = frame_orig.copy()
                    latest_frame_id += 1

                    result = latest_result
                    result_frame = latest_result_frame

                if result is None or result_frame is None:
                    self.emit_status("The face has not been identified.")

                else:
                    bbox = result["bbox"]
                    right_eye = result["right_eye"]
                    left_eye = result["left_eye"]
                    nose = result["nose"]

                    # Bbox và landmark phải được áp dụng lên đúng frame đã
                    # đưa vào RetinaFace, tránh crop lệch khi detector chậm.
                    detection_frame = result_frame.copy()
                    distance_to_center = calculate_center_dist( detection_frame, bbox)

                    if distance_to_center > DIST2CENTER_THRESHOLD:
                        self.emit_status("Please center your face in the frame.")
                    else:
                        face_size = calculate_area(detection_frame, bbox)

                        if face_size <= MIN_AREA:
                            self.emit_status("Please bring your face closer.")
                        elif face_size > MAX_AREA:
                            self.emit_status("Please move your face further away." )

                        else:
                            nose_to_right, nose_to_left = validate_face_pose(nose, right_eye, left_eye)
                            pose_difference = abs(nose_to_left - nose_to_right)
                            if pose_difference > POSE_THRESHOLD:
                                self.emit_status("Please look straight ahead.")
                            else:
                                cropped_face = crop_face(img=detection_frame, bbox=bbox, show=False, return_numpy=True)
                                if (cropped_face is None or cropped_face.size == 0):
                                    self.emit_status("It's not possible to crop the face.")
                                else:
                                    best_frame = cropped_face
                                    self.emit_status("Face is detected.")
                                    self.frame_ready.emit(cv_to_qimage(frame_display))

                                    # Stop queuing camera frames so the GUI can display
                                    # the detected status immediately and keep it visible.
                                    if not self._stop_event.wait(CAPTURE_CONFIRMATION_SECONDS):
                                        self.emit_status("Face is being processed...")
                                    break

                self.frame_ready.emit(cv_to_qimage(frame_display))

        except Exception as exc:
            self.error_occurred.emit(f"Camera stream processing error: {exc}")
            self._stop_event.set()
        finally:
            detector_stop_event.set()

            camera.release()

            detector_thread.join()

        if self._stop_event.is_set():
            return

        if best_frame is None:
            return

        try:
            self.process_captured_face(best_frame)

        except Exception as exc:
            self.error_occurred.emit(f"Face processing error: {exc}")


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()

        self.model = EXTRACTOR.eval()
        self.worker: CameraWorker | None = None

        self.setWindowTitle("Face Recognition System")
        self.resize(1000, 780)

        self.camera_label = QLabel("The camera is not turned on.")
        self.camera_label.setAlignment(
            Qt.AlignmentFlag.AlignCenter
        )
        self.camera_label.setMinimumSize(900, 550)
        self.camera_label.setStyleSheet(
            """
            QLabel {
                background-color: #202020;
                color: white;
                border-radius: 8px;
            }
            """
        )

        self.status_label = QLabel("Ready")
        self.status_label.setAlignment(
            Qt.AlignmentFlag.AlignCenter
        )
        self.status_label.setStyleSheet(
            """
            QLabel {
                font-size: 18px;
                font-weight: 600;
                padding: 8px;
            }
            """
        )

        self.result_label = QLabel("")
        self.result_label.setAlignment(
            Qt.AlignmentFlag.AlignCenter
        )
        self.result_label.setStyleSheet(
            """
            QLabel {
                font-size: 22px;
                font-weight: bold;
                padding: 8px;
            }
            """
        )


        self.name_input = QLineEdit()
        self.name_input.setPlaceholderText(
            "Enter the name of the person to be registered."
        )
        self.name_input.setMinimumHeight(40)
        self.name_input.setMaxLength(50)

        self.register_button = QPushButton(
            "Register your face."
        )
        self.register_button.setMinimumHeight(45)
        self.register_button.clicked.connect(
            self.start_registration
        )

        self.identify_button = QPushButton(
            "Facial Recognition"
        )
        self.identify_button.setMinimumHeight(45)
        self.identify_button.clicked.connect(
            self.start_identification
        )

        self.stop_camera_button = QPushButton(
            "Stop Camera"
        )
        self.stop_camera_button.setMinimumHeight(45)
        self.stop_camera_button.clicked.connect(
            self.stop_camera
        )
        self.stop_camera_button.setEnabled(False)

        self.close_button = QPushButton(
            "Quit App"
        )
        self.close_button.setMinimumHeight(45)
        self.close_button.clicked.connect(self.close)

        action_layout = QHBoxLayout()
        action_layout.addWidget(self.register_button)
        action_layout.addWidget(self.identify_button)
        action_layout.addWidget(self.stop_camera_button)
        action_layout.addWidget(self.close_button)

        layout = QVBoxLayout()
        layout.addWidget(self.camera_label, stretch=1)
        layout.addWidget(self.name_input)
        layout.addLayout(action_layout)
        layout.addWidget(self.status_label)
        layout.addWidget(self.result_label)

        container = QWidget()
        container.setLayout(layout)

        self.setCentralWidget(container)

    def set_action_buttons_enabled(
        self,
        enabled: bool,
    ) -> None:
        self.register_button.setEnabled(enabled)
        self.identify_button.setEnabled(enabled)
        self.stop_camera_button.setEnabled(not enabled)


    def start_registration(self) -> None:
        name = self.name_input.text().strip()

        if not name:
            QMessageBox.warning(
                self,
                "Name missing",
                "Please enter the name of the person you wish to register.",
            )
            return

        self.start_camera_worker(
            mode="register",
            name=name,
        )

    def start_identification(self) -> None:
        self.start_camera_worker(
            mode="identify",
        )

    def start_camera_worker(
        self,
        mode: OperationMode,
        name: str | None = None,
    ) -> None:
        if self.worker is not None and self.worker.isRunning():
            QMessageBox.information(
                self,
                "Camera is running",
                "Please wait for the current camera to complete.",
            )
            return

        self.result_label.clear()
        self.status_label.setText("Opening camera...")

        self.set_action_buttons_enabled(False)

        self.worker = CameraWorker(
            model=self.model,
            detector=DETECTOR,
            mode=mode,
            name=name,
            camera_index=0,
        )

        self.worker.frame_ready.connect(
            self.update_camera_frame
        )

        self.worker.status_changed.connect(
            self.status_label.setText
        )

        self.worker.registered.connect(
            self.show_registration_result
        )

        self.worker.recognized.connect(
            self.show_recognition_result
        )

        self.worker.unknown.connect(
            self.show_unknown_result
        )

        self.worker.error_occurred.connect(
            self.show_error
        )

        self.worker.finished.connect(
            self.on_worker_finished
        )

        self.worker.start()

    def stop_camera(self) -> None:
        if self.worker is not None and self.worker.isRunning():
            self.worker.stop()
            self.status_label.setText("Stopping camera...")

    def update_camera_frame(self, image: QImage) -> None:
        pixmap = QPixmap.fromImage(image)

        scaled_pixmap = pixmap.scaled(
            self.camera_label.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )

        self.camera_label.setPixmap(scaled_pixmap)

    def show_registration_result(
        self,
        name: str,
        captured_image: QImage,
    ) -> None:
        self.update_camera_frame(captured_image)

        self.status_label.setText(
            "Face registration successful"
        )

        self.result_label.setText(
            f"Registration successful: {name}"
        )

        self.result_label.setStyleSheet(
            """
            QLabel {
                font-size: 22px;
                font-weight: bold;
                color: #16833b;
                padding: 8px;
            }
            """
        )

    def show_recognition_result(
        self,
        name: str,
        similarity: float,
        captured_image: QImage,
    ) -> None:
        self.update_camera_frame(captured_image)

        self.status_label.setText(
            "Face recognition successful"
        )

        self.result_label.setText(
            f"Name: {name} - Similarity: {round(similarity, 2)*100}%"
        )
        self.result_label.setStyleSheet(
            """
            QLabel {
                font-size: 22px;
                font-weight: bold;
                color: #16833b;
                padding: 8px;
            }
            """
        )

    def show_unknown_result(
        self,
        similarity: float,
        captured_image: QImage,
    ) -> None:
        self.update_camera_frame(captured_image)

        self.status_label.setText(
            "User not identified"
        )

        self.result_label.setText(
            f"Name: Unknown\n - Similarity: {round(similarity, 2)*100}")

        self.result_label.setStyleSheet(
            """
            QLabel {
                font-size: 22px;
                font-weight: bold;
                color: #c0392b;
                padding: 8px;
            }
            """
        )

    def show_error(self, message: str) -> None:
        self.status_label.setText(message)
        self.result_label.setText("An error has occurred.")
        QMessageBox.critical(self, "Error", message)

    def on_worker_finished(self) -> None:
        if self.status_label.text() == "Stopping camera...":
            self.status_label.setText("Stopped camera")

        self.set_action_buttons_enabled(True)

        if self.worker is not None:
            self.worker.deleteLater()
            self.worker = None

    def closeEvent(self, event: QCloseEvent) -> None:
        if self.worker is not None and self.worker.isRunning():
            self.worker.stop()
            self.worker.wait(5000)

        event.accept()


def main() -> int:
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    return app.exec()

if __name__ == "__main__":
    raise SystemExit(main())
