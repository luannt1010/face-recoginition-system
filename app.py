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
from PyQt6.QtWidgets import (QApplication, QFileDialog, QHBoxLayout, QLabel, QLineEdit,
                             QMainWindow, QMessageBox, QPushButton, QVBoxLayout, QWidget)
from src import (FaceRepository, calculate_area, calculate_center_dist, extract_embedding,
                 load_model, return_landmark, validate_face_pose, crop_face)

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


def load_embedding_model(checkpoint_path: str | Path) -> torch.nn.Module:
    """Nạp model tạo embedding từ checkpoint được chọn."""
    checkpoint = Path(checkpoint_path).expanduser().resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Not found checkpoint: {checkpoint}")
    model = load_model(model_type=MODEL_TYPE, model_size=MODEL_SIZE, embedding_dim=EMBEDDING_DIM,
                       dropout_rate=DROPOUT_RATE, sd_path=str(checkpoint))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    return model


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

    def __init__(self, model: torch.nn.Module, mode: OperationMode, name: str | None = None, camera_index: int = 0) -> None:
        super().__init__()

        self.model = model
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
        success_start_time: float | None = None

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
                    result = return_landmark(frame_copy)

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
                # Sau khi đã có frame hợp lệ, vẫn hiển thị camera trực tiếp
                # để hình không bị đứng hoặc giật trong lúc xác nhận.
                if success_start_time is not None:
                    self.frame_ready.emit(cv_to_qimage(frame_display))

                    if (time.perf_counter() - success_start_time >= CAPTURE_CONFIRMATION_SECONDS):
                        self.emit_status("Face is being processed...")
                        break

                    time.sleep(0.01)
                    continue

                with lock:
                    latest_frame = frame_orig.copy()
                    latest_frame_id += 1

                    result = latest_result
                    result_frame = latest_result_frame

                if result is None or result_frame is None:
                    self.emit_status("The face has not been identified.")

                else:
                    (bbox, right_eye, left_eye, nose, mouth_right, mouth_left) = result

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
                                cropped_face = crop_face(detection_frame, bbox, show=False, return_numpy=True)
                                if (cropped_face is None or cropped_face.size == 0):
                                    self.emit_status("It's not possible to crop the face.")
                                else:
                                    best_frame = cropped_face.copy()
                                    success_start_time = time.perf_counter()
                                    self.emit_status("Face is detected.")

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
    def __init__(
        self,
        model: torch.nn.Module,
        checkpoint_path: str | Path,
    ) -> None:
        super().__init__()

        self.model = model
        self.checkpoint_path = Path(checkpoint_path).resolve()
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

        self.checkpoint_label = QLabel("Checkpoint create embedding:")

        self.checkpoint_path_input = QLineEdit(
            str(self.checkpoint_path)
        )
        self.checkpoint_path_input.setReadOnly(True)
        self.checkpoint_path_input.setMinimumHeight(40)
        self.checkpoint_path_input.setToolTip(
            str(self.checkpoint_path)
        )

        self.select_checkpoint_button = QPushButton(
            "Choose checkpoint"
        )
        self.select_checkpoint_button.setMinimumHeight(40)
        self.select_checkpoint_button.clicked.connect(
            self.select_checkpoint
        )

        checkpoint_layout = QHBoxLayout()
        checkpoint_layout.addWidget(self.checkpoint_label)
        checkpoint_layout.addWidget(
            self.checkpoint_path_input,
            stretch=1,
        )
        checkpoint_layout.addWidget(self.select_checkpoint_button)

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
        layout.addLayout(checkpoint_layout)
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
        self.select_checkpoint_button.setEnabled(enabled)
        self.stop_camera_button.setEnabled(not enabled)

    def select_checkpoint(self) -> None:
        """Chọn và nạp checkpoint dùng để tạo embedding."""

        if self.worker is not None and self.worker.isRunning():
            QMessageBox.information(
                self,
                "Camera is running",
                "Please pause the camera before changing checkpoints.",
            )
            return

        selected_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select checkpoint model to create embeddings.",
            str(self.checkpoint_path.parent),
            "PyTorch checkpoint (*.pth *.pt);; All files (*)",
        )

        if not selected_path:
            return

        self.result_label.clear()
        self.status_label.setText("Loading checkpoint...")
        self.set_action_buttons_enabled(False)
        self.stop_camera_button.setEnabled(False)

        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        QApplication.processEvents()

        try:
            new_model = load_embedding_model(selected_path)

        except Exception as exc:
            self.status_label.setText("Can't load checkpoint")
            self.result_label.setText("Checkpoint is invalid")
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
            QMessageBox.critical(
                self,
                "Error checkpoint",
                f"Cannot load model from selected checkpoint:\n{exc}",
            )

        else:
            previous_model = self.model
            self.model = new_model
            self.checkpoint_path = Path(selected_path).resolve()

            self.checkpoint_path_input.setText(
                str(self.checkpoint_path)
            )
            self.checkpoint_path_input.setToolTip(
                str(self.checkpoint_path)
            )

            self.status_label.setText("Checkpoint has been successfully loaded.")
            self.result_label.setText(
                "The model for creating embeddings has been selected.:\n"
                f"{self.checkpoint_path.name}"
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

            del previous_model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        finally:
            QApplication.restoreOverrideCursor()
            self.set_action_buttons_enabled(True)

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
            "User: Unknown\n"
            f"Highest Similarity: {similarity:.4f}"
        )

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
    try:
        model = load_embedding_model(CHECKPOINT_PATH)
    except Exception as exc:
        QMessageBox.critical(None, "Initialization error", f"Unable to load the recognition model.:\n{exc}")
        return 1
    window = MainWindow(model=model, checkpoint_path=CHECKPOINT_PATH)
    window.show()
    return app.exec()

if __name__ == "__main__":
    raise SystemExit(main())
