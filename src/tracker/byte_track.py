import numpy as np
import supervision as sv
from trackers import ByteTrackTracker


class FaceTracker:
    def __init__(self, fps=30):
        self.tracker = ByteTrackTracker(frame_rate=fps, track_activation_threshold=0.5, high_conf_det_threshold=0.6,
                                        lost_track_buffer=30, minimum_consecutive_frames=2, minimum_iou_threshold=0.1)

    def update(self, faces):
        
        if faces is None or len(faces) == 0:
            detections = sv.Detections(xyxy=np.empty((0, 4), dtype=np.float32), confidence=np.empty((0,), dtype=np.float32))
            self.tracker.update(detections)
            return []

        bboxes = np.array([face["bbox"] for face in faces], dtype=np.float32)
        scores = np.array( [face["det_score"] for face in faces], dtype=np.float32)

        landmarks = np.array([face["kps"] for face in faces], dtype=np.float32)

        detections = sv.Detections(xyxy=bboxes, confidence=scores, data={"landmarks": landmarks})
        tracked = self.tracker.update(detections)
        results = []
        for bbox, score, track_id, landmark in zip(tracked.xyxy, tracked.confidence,
                                                   tracked.tracker_id, tracked.data["landmarks"]):
            # -1 = chưa đủ điều kiện trở thành track ổn định
            if track_id == -1:
                continue
            results.append({"track_id": int(track_id), "bbox": bbox.astype(int), 
                            "score": float(score), "landmarks": landmark.astype(int)})

        return results