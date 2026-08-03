import cv2
import numpy as np
from src.metrics import calculate_euclid_distance

RADIUS = 5
MIN_AREA = 0.2
MAX_AREA = 0.4
DIST2CENTER_THRESHOLD = 150
POSE_THRESHOLD = 7
RED = (0, 0, 255) #BGR
GREEN = (0, 255, 0)
ORG = (30, 50)
FONTSCALE = 1.0
THICKNESS = 2

def calculate_area(frame, bbox):
    frame_h, frame_w = frame.shape[:2]
    frame_area = frame_w * frame_h

    x1, y1, x2, y2 = bbox
    bbox_w = x2 - x1
    bbox_h = y2 - y1
    bbox_area = bbox_w * bbox_h

    return bbox_area / frame_area

def calculate_center_dist(frame, bbox):
    frame_h, frame_w = frame.shape[:2]
    frame_center = np.array([frame_h/2, frame_w/2])

    x1, y1, x2, y2 = bbox
    bbox_center = np.array([(y1+y2)/2, (x1+x2)/2])

    dist = calculate_euclid_distance(frame_center, bbox_center, normalize=False)
    return dist.squeeze()

def validate_face_pose(nose_point, right_point, left_point):
    nose2right = calculate_euclid_distance(nose_point, right_point, normalize=False)
    nose2left = calculate_euclid_distance(nose_point, left_point, normalize=False)
    return nose2right.squeeze(), nose2left.squeeze()

def return_landmark(frame):
    from retinaface import RetinaFace
    obj = RetinaFace.detect_faces(frame)
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

    return bbox, right_eye, left_eye, nose, mouth_right, mouth_left

def draw_img(frame, bbox, right_eye, left_eye, nose, mouth_right, mouth_left):
    img = frame.copy()
    top_left = tuple(bbox[:2])
    bottom_right = tuple(bbox[2:])
    img = cv2.rectangle(img, pt1=top_left, pt2=bottom_right, color=GREEN, thickness=3)
    img = cv2.circle(img, right_eye, RADIUS, RED, -1)
    img = cv2.circle(img, left_eye, RADIUS, RED, -1)
    img = cv2.circle(img, nose, RADIUS, RED, -1)
    img = cv2.circle(img, mouth_right, RADIUS, RED, -1)
    img = cv2.circle(img, mouth_left, RADIUS, RED, -1)
    return img
