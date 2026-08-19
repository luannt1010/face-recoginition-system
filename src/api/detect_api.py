import numpy as np
import cv2
from fastapi import HTTPException, UploadFile, File, APIRouter
from pydantic import BaseModel
from src.detection import SCRFDDetector

detector = SCRFDDetector()
detector.warmup()

router = APIRouter(prefix="/detection", tags=["Detection"])


class Output(BaseModel):
    bbox: list[float]
    kps: list[list[float]]
    det_score: float 

async def read_frame(file: UploadFile):
    content = await file.read()
    if not content:
        raise HTTPException(status_code=400, detail="Image cannot be empty.")
    image_array = np.frombuffer(content, dtype=np.uint8)
    frame = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
    if frame is None:
        raise HTTPException(status_code=400, detail="Invalid image.")

    return frame

@router.post("/detect_one_face", response_model=list[Output])
async def detect_one_face(file: UploadFile=File(...)):
    try:
        frame = await read_frame(file)
        face = detector.detect_one_face(frame)
        result = []
        if face is None or len(face) == 0:
            return result
        for f in face:
            result.append({"bbox": f["bbox"].tolist(),
                           "kps": f["kps"].tolist(),
                           "det_score": float(f["det_score"])})
        return result
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Detect Failed {e}.")

@router.post("/detect_many_faces", response_model=list[Output])
async def detect_many_faces(file: UploadFile=File(...)):
    try:
        frame = await read_frame(file)
        faces = detector.detect_many_faces(frame)
        results = []
        if faces is None or len(faces) == 0:
            return results
        for face in faces:
            results.append({"bbox": face["bbox"].tolist(),
                            "kps": face["kps"].tolist(), 
                            "det_score": float(face["det_score"])})
        return results
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Detect Many Faces Failed {e}.")
