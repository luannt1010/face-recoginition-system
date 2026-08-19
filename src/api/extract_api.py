import cv2
import numpy as np
from fastapi import FastAPI, HTTPException, UploadFile, File, APIRouter
from pydantic import BaseModel, Field
from src.extractor import Extractor

WP = r".\checkpoints\test_SGD\checkpoints\face_embedding_model.onnx"
extractor = Extractor(WP)
extractor.warmup()

router = APIRouter(prefix="/extraction", tags=["Extraction"])


class Output(BaseModel):
    embedding: list[float] = Field(min_length=512, max_length=512)

async def read_frame(file: UploadFile):
    content = await file.read()
    if not content:
        raise HTTPException(status_code=400, detail="Image cannot be empty.")
    image_array = np.frombuffer(content, dtype=np.uint8)
    frame = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
    if frame is None:
        raise HTTPException(status_code=400, detail="Invalid image.")

    return frame

@router.post("/extract", response_model=Output)
async def extract(file: UploadFile=File(...)):
    try:
        frame = await read_frame(file)
        result = extractor.extract(frame)
        return {"embedding": result.tolist()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Extract Failed {e}.")
