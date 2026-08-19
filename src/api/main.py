from fastapi import FastAPI
from src.api.detect_api import router as detection_router
from src.api.extract_api import router as extraction_router
from src.api.repository_api import router as database_router
app = FastAPI(title="Face Recognition API")

app.include_router(detection_router)
app.include_router(extraction_router)
app.include_router(database_router)

@app.get("/")
def root():
    return {"message": "Face Recognition API"}