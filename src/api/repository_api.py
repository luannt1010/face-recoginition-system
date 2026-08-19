import numpy as np
from fastapi import FastAPI, HTTPException, APIRouter
from pydantic import BaseModel, Field
from src.database import FaceRepository

repository = FaceRepository()

class SearchInput(BaseModel):
    embedding: list[float] = Field(min_length=512, max_length=512)

class InsertInput(BaseModel):
    embedding: list[float] = Field(min_length=512, max_length=512)
    name: str 

class SeachOutput(BaseModel):
    name: str = Field(min_length=1, max_length=50)
    similarity: float

router = APIRouter(prefix="/database", tags=["Database"])

@router.post("/search", response_model=SeachOutput)
def search(request: SearchInput):
    try:
        embedding = np.asarray(request.embedding, dtype=np.float32)
        results = repository.search(embedding)
        if results is None:
            return {"name": "Unknow", "similarity": 0.0}
        return results
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Search Failed {e}.")

@router.post("/insert")
def insert(request: InsertInput):
    try:
        embedding = np.asarray(request.embedding, dtype=np.float32)
        repository.insert_embedding(embedding, request.name)
        return {"message": "Insert successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Insert Failed {e}.")
