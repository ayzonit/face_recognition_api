
import os, uuid, asyncio
from fastapi import FastAPI, UploadFile, File, Depends, HTTPException, BackgroundTasks
from fastapi.responses import StreamingResponse
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from contextlib import asynccontextmanager
from database import init_db, get_db
from models import RoiDetection, RoiResponse, UploadResponse, JobStatusResponse
from processing import process_video

UPLOAD_DIR = os.getenv("UPLOAD_DIR", "/app/uploads")
OUTPUT_DIR = os.getenv("OUTPUT_DIR", "/app/uploads")
MAX_UPLOAD_SIZE = int(os.getenv("MAX_UPLOAD_SIZE_MB", 100))

job_status: dict[str, str] = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    os.makedirs(UPLOAD_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    await init_db()
    yield
    
    
app = FastAPI(title="Face Detection API", version="1.0", lifespan=lifespan)

async def run_processing(job_id: uuid.UUID, db: AsyncSession):
    try:
        job_status[str(job_id)] = "processing"
        await process_video(job_id, db)
        job_status[str(job_id)] = "done"
        
    except Exception as e:
        job_status[str(job_id, db)] = "failed"
        raise e
    
    
@app.post("/upload", response_model=UploadResponse, status_code=202)
async def upload_video(background_tasks: BackgroundTasks, 
                       file: UploadFile = File(), db: AsyncSession = Depends(get_db)):
    
    if not file.content_type.startswith("video/"):
        raise HTTPException(status_code=400, detail="Please upload a video.")
    contents = await file.read()
    if len(contents) > MAX_UPLOAD_SIZE * 1024 * 1024:
        raise HTTPException(status_code=400, detail=f"Please upload a file under {MAX_UPLOAD_SIZE}MB")
    
    job_id: uuid.UUID
    input_path = os.path.join(UPLOAD_DIR, f"{job_id}.mp4")
    
    with open(input_path, "wb") as f:
        f.write(contents)
        
    background_tasks.add_task(run_processing, job_id, db)
    
    return UploadResponse(job_id=job_id, message="Processing video.", frame_count=0, faces_detected=0)