
import os, uuid, asyncio
from fastapi import FastAPI, UploadFile, File, Depends, HTTPException, BackgroundTasks
from fastapi.responses import StreamingResponse
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from contextlib import asynccontextmanager
from database import init_db, get_db, AsyncSessionLocal
from models import RoiDetection, RoiResponse, UploadResponse, JobStatusResponse
from processing import process_video
from typing import Optional

UPLOAD_DIR = os.getenv("UPLOAD_DIR", "/app/uploads")
OUTPUT_DIR = os.getenv("OUTPUT_DIR", "/app/outputs")
MAX_UPLOAD_SIZE = int(os.getenv("MAX_UPLOAD_SIZE_MB", 200))

job_status: dict[str, str] = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    os.makedirs(UPLOAD_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    await init_db()
    yield
    
    
app = FastAPI(title="Face Detection API", 
              version="1.0.0", lifespan=lifespan, 
              root_path="/", openapi_url="/openapi.json",)



async def run_processing(job_id: uuid.UUID):
    try:
        job_status[str(job_id)] = "processing"
        async with AsyncSessionLocal() as db:
            await process_video(job_id, db)
        job_status[str(job_id)] = "done"
        
    except Exception as e:
        job_status[str(job_id)] = "failed"
        raise
    
    
@app.post("/api/upload", response_model=UploadResponse, status_code=202)
async def upload_video(background_tasks: BackgroundTasks, file: UploadFile = File()):
    
    if not file.content_type or not file.content_type.startswith("video/"):
        raise HTTPException(status_code=400, detail="Please upload a video.")
    contents = await file.read()
    if len(contents) > MAX_UPLOAD_SIZE * 1024 * 1024:
        raise HTTPException(status_code=400, detail=f"Please upload a file under {MAX_UPLOAD_SIZE}MB")
    
    job_id = uuid.uuid4()
    input_path = os.path.join(UPLOAD_DIR, f"{job_id}.mp4")
    
    with open(input_path, "wb") as f:
        f.write(contents)
        
    background_tasks.add_task(run_processing, job_id)
    
    return UploadResponse(job_id=job_id, message="Processing video.", frame_count=0, faces_detected=0)


@app.get("/api/stream/{job_id}", status_code=200)
async def stream_video(job_id: uuid.UUID):
    status = job_status.get(str(job_id))
    
    if status is None:
        raise HTTPException(status_code=404, detail="Job not found.")
    if status == "processing":
        raise HTTPException(status_code=202, detail="Video is getting processed.")
    if status == "failed":
        raise HTTPException(status_code=500, detail="Processing failed for this video.")
    
    output_path = os.path.join(OUTPUT_DIR, f"{job_id}.mp4")
    if not os.path.exists(output_path):
        raise HTTPException(status_code=404, detail="File not found.")
    
    def iter_file():
        with open(output_path, "rb") as f:
            while chunk := f.read(1024*1024):
                yield chunk
                
    return StreamingResponse(iter_file(), media_type="video/mp4")


@app.get("/api/roi/{job_id}", response_model=list[RoiResponse], status_code=200)
async def get_roi(job_id: uuid.UUID, frame: Optional[int] = None, db: AsyncSession = Depends(get_db)):
    status = job_status.get(str(job_id))
    
    if status is None:
        raise HTTPException(status_code=404, detail="Job not found.")
    if status == "processing":
        raise HTTPException(status_code=202, detail="Video is still processing.")
    
    query = select(RoiDetection).where(RoiDetection.job_id == job_id)
    
    if frame is not None:
        query = query.where(RoiDetection.frame_index == frame)
        
    query = query.order_by(RoiDetection.frame_index)
    result = await db.execute(query)
    records = result.scalars().all()
    
    if not records:
        raise HTTPException(status_code=404, detail="No detections were found for this job")
    
    return records