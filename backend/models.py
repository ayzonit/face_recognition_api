import uuid
from sqlalchemy import Column, Integer, Float, DateTime
from sqlalchemy.dialects.postgresql import UUID
from pydantic import BaseModel, ConfigDict
from typing import Optional
from database import Base
from datetime import datetime
from zoneinfo import ZoneInfo 


class RoiDetection(Base):
    __tablename__ = "roi_detections"
    id = Column(Integer, primary_key=True, autoincrement=True)
    job_id = Column(UUID(as_uuid=True), nullable=False, index=True)
    frame_index = Column(Integer, nullable=False)
    x = Column(Float, nullable=False)
    y = Column(Float, nullable=False)
    width = Column(Float, nullable=False)
    height = Column(Float, nullable=False)
    confidence = Column(Float, nullable=True)
    timestamp = Column(Float, nullable=True)
    created_at = Column(DateTime(timezone=True), 
                        default=lambda: datetime.now(ZoneInfo("Asia/Kolkata")),
                        nullable=False)
    

class RoiResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: int
    job_id: uuid.UUID
    frame_index: int
    x: float
    y: float
    width: float
    height: float
    confidence: Optional[float]
    timestamp: Optional[float]
    created_at: datetime
    
    
class UploadResponse(BaseModel):
    job_id: uuid.UUID
    message: str
    frame_count: int
    faces_detected: int
    
    
class JobStatusResponse(BaseModel):
    job_id: uuid.UUID
    status: str