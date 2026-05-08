import os, av, uuid, mediapipe as mp, numpy as np
from PIL import Image, ImageDraw
from sqlalchemy.ext.asyncio import AsyncSession
from models import RoiDetection

UPLOAD_DIR = os.getenv("UPLOAD_DIR", "/app/uploads")
OUTPUT_DIR = os.getenv("OUTPUT_DIR", "/app/outputs")

mp_face = mp.solutions.face_detection

def draw_roi(frame: np.ndarray, x, y, w, h) -> np.ndarray:
    img = Image.fromarray(frame)
    draw = ImageDraw.Draw(img)
    fh, fw = frame.shape[:2]
    x0 = int(x*fw)
    y0 = int(y*fh)
    x1 = int((x+w)*fw)
    y1 = int((y+h)*fh)
    x0, y0 = max(0, x0), max(0, y0)
    x1, y1 = min(fw-1, x1), min(fh-1, y1)
    draw.rectangle([x0, y0, x1, y1], outline=(0, 255, 0), width=3)
    return np.array(img)


def clamp(val: float) -> float:
    return max(0.0, min(1.0, val))


async def process_video(job_id: uuid.UUID, db: AsyncSession):
    input_path = os.path.join(UPLOAD_DIR, f"{job_id}.mp4")
    output_path = os.path.join(OUTPUT_DIR, f"{job_id}.mp4")
    frame_count = 0
    faces_detected = 0
    roi_records = []
    
    try:
        with mp_face.FaceDetection(model_selection=0, min_detection_confidence=0.4) as detector:
            frames = []
            with av.open(input_path) as inp:
                video_stream = inp.streams.video[0]
                fps = float(video_stream.average_rate)
                width = video_stream.width
                height = video_stream.height
                for frame in inp.decode(video=0):
                    rgb = frame.to_ndarray(format="rgb24")
                    ts = float(frame.pts * video_stream.time_base) if frame.pts is not None else 0.0
                    results = detector.process(rgb)
                    if results.detections:
                        d = results.detections[0]
                        bb = d.location_data.relative_bounding_box
                        x = clamp(bb.xmin)
                        y = clamp(bb.ymin)
                        w = clamp(bb.width)
                        h = clamp(bb.height)
                        conf = float(d.score[0])
                        roi_records.append(RoiDetection(job_id=job_id, 
                                    frame_index=frame_count, 
                                    x=x, 
                                    y=y,
                                    width=w, 
                                    height=h,
                                    confidence = conf,
                                    timestamp = ts,
                            )
                        )
                        annotated = draw_roi(rgb, x, y, w, h)
                        faces_detected += 1
                    else:
                        annotated = rgb
                        
                    frames.append(annotated)
                    frame_count += 1
                    
            if roi_records:
                db.add_all(roi_records)
                await db.commit()
                
            with av.open(output_path, "w") as out:
                stream = out.add_stream("libx264", rate=fps)
                stream.width = width
                stream.height = height
                stream.pix_fmt = "yuv420p"
                
                for i, annotated_frame in enumerate(frames):
                    vf = av.VideoFrame.from_ndarray(annotated_frame, format="rgb24")
                    vf.pts = i
                    for pkt in stream.encode(vf):
                        out.mux(pkt)
                for pkt in stream.encode(None):
                    out.mux(pkt)
                    
    except Exception as e:
        raise RuntimeError(f"Video processing failed for job {job_id}") from e
    
    return {"frame_count": frame_count, "faces_detected": faces_detected}