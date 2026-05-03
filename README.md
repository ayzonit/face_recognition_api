# Face Detection API

A fully containerized video processing system that detects faces in video frames, stores bounding box data and serves annotated video streams built without using OpenCV.

---

## Overview

This project implements a face detection pipeline that processes videos asynchronously and returns both annotated output and structured ROI data.

The system is designed to:

- process videos frame-by-frame  
- detect faces using MediaPipe  
- draw bounding boxes without OpenCV  
- store per-frame ROI data  
- stream processed video efficiently  

---

## Architecture

<img width="1536" height="1024" alt="face_detection_image" src="https://github.com/user-attachments/assets/035a96a4-44b8-4526-98f2-4bc5c5f1ae5f" />

---

## Request Flow

1. Client uploads a video via `POST /upload`  
2. Backend processes the video asynchronously  
3. Frames are decoded using PyAV  
4. Face detection is performed on each frame  
5. Bounding boxes are drawn using Pillow  
6. ROI data is stored in PostgreSQL  
7. Processed video is encoded and saved  
8. Client can stream video or fetch ROI data  

---

## Components

### Backend
Handles video upload, processing, and streaming.

- FastAPI-based service  
- Uses MediaPipe for face detection  
- Uses Pillow for drawing bounding boxes  
- Uses PyAV for video decoding and encoding  

---

### Database
PostgreSQL database storing:

- job_id  
- frame number  
- bounding box coordinates  

---

### Frontend
Served using Nginx.

- Simple HTML/JavaScript interface  
- Proxies `/api/*` requests to backend  

---

### Docker
Containerized environment for all services.

- Services communicate over a private Docker network  
- Only port 80 is exposed  

---

## Tech Stack

- Backend: FastAPI (Python)  
- Face Detection: MediaPipe  
- Image Processing: Pillow  
- Video Processing: PyAV  
- Database: PostgreSQL  
- Frontend: HTML, JavaScript, Nginx  
- Containerization: Docker  

---

## API Behavior

### Upload

`POST /upload`

- Accepts a video file  
- Starts background processing  
- Returns a job ID  


### Stream

`GET /stream/{job_id}`

- Streams annotated video

### ROI Data

`GET /roi/{job_id}`

- Returns bounding box data per frame

### Design Decisions
#### No OpenCV:

Bounding boxes are drawn using Pillow instead of OpenCV, reducing dependency complexity while maintaining required functionality.

#### Frame-Level Processing:

Each frame is processed independently, ensuring consistent ROI extraction and simplifying storage.

#### Asynchronous Processing:

Video processing is handled in the background. The upload endpoint returns immediately with a job ID.

#### Containerized Architecture:

All services run in isolated Docker containers connected via a private network, ensuring reproducibility and clean separation of concerns.

### Setup and Installation
#### Prerequisites
- Docker

Author

Ayush Bhattacharjee
