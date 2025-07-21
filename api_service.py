from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import cv2
import numpy as np
from ultralytics import YOLO
import io
import base64
import time
import json
import os
from pathlib import Path
import uvicorn
from datetime import datetime
import threading
import queue

# Initialize FastAPI app
app = FastAPI(
    title="Firearm Detection API",
    description="Real-time firearm and person detection using YOLOv8",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables
model = None
detection_history = []
processing_queue = queue.Queue()

class DetectionRequest(BaseModel):
    confidence_threshold: float = 0.5
    save_image: bool = False

class DetectionResponse(BaseModel):
    success: bool
    detections: List[Dict[str, Any]]
    processing_time: float
    image_size: Dict[str, int]
    timestamp: str

class VideoDetectionRequest(BaseModel):
    confidence_threshold: float = 0.5
    output_format: str = "mp4"

class SystemStatus(BaseModel):
    model_loaded: bool
    gpu_available: bool
    total_detections: int
    uptime: float

def load_model():
    """Load the YOLOv8 model"""
    global model
    try:
        model_path = "best.pt"
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found at {model_path}")
        
        model = YOLO(model_path)
        print(f"✅ Model loaded successfully from {model_path}")
        return True
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return False

def get_detection_results(results, confidence_threshold=0.5):
    """Extract detection results from YOLOv8 output"""
    detections = []
    class_names = ['gun', 'person']
    
    for result in results:
        if result.boxes is not None:
            for box in result.boxes:
                confidence = float(box.conf[0])
                if confidence >= confidence_threshold:
                    class_id = int(box.cls[0])
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    
                    detection = {
                        "class": class_names[class_id],
                        "class_id": class_id,
                        "confidence": confidence,
                        "bbox": {
                            "x1": float(x1),
                            "y1": float(y1),
                            "x2": float(x2),
                            "y2": float(y2)
                        },
                        "center": {
                            "x": float((x1 + x2) / 2),
                            "y": float((y1 + y2) / 2)
                        }
                    }
                    detections.append(detection)
    
    return detections

def encode_image_to_base64(image):
    """Encode OpenCV image to base64 string"""
    _, buffer = cv2.imencode('.jpg', image)
    return base64.b64encode(buffer).decode('utf-8')

def draw_detections_on_image(image, detections):
    """Draw bounding boxes and labels on image"""
    annotated_image = image.copy()
    colors = [(0, 0, 255), (0, 255, 0)]  # Red for gun, Green for person
    
    for detection in detections:
        bbox = detection['bbox']
        x1, y1, x2, y2 = int(bbox['x1']), int(bbox['y1']), int(bbox['x2']), int(bbox['y2'])
        class_name = detection['class']
        confidence = detection['confidence']
        class_id = detection['class_id']
        
        # Draw bounding box
        color = colors[class_id]
        cv2.rectangle(annotated_image, (x1, y1), (x2, y2), color, 2)
        
        # Draw label
        label = f"{class_name}: {confidence:.2f}"
        label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
        cv2.rectangle(annotated_image, (x1, y1 - label_size[1] - 10), 
                     (x1 + label_size[0], y1), color, -1)
        cv2.putText(annotated_image, label, (x1, y1 - 5), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    return annotated_image

@app.on_event("startup")
async def startup_event():
    """Initialize the API on startup"""
    global model
    print("🚀 Starting Firearm Detection API...")
    
    # Load model
    if load_model():
        print("✅ API ready for detection requests")
    else:
        print("❌ Failed to load model - API may not function properly")

@app.get("/", response_model=Dict[str, str])
async def root():
    """Root endpoint"""
    return {
        "message": "Firearm Detection API",
        "version": "1.0.0",
        "status": "running"
    }

@app.get("/health", response_model=SystemStatus)
async def health_check():
    """Health check endpoint"""
    import torch
    
    return SystemStatus(
        model_loaded=model is not None,
        gpu_available=torch.cuda.is_available(),
        total_detections=len(detection_history),
        uptime=time.time() - app.start_time if hasattr(app, 'start_time') else 0
    )

@app.post("/detect/image", response_model=DetectionResponse)
async def detect_image(
    file: UploadFile = File(...),
    confidence_threshold: float = 0.5,
    save_image: bool = False,
    return_annotated: bool = False
):
    """Detect objects in uploaded image"""
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    # Validate file type
    if not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    try:
        # Read image
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            raise HTTPException(status_code=400, detail="Invalid image format")
        
        # Get image dimensions
        height, width = image.shape[:2]
        
        # Run detection
        start_time = time.time()
        results = model(image, verbose=False)
        processing_time = time.time() - start_time
        
        # Extract detections
        detections = get_detection_results(results, confidence_threshold)
        
        # Save to history
        detection_record = {
            "timestamp": datetime.now().isoformat(),
            "filename": file.filename,
            "detections": detections,
            "processing_time": processing_time
        }
        detection_history.append(detection_record)
        
        # Limit history size
        if len(detection_history) > 1000:
            detection_history.pop(0)
        
        # Save annotated image if requested
        annotated_image_base64 = None
        if return_annotated:
            annotated_image = draw_detections_on_image(image, detections)
            annotated_image_base64 = encode_image_to_base64(annotated_image)
        
        # Save image to disk if requested
        if save_image:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"detection_{timestamp}_{file.filename}"
            cv2.imwrite(filename, image)
        
        response_data = {
            "success": True,
            "detections": detections,
            "processing_time": processing_time,
            "image_size": {"width": width, "height": height},
            "timestamp": datetime.now().isoformat()
        }
        
        if annotated_image_base64:
            response_data["annotated_image"] = annotated_image_base64
        
        return DetectionResponse(**response_data)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Processing error: {str(e)}")

@app.post("/detect/video")
async def detect_video(
    file: UploadFile = File(...),
    confidence_threshold: float = 0.5,
    output_format: str = "mp4"
):
    """Detect objects in uploaded video"""
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    # Validate file type
    if not file.content_type.startswith('video/'):
        raise HTTPException(status_code=400, detail="File must be a video")
    
    try:
        # Save uploaded video temporarily
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        temp_input = f"temp_input_{timestamp}.mp4"
        temp_output = f"temp_output_{timestamp}.{output_format}"
        
        with open(temp_input, "wb") as buffer:
            contents = await file.read()
            buffer.write(contents)
        
        # Process video
        cap = cv2.VideoCapture(temp_input)
        if not cap.isOpened():
            raise HTTPException(status_code=400, detail="Could not open video file")
        
        # Get video properties
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        
        # Setup video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(temp_output, fourcc, fps, (width, height))
        
        frame_count = 0
        total_detections = 0
        start_time = time.time()
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Run detection
            results = model(frame, verbose=False)
            detections = get_detection_results(results, confidence_threshold)
            total_detections += len(detections)
            
            # Draw detections
            annotated_frame = draw_detections_on_image(frame, detections)
            out.write(annotated_frame)
            
            frame_count += 1
        
        processing_time = time.time() - start_time
        
        # Cleanup
        cap.release()
        out.release()
        os.remove(temp_input)
        
        # Return video file
        def generate_video():
            with open(temp_output, "rb") as video_file:
                while chunk := video_file.read(8192):
                    yield chunk
            os.remove(temp_output)
        
        return StreamingResponse(
            generate_video(),
            media_type=f"video/{output_format}",
            headers={
                "Content-Disposition": f"attachment; filename=detected_video.{output_format}",
                "X-Processing-Time": str(processing_time),
                "X-Total-Frames": str(frame_count),
                "X-Total-Detections": str(total_detections)
            }
        )
        
    except Exception as e:
        # Cleanup on error
        for temp_file in [temp_input, temp_output]:
            if os.path.exists(temp_file):
                os.remove(temp_file)
        raise HTTPException(status_code=500, detail=f"Processing error: {str(e)}")

@app.get("/detections/history")
async def get_detection_history(limit: int = 100):
    """Get recent detection history"""
    return {
        "total_detections": len(detection_history),
        "recent_detections": detection_history[-limit:] if detection_history else []
    }

@app.get("/detections/stats")
async def get_detection_stats():
    """Get detection statistics"""
    if not detection_history:
        return {"message": "No detections recorded yet"}
    
    # Calculate statistics
    total_detections = sum(len(record["detections"]) for record in detection_history)
    gun_detections = sum(
        len([d for d in record["detections"] if d["class"] == "gun"])
        for record in detection_history
    )
    person_detections = sum(
        len([d for d in record["detections"] if d["class"] == "person"])
        for record in detection_history
    )
    
    avg_processing_time = sum(record["processing_time"] for record in detection_history) / len(detection_history)
    
    return {
        "total_requests": len(detection_history),
        "total_detections": total_detections,
        "gun_detections": gun_detections,
        "person_detections": person_detections,
        "average_processing_time": avg_processing_time,
        "last_detection": detection_history[-1]["timestamp"] if detection_history else None
    }

@app.delete("/detections/history")
async def clear_detection_history():
    """Clear detection history"""
    global detection_history
    detection_history.clear()
    return {"message": "Detection history cleared"}

if __name__ == "__main__":
    # Set startup time
    app.start_time = time.time()
    
    # Run the API
    uvicorn.run(
        "api_service:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    ) 