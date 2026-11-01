# Firearm Detection API

A FastAPI service that runs a YOLOv8 model to detect two classes, **gun** and **person**, in uploaded images and videos, with a small Next.js front end.

## How it works

- **Images:** returns each detection's class, confidence and bounding box as JSON, and can also return the image with boxes drawn on it.
- **Videos:** runs the model frame by frame on an uploaded video and returns the annotated video.
- **Confidence threshold:** detections below it are dropped (default 0.5, adjustable per request).
- **History:** the API keeps the last 1,000 detections in memory, so `/detections/history` and `/detections/stats` reset when it restarts.

## Endpoints

| Method | Path | Input | Output |
|---|---|---|---|
| GET | `/health` | | Model status and detection count |
| POST | `/detect/image` | Image | JSON detections (optionally an annotated image) |
| POST | `/detect/video` | Video | Annotated video |
| GET | `/detections/history` | | Recent detections |
| GET | `/detections/stats` | | Counts per class |

## Running it

The model weights are not included. Put a trained YOLOv8 checkpoint at `best.pt`.

```bash
docker-compose up --build firearm-detection   # API on http://localhost:8000
cd frontend && npm install && npm run dev     # Next.js front end
```
