# Firearm Detection API

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue?logo=python)](https://www.python.org/) [![FastAPI](https://img.shields.io/badge/FastAPI-API-green?logo=fastapi)](https://fastapi.tiangolo.com/) [![YOLOv8](https://img.shields.io/badge/YOLOv8-vision-orange?logo=github)](https://github.com/ultralytics/ultralytics) [![Docker](https://img.shields.io/badge/Docker-ready-blue?logo=docker)](https://www.docker.com/)

## Overview

**Firearm Detection API** is a production-ready, real-time computer vision system for detecting firearms and persons in images and videos. Built with FastAPI and powered by YOLOv8, it is designed for security, surveillance, and public safety applications. The API supports both image and video analysis, provides detection history and statistics, and is fully containerized for easy deployment.

---

## Key Features
- **Real-time detection** of firearms and persons in images and videos
- **YOLOv8** deep learning model for high accuracy and speed
- **FastAPI** backend with async endpoints and auto-generated docs
- **Detection history & statistics** endpoints for monitoring
- **Dockerized** for seamless deployment anywhere
- **Nginx reverse proxy** for production use
- **Extensible**: Easily swap models or add new classes

---

## Architecture
- **Model**: YOLOv8 (PyTorch, pre-trained and fine-tuned for firearm/person detection)
- **Backend**: FastAPI (Python 3.9+)
- **Deployment**: Docker Compose, Nginx (optional)
- **Workflow**:
  1. Upload image or video
  2. Model inference (GPU/CPU)
  3. Return JSON results and/or annotated media
  4. Store detection history and stats

---

## API Endpoints
- `GET /` — API info
- `GET /health` — Health check
- `POST /detect/image` — Detect objects in an image (returns JSON, optionally annotated image)
- `POST /detect/video` — Detect objects in a video (returns processed video)
- `GET /detections/history` — Recent detection history
- `GET /detections/stats` — Detection statistics

---

## Example Use Cases
- **Smart surveillance**: Real-time alerts for firearms in CCTV feeds
- **Event security**: Screening at stadiums, concerts, or public gatherings
- **Law enforcement**: Automated evidence review
- **Research**: Dataset creation, model benchmarking

---

## Quickstart

### Prerequisites
- Docker & Docker Compose
- Model file: `best.pt` in the `firearm_detection/` directory

### 1. Build & Run (API only)
```bash
docker-compose up --build firearm-detection
```

### 2. Build & Run (API + nginx proxy)
```bash
docker-compose up --build
```
- API: http://localhost:8000/
- Via nginx: http://localhost/firearm/

---

## File Structure
```
firearm_detection/
├── api_service.py
├── best.pt
├── Dockerfile.firearm-detection
├── requirements.firearm.txt
├── docker-compose.yml
├── nginx.conf
└── README.md
```

---

## Example Request
```bash
curl -X POST "http://localhost:8000/detect/image" -F "file=@test.jpg"
```

