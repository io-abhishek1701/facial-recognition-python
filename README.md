# QuickFace AI

## Overview

QuickFace AI is a real-time facial recognition system built using **FastAPI**, **FaceNet**, **MTCNN**, **OpenCV**, and **SQLite**. The application supports enrolling a person using up to **five facial images**, generates facial embeddings, and performs real-time recognition through a webcam using deep learning.

The project demonstrates the complete facial recognition pipeline, including face detection, feature extraction, embedding comparison, and live recognition through a simple web interface.

---

## Features

- Real-time facial recognition using a webcam
- Face enrollment using up to five images per person
- Face detection using MTCNN
- Face embedding generation using FaceNet
- Cosine similarity-based face matching
- SQLite database for storing enrolled users
- Live confidence score display
- REST API built with FastAPI
- Interactive frontend using HTML, CSS, and JavaScript
- Image preview during enrollment
- View all enrolled persons
- Delete enrolled persons
- Backend health monitoring
- Responsive user interface

---

## Technology Stack

### Backend

- Python 3.10
- FastAPI
- SQLAlchemy
- SQLite
- OpenCV
- MTCNN
- keras-facenet
- NumPy
- Uvicorn

### Frontend

- HTML5
- CSS3
- JavaScript

---

## Project Structure

```text
QuickFace-AI/
│
├── backend/
│   ├── app.py
│   ├── database.py
│   ├── models.py
│   ├── schemas.py
│   ├── face_service.py
│   ├── recognition.py
│   ├── utils.py
│   ├── quickface.db
│   └── enrolled_faces/
│
├── frontend/
│   ├── index.html
│   ├── style.css
│   └── script.js
│
├── requirements.txt
└── README.md
```

---

## System Workflow

### Face Enrollment

1. Enter the person's name.
2. Upload up to five facial images.
3. Detect the face using MTCNN.
4. Generate FaceNet embeddings.
5. Compute an average embedding from all uploaded images.
6. Store the embedding and user information in SQLite.

---

### Real-Time Face Recognition

1. Start the webcam.
2. Capture live video frames.
3. Detect the face using MTCNN.
4. Generate a FaceNet embedding.
5. Compare the embedding with enrolled embeddings.
6. Display the recognized person's name and confidence score.

---

## Installation

### Clone the Repository

```bash
git clone https://github.com/yourusername/QuickFace-AI.git

cd QuickFace-AI
```

---

### Create a Virtual Environment

#### Windows

```bash
python -m venv venv

venv\Scripts\activate
```

#### macOS / Linux

```bash
python3 -m venv venv

source venv/bin/activate
```

---

### Install Dependencies

```bash
pip install -r requirements.txt
```

---

## Running the Backend

```bash
cd backend

uvicorn app:app --reload
```

The backend will be available at:

```
http://127.0.0.1:8000
```

Interactive API documentation:

```
http://127.0.0.1:8000/docs
```

---

## Running the Frontend

Open the `frontend` folder using **Live Server** (VS Code) or any local web server.

Example:

```
http://127.0.0.1:5500
```

---

## API Endpoints

| Method | Endpoint | Description |
|---------|----------|-------------|
| POST | `/enroll` | Enroll a new person |
| POST | `/recognize` | Recognize a face |
| GET | `/persons` | Retrieve all enrolled persons |
| DELETE | `/person/{id}` | Delete an enrolled person |
| GET | `/health` | Check backend health |

---

## Recognition Pipeline

```
Input Image
      │
      ▼
Face Detection (MTCNN)
      │
      ▼
Face Cropping
      │
      ▼
FaceNet Embedding Generation
      │
      ▼
Cosine Similarity Comparison
      │
      ▼
Known Person / Unknown Person
```

---

## Current Capabilities

- Multi-image enrollment (up to five images per person)
- Face embedding generation using FaceNet
- Real-time webcam recognition
- Confidence score calculation
- Person management dashboard
- SQLite database integration
- RESTful backend API

---

## Future Enhancements

- Store multiple embeddings per person instead of an averaged embedding
- Face tracking for smoother real-time recognition
- Attendance logging
- PostgreSQL support
- JWT-based authentication
- Docker deployment
- Recognition history and analytics
- Role-based access control
- Cloud deployment
- Model optimization for faster inference

---

## License

This project is intended for educational and learning purposes.
