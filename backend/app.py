from fastapi import FastAPI
from database import engine, Base

from fastapi import FastAPI, UploadFile, File, Form, Depends
from sqlalchemy.orm import Session
from database import engine, Base, get_db
from models import Person
from face_service import detect_face, generate_embedding

import os
import cv2
import json

# Create all database tables
Base.metadata.create_all(bind=engine)

# Create FastAPI application
app = FastAPI(
    title="QuickFace AI",
    description="Face Recognition System using FaceNet",
    version="1.0.0"
)

# Create folder for storing enrolled face images
os.makedirs("enrolled_faces", exist_ok=True)

@app.post("/enroll")
async def enroll_person(
    name: str = Form(...),
    image: UploadFile = File(...),
    db: Session = Depends(get_db)
):
    # Save uploaded image
    image_path = os.path.join("enrolled_faces", image.filename)

    with open(image_path, "wb") as f:
        f.write(await image.read())

    # Read image using OpenCV
    img = cv2.imread(image_path)

    if img is None:
        return {"error": "Invalid image"}

    # Detect face
    face = detect_face(img)

    if face is None:
        return {"error": "No face detected"}

    # Generate embedding
    embedding = generate_embedding(face)

    # Convert embedding to JSON string
    embedding_json = json.dumps(embedding.tolist())

    # Save to database
    person = Person(
        name=name,
        embedding=embedding_json,
        image_path=image_path
    )

    db.add(person)
    db.commit()
    db.refresh(person)

    return {
        "message": "Person enrolled successfully",
        "id": person.id,
        "name": person.name
    }