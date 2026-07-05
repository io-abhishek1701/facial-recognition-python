from fastapi import FastAPI, UploadFile, File, Form, Depends
from sqlalchemy.orm import Session

from database import Base, engine, get_db
from models import Person
from schemas import PersonResponse
from face_service import detect_face, generate_embedding

from recognition import compare_embeddings, is_match
import numpy as np

import cv2
import json
import os

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

@app.get("/persons", response_model=list[PersonResponse])
def get_all_persons(db: Session = Depends(get_db)):
    return db.query(Person).all()

@app.post("/enroll")
async def enroll_person(
    name: str = Form(...),
    image: UploadFile = File(...),
    db: Session = Depends(get_db)
):

    image_path = os.path.join("enrolled_faces", image.filename)

    with open(image_path, "wb") as buffer:
        buffer.write(await image.read())

    img = cv2.imread(image_path)

    if img is None:
        return {"success": False, "message": "Invalid image"}

    face = detect_face(img)

    if face is None:
        return {"success": False, "message": "No face detected"}

    embedding = generate_embedding(face)

    embedding_json = json.dumps(embedding.tolist())

    person = Person(
        name=name,
        embedding=embedding_json,
        image_path=image_path
    )

    db.add(person)
    db.commit()
    db.refresh(person)

    return {
        "success": True,
        "id": person.id,
        "name": person.name
    }

@app.post("/recognize")
async def recognize_person(
    image: UploadFile = File(...),
    db: Session = Depends(get_db)
):

    image_path = os.path.join("enrolled_faces", "temp_" + image.filename)

    with open(image_path, "wb") as buffer:
        buffer.write(await image.read())

    img = cv2.imread(image_path)

    if img is None:
        return {
            "success": False,
            "message": "Invalid image"
        }

    face = detect_face(img)

    if face is None:
        return {
            "success": False,
            "message": "No face detected"
        }

    new_embedding = generate_embedding(face)

    persons = db.query(Person).all()

    best_match = None
    highest_score = 0

    for person in persons:

        stored_embedding = np.array(json.loads(person.embedding))

        score = compare_embeddings(
            new_embedding,
            stored_embedding
        )

        if score > highest_score:
            highest_score = score
            best_match = person

    os.remove(image_path)

    if best_match and is_match(highest_score):

        return {
            "success": True,
            "name": best_match.name,
            "confidence": round(highest_score * 100, 2)
        }

    return {
        "success": False,
        "name": "Unknown",
        "confidence": round(highest_score * 100, 2)
    }