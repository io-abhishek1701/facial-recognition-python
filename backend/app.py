from pathlib import Path
from uuid import uuid4
import json

import os
import cv2
import numpy as np
from fastapi import Depends, FastAPI, File, Form, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session

from database import Base, engine, get_db
from face_service import detect_face, generate_embedding
from models import Person
from recognition import compare_embeddings, is_match


BASE_DIR = Path(__file__).resolve().parent
ENROLLED_FACES_DIR = BASE_DIR / "enrolled_faces"
ENROLLED_FACES_DIR.mkdir(exist_ok=True)

# Create all database tables
Base.metadata.create_all(bind=engine)

# Create FastAPI application
app = FastAPI(
    title="QuickFace AI",
    description="Face Recognition System using FaceNet",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


async def save_uploaded_image(image: UploadFile, prefix: str = "") -> Path:
    suffix = Path(image.filename or "").suffix or ".jpg"
    image_path = ENROLLED_FACES_DIR / f"{prefix}{uuid4().hex}{suffix}"
    image_path.write_bytes(await image.read())
    return image_path


def stored_image_path(image_path: Path) -> str:
    return str(image_path.relative_to(BASE_DIR))


def remove_file_if_exists(image_path: Path) -> None:
    try:
        image_path.unlink(missing_ok=True)
    except OSError:
        pass


@app.post("/enroll")
async def enroll_person(
    name: str = Form(...),
    images: list[UploadFile] = File(...),
    db: Session = Depends(get_db)
):

    if len(images) == 0:
        return {
            "success": False,
            "message": "Please upload at least one image."
        }

    if len(images) > 5:
        return {
            "success": False,
            "message": "Maximum 5 images allowed."
        }

    embeddings = []
    first_image_path = None

    for index, image in enumerate(images, start=1):
        person_folder = os.path.join(
        "enrolled_faces",
        name
        )

        os.makedirs(
            person_folder,
            exist_ok=True
        )

        extension = os.path.splitext(image.filename)[1]
        image_path = os.path.join(
        person_folder,
        f"image_{index}{extension}"
        )

        with open(image_path, "wb") as buffer:
            buffer.write(await image.read())

        if first_image_path is None:
            first_image_path = image_path

        img = cv2.imread(image_path)

        if img is None:
            continue

        face = detect_face(img)

        if face is None:
            continue

        embedding = generate_embedding(face)

        embeddings.append(embedding)

    if len(embeddings) == 0:
        return {
            "success": False,
            "message": "No valid face detected in uploaded images."
        }

    average_embedding = np.mean(
        embeddings,
        axis=0
    )

    embedding_json = json.dumps(
        average_embedding.tolist()
    )

    person = Person(
        name=name,
        embedding=embedding_json,
        image_path=first_image_path
    )

    db.add(person)
    db.commit()
    db.refresh(person)

    return {
    "success": True,
    "message": f"{person.name} enrolled successfully using {len(embeddings)} image(s).",
    "id": person.id,
    "name": person.name,
    "images_processed": len(embeddings)
}

@app.post("/recognize")
async def recognize_person(
    image: UploadFile = File(...),
    db: Session = Depends(get_db),
):
    image_path = await save_uploaded_image(image, prefix="temp_")

    try:
        img = cv2.imread(str(image_path))

        if img is None:
            return {
                "success": False,
                "message": "Invalid image",
                "name": "Unknown",
                "confidence": 0,
                "box": None,
            }

        face, box = detect_face(
            img,
            return_box=True
        )

        if face is None:
            return {
                "success": False,
                "message": "No face detected",
                "name": "Unknown",
                "confidence": 0,
                "box": None,
            }

        persons = db.query(Person).all()

        if not persons:
            return {
                "success": False,
                "message": "No enrolled persons found",
                "name": "Unknown",
                "confidence": 0,
                "box": box,
            }

        new_embedding = generate_embedding(face)

        best_match = None
        highest_score = 0

        for person in persons:

            stored_embedding = np.array(
                json.loads(person.embedding)
            )

            score = compare_embeddings(
                new_embedding,
                stored_embedding
            )

            if score > highest_score:
                highest_score = score
                best_match = person

        confidence = round(highest_score * 100, 2)

        if best_match and is_match(highest_score):

            return {
                "success": True,
                "message": "Match found",
                "name": best_match.name,
                "confidence": confidence,
                "box": box,
            }

        return {
            "success": False,
            "message": "No matching person found",
            "name": "Unknown",
            "confidence": confidence,
            "box": box,
        }

    finally:
        remove_file_if_exists(image_path)


@app.get("/persons")
def get_persons(db: Session = Depends(get_db)):
    persons = db.query(Person).all()

    result = []

    for person in persons:
        result.append(
            {
                "id": person.id,
                "name": person.name,
                "image_path": person.image_path,
                "created_at": person.created_at,
            }
        )

    return result


@app.delete("/person/{person_id}")
def delete_person(
    person_id: int,
    db: Session = Depends(get_db),
):
    person = db.query(Person).filter(Person.id == person_id).first()

    if person is None:
        return {
            "success": False,
            "message": "Person not found",
        }

    saved_path = Path(person.image_path)
    if not saved_path.is_absolute():
        saved_path = BASE_DIR / saved_path

    db.delete(person)
    db.commit()
    remove_file_if_exists(saved_path)

    return {
        "success": True,
        "message": "Person deleted successfully",
    }


@app.get("/health")
def health():
    return {
        "status": "online",
        "database": "connected",
        "model": "FaceNet",
        "detector": "MTCNN",
    }
