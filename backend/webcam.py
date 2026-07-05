import cv2
import json
import numpy as np

from database import SessionLocal
from models import Person
from face_service import detect_face, generate_embedding
from recognition import compare_embeddings, is_match


def start_webcam():

    db = SessionLocal()

    cap = cv2.VideoCapture(0)

    while True:

        ret, frame = cap.read()

        if not ret:
            break

        face = detect_face(frame)

        label = "Unknown"

        if face is not None:

            embedding = generate_embedding(face)

            persons = db.query(Person).all()

            highest_score = 0

            best_person = None

            for person in persons:

                stored_embedding = np.array(
                    json.loads(person.embedding)
                )

                score = compare_embeddings(
                    embedding,
                    stored_embedding
                )

                if score > highest_score:
                    highest_score = score
                    best_person = person

            if best_person and is_match(highest_score):
                label = f"{best_person.name} ({highest_score:.2f})"

        cv2.putText(
            frame,
            label,
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2
        )

        cv2.imshow("QuickFace AI", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    db.close()


if __name__ == "__main__":
    start_webcam()