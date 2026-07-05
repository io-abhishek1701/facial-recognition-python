import cv2
import numpy as np
from mtcnn import MTCNN
from keras_facenet import FaceNet

# Load detector and FaceNet model once when the application starts
detector = MTCNN()
embedder = FaceNet()


def detect_face(image):
    """
    Detects the largest face in an image and returns the cropped face.
    """

    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    faces = detector.detect_faces(rgb)

    if len(faces) == 0:
        return None

    # Take the largest detected face
    face = max(faces, key=lambda x: x["box"][2] * x["box"][3])

    x, y, w, h = face["box"]

    x = max(0, x)
    y = max(0, y)

    cropped = rgb[y:y+h, x:x+w]

    cropped = cv2.resize(cropped, (160, 160))

    return cropped


def generate_embedding(face):
    """
    Generates a 128-D embedding using FaceNet.
    """

    embedding = embedder.embeddings([face])[0]

    return embedding