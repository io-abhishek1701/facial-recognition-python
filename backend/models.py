from sqlalchemy import (
    Column,
    Integer,
    String,
    DateTime,
    Text,
    ForeignKey
)
from datetime import datetime

from database import Base
class Person(Base):
    __tablename__ = "persons"

    id = Column(Integer, primary_key=True, index=True)

    name = Column(String, nullable=False)

    embedding = Column(String, nullable=False)

    image_path = Column(String, nullable=False)

    created_at = Column(DateTime, default=datetime.utcnow)

class FaceImage(Base):
    __tablename__ = "face_images"

    id = Column(Integer, primary_key=True)

    person_id = Column(Integer, ForeignKey("persons.id"))

    embedding = Column(Text)

    image_path = Column(String)