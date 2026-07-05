from sqlalchemy import Column, Integer, String, DateTime
from database import Base
from datetime import datetime


class Person(Base):
    __tablename__ = "persons"

    id = Column(Integer, primary_key=True, index=True)

    name = Column(String, nullable=False)

    embedding = Column(String, nullable=False)

    image_path = Column(String, nullable=False)

    created_at = Column(DateTime, default=datetime.utcnow)