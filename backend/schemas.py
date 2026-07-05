from pydantic import BaseModel
from datetime import datetime


class PersonResponse(BaseModel):
    id: int
    name: str
    image_path: str
    created_at: datetime

    class Config:
        from_attributes = True