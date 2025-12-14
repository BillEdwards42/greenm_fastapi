from pydantic import BaseModel, Field, field_validator, ConfigDict
import re
from datetime import datetime
from typing import Optional

class UserBase(BaseModel):
    """
    Validates the data payload stored in firestore.
    """
    alert_time: str = Field(..., description = "Target notification time in HH:MM format (24 hour)")
    fcm_token: Optional[str] = Field(None, description = "Firebase Cloud Messaging token for push notifications")
    is_active: bool = Field(True)

    @field_validator("alart_time")
    @classmethod
    def validate_time_format(cls, v: str) -> str:
        if not re.match(r'^(?:[01]\d|2[0-3]):[0-5]\d$', v):
            raise ValueError("Time must be in HH:MM format")
        return v
    
class UserUpdate(UserBase):
    pass

class UserInDB(UserBase):
    """
    Represents a user object as stored in the database, including internal fields.
    """
    uid: str
    created_at: datetime
    updated_at: datetime
    last_active_at: datetime

    model_config = ConfigDict(from_attributes=True)
