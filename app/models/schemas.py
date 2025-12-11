from typing import Optional, Literal
from pydantic import BaseModel, Field, field_validator

class SpeechRequest(BaseModel):
    model: str = Field(..., description="The ID of the model to use: 'tts-1' or 'tts-1-hd'")
    input: str = Field(..., description="The text to generate audio for", max_length=4096)
    voice: str = Field(..., description="The voice to use for the audio generation")
    response_format: Optional[str] = Field(
        "mp3", 
        description="The format to audio in. Supported: mp3, opus, aac, flac, wav"
    )
    speed: Optional[float] = Field(
        1.0, 
        ge=0.25, 
        le=4.0, 
        description="The speed of the generated audio. Select a value from 0.25 to 4.0. 1.0 is the default."
    )

    @field_validator('response_format')
    @classmethod
    def validate_format(cls, v: str) -> str:
        v = v.lower()
        allowed = ["mp3", "opus", "aac", "flac", "wav"]
        if v not in allowed:
            raise ValueError(f"response_format must be one of {allowed}")
        return v

class ErrorMessage(BaseModel):
    message: str
    type: str = "invalid_request_error"
    param: Optional[str] = None
    code: Optional[str] = None

class ErrorResponse(BaseModel):
    error: ErrorMessage
