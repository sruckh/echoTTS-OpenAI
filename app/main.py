import logging
from fastapi import FastAPI, Header, HTTPException, Depends, status
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware

from app.config import settings
from app.models.schemas import SpeechRequest, ErrorResponse
from app.audio_processor import runpod_client

# Setup Logging
logging.basicConfig(
    level=settings.LOG_LEVEL,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("echo-tts")

app = FastAPI(title="Echo TTS OpenAI Bridge")

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

async def verify_token(authorization: str = Header(None)):
    if not settings.REQUIRE_AUTH:
        return
    
    if not authorization:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing Authorization Header"
        )
    
    scheme, _, token = authorization.partition(" ")
    if scheme.lower() != "bearer":
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid Authentication Scheme"
        )
    
    if not settings.BRIDGE_TOKEN or token != settings.BRIDGE_TOKEN.get_secret_value():
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid Token"
        )

@app.get("/health")
async def health_check():
    return {"status": "ok"}

@app.post(
    "/v1/audio/speech", 
    response_class=StreamingResponse,
    responses={
        400: {"model": ErrorResponse},
        401: {"model": ErrorResponse},
        500: {"model": ErrorResponse}
    }
)
async def create_speech(
    request: SpeechRequest, 
    _auth: None = Depends(verify_token)
):
    # Validate voice mapping (fail fast for unmapped voices)
    if request.voice.lower() not in runpod_client.voice_map:
        return JSONResponse(
            status_code=400,
            content={"error": {"message": f"Unsupported voice '{request.voice}'.", "type": "invalid_request_error"}},
        )

    if not request.input or not request.input.strip():
        return JSONResponse(
            status_code=400,
            content={"error": {"message": "Input text is empty.", "type": "invalid_request_error"}},
        )

    logger.info(f"Processing request: voice={request.voice}, fmt={request.response_format}")

    async def audio_generator():
        try:
            audio_bytes = await runpod_client.process_text(
                request.input, request.voice, request.speed
            )
            yield audio_bytes
        except Exception as e:
            logger.error(f"Error processing request: {e}")
            return

    media_type = f"audio/{request.response_format}"
    # WAV special case? audio/wav
    if request.response_format == "mp3":
        media_type = "audio/mpeg"
    
    filename = f"speech.{request.response_format}"
    
    return StreamingResponse(
        audio_generator(),
        media_type=media_type,
        headers={"Content-Disposition": f'attachment; filename="{filename}"'}
    )
