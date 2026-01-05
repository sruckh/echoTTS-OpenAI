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
    return {"status": "ok", "streaming_enabled": settings.ENABLE_STREAMING}

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

    # Determine Media Type
    media_type = f"audio/{request.response_format}"
    if request.response_format == "mp3":
        media_type = "audio/mpeg"
    elif request.response_format == "pcm":
        media_type = "audio/pcm" # Custom

    # Streaming Mode
    if request.stream and settings.ENABLE_STREAMING:
        logger.info(f"Processing streaming request: voice={request.voice}, fmt={request.response_format}")
        
        # 1. Get raw PCM stream (48kHz)
        pcm_stream = runpod_client.stream_speech(request.input, request.voice, request.speed)
        
        # 2. Transcode if needed
        if request.response_format != "pcm":
            output_stream = runpod_client.transcode_stream(pcm_stream, request.response_format)
        else:
            output_stream = pcm_stream

        return StreamingResponse(
            output_stream,
            media_type=media_type,
            headers={
                "Cache-Control": "no-cache, no-transform",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no"
            }
        )

    # Batch Mode
    logger.info(f"Processing batch request: voice={request.voice}, fmt={request.response_format}")

    async def audio_generator():
        try:
            # Get raw PCM (or audio) from RunPod
            audio_bytes = await runpod_client.process_text(
                request.input, request.voice, request.speed
            )
            
            # If format is not PCM, we need to transcode the single batch
            if request.response_format != "pcm":
                # Helper to turn bytes into a generator for transcode_stream
                async def bytes_gen():
                    yield audio_bytes
                
                async for chunk in runpod_client.transcode_stream(bytes_gen(), request.response_format):
                    yield chunk
            else:
                yield audio_bytes
        except Exception as e:
            logger.error(f"Error processing batch request: {e}")
            return
    
    filename = f"speech.{request.response_format}"
    
    return StreamingResponse(
        audio_generator(),
        media_type=media_type,
        headers={"Content-Disposition": f'attachment; filename="{filename}"'}
    )

@app.post("/api/tts/stream")
async def stream_tts(
    request: SpeechRequest,
    _auth: None = Depends(verify_token)
):
    """
    Tier 2 streaming endpoint for EchoTTS service.
    Outputs 48kHz PCM audio via SSE-like binary stream.
    """
    if not settings.ENABLE_STREAMING:
        return JSONResponse(
            status_code=503,
            content={"error": "Streaming is disabled", "message": "Set ENABLE_STREAMING=true to enable"}
        )

    # Validate voice mapping
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

    logger.info(f"Processing streaming request: voice={request.voice}")

    return StreamingResponse(
        runpod_client.stream_speech(request.input, request.voice, request.speed),
        media_type="audio/octet-stream",
        headers={
            "Cache-Control": "no-cache, no-transform",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"
        }
    )
