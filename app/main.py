import logging
import asyncio
from fastapi import FastAPI, Header, HTTPException, Depends, status
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware

from app.config import settings
from app.models.schemas import SpeechRequest, ErrorResponse
from app.text_chunking import split_text_into_chunks
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
    # Validate Voice
    if request.voice.lower() not in runpod_client.voice_map:
        # Fallback is handled in client, but explicit validation is requested
        # However, the config allows a "map all to default" behavior if we just didn't map it.
        # Strict validation:
        if request.voice.lower() not in runpod_client.voice_map:
             # Check if we should fail or fallback.
             # Implementation doc: "Fail fast if an unmapped voice is requested."
             return JSONResponse(
                 status_code=400,
                 content={"error": {"message": f"Unsupported voice '{request.voice}'.", "type": "invalid_request_error"}}
             )

    chunks = list(split_text_into_chunks(
        request.input, 
        target_word_count=settings.MAX_WORDS_PER_CHUNK, 
        overlap_words=settings.CHUNK_OVERLAP
    ))
    
    if not chunks:
        # Handle empty text
         return JSONResponse(
             status_code=400,
             content={"error": {"message": "Input text is empty.", "type": "invalid_request_error"}}
         )

    logger.info(f"Processing request: {len(chunks)} chunks, voice={request.voice}, fmt={request.response_format}")

    async def audio_generator():
        # Create tasks for all chunks
        # We need to yield them in order.
        # We can start them all (throttled by semaphore in client)
        tasks = []
        for chunk in chunks:
            tasks.append(asyncio.create_task(
                runpod_client.process_chunk(chunk, request.voice, request.speed)
            ))
        
        for i, task in enumerate(tasks):
            try:
                audio_bytes = await task
                yield audio_bytes
            except Exception as e:
                logger.error(f"Error processing chunk {i}: {e}")
                # We can't easily send a JSON error inside a stream efficiently without breaking the format.
                # But we should probably stop.
                # In HTTP/1.1 chunked, we can't change status code now.
                # We just stop the stream.
                break

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
