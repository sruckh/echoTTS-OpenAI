import asyncio
import httpx
import logging
import json
import os
import random
import torch
import numpy as np
from typing import Optional, Dict
from app.config import settings

logger = logging.getLogger(__name__)

# Attempt to import the local decoder for in-process decoding
try:
    from scripts.decode_linacodec import SafeLinaCodec
    DECODER = SafeLinaCodec()
    logger.info("Local SafeLinaCodec initialized for in-process decoding.")
except ImportError:
    DECODER = None
    logger.warning("SafeLinaCodec not found in scripts/decode_linacodec.py. Falling back to subprocess.")

class RunPodClient:
    def __init__(self):
        self.semaphore = asyncio.Semaphore(settings.MAX_CONCURRENT_REQUESTS)
        self.voice_map = self._load_voice_map()
        
        # Determine the base URL for run/status requests
        # Expected settings.RUNPOD_ENDPOINT format: https://api.runpod.ai/v2/{ID}/runsync or just /run
        # We need to split it to handle /run vs /status/{id}
        # If the user provides the full runsync URL, we should parse out the base.
        
        endpoint = settings.RUNPOD_ENDPOINT.rstrip('/')
        if endpoint.endswith('/runsync'):
            self.base_url = endpoint[:-8] # remove /runsync
            self.run_endpoint = endpoint
        elif endpoint.endswith('/run'):
            self.base_url = endpoint[:-4] # remove /run
            self.run_endpoint = endpoint
        else:
            # Assume base provided? or just append /run
            self.base_url = endpoint
            self.run_endpoint = f"{endpoint}/run"
            
        self.headers = {
            "Authorization": f"Bearer {settings.RUNPOD_API_KEY.get_secret_value()}",
            "Content-Type": "application/json"
        }

    def _load_voice_map(self) -> Dict[str, str]:
        # Priority 1: File
        if settings.VOICE_MAP_FILE and os.path.exists(settings.VOICE_MAP_FILE):
            try:
                with open(settings.VOICE_MAP_FILE, 'r') as f:
                    mapping = json.load(f)
                    # Normalize keys to lower case
                    return {k.lower(): v for k, v in mapping.items()}
            except Exception as e:
                logger.error(f"Failed to load voice map from file {settings.VOICE_MAP_FILE}: {e}")

        # Priority 2: JSON String in Env
        map_str = settings.VOICE_MAP
        if map_str and map_str.strip().startswith('{'):
            try:
                mapping = json.loads(map_str)
                return {k.lower(): v for k, v in mapping.items()}
            except Exception as e:
                logger.warning(f"Failed to parse VOICE_MAP as JSON: {e}. Falling back to comma-separated.")

        # Priority 3: Comma-separated String
        mapping = {}
        if not map_str:
            return mapping
        pairs = map_str.split(',')
        for p in pairs:
            if ':' in p:
                k, v = p.split(':', 1)
                mapping[k.strip().lower()] = v.strip()
        return mapping

    def get_voice_file(self, voice_name: str) -> str:
        # Default to the first mapped voice if not found, or error?
        # Implementation doc says: "Default: pick a single safe default voice file... request validation enforces... voice exists"
        # We'll return the mapped one. Validation should happen before calling this.
        # But for safety, fallback to the first one in the map if exists.
        v_lower = voice_name.lower()
        if v_lower in self.voice_map:
            return self.voice_map[v_lower]
        # Fallback (should be caught by validation really)
        if self.voice_map:
            return list(self.voice_map.values())[0]
        return "EARS p004 freeform.mp3" # Hard fallback

    async def process_text(self, text: str, voice: str, speed: float = 1.0) -> bytes:
        """
        Aggregate streaming chunks into a single byte buffer for batch requests.
        Uses the streaming path to ensure reliability if the backend batch path is broken.
        """
        async with self.semaphore:
            full_audio = b""
            chunk_count = 0
            async for chunk in self.stream_speech(text, voice, speed):
                full_audio += chunk
                chunk_count += 1
            
            return full_audio

    async def process_chunk(self, text: str, voice: str, speed: float = 1.0) -> bytes:
        # Backwards-compatible alias; chunking is handled upstream now.
        return await self.process_text(text, voice, speed)

    async def stream_speech(self, text: str, voice: str, speed: float = 1.0):
        """
        Stream speech by requesting tokens from RunPod and decoding locally.
        (Tier 2 Middleware Implementation)
        """
        # RunPod Serverless Streaming requires POST to /run followed by GET to /stream
        if self.run_endpoint.endswith('/runsync'):
            run_url = self.run_endpoint[:-8] + "/run"
            stream_base_url = self.run_endpoint[:-8] + "/stream"
        else:
            run_url = self.run_endpoint
            stream_base_url = self.base_url + "/stream"

        payload = {
            "input": {
                "text": text,
                "speaker_voice": self.get_voice_file(voice),
                "stream": True,
                "output_format": "linacodec_tokens",
                "parameters": {
                    "seed": random.randint(1, 65535),
                }
            }
        }
        
        async with httpx.AsyncClient(timeout=30.0) as client:
            try:
                # 1. Submit Job
                resp = await client.post(run_url, json=payload, headers=self.headers)
                resp.raise_for_status()
                job_data = resp.json()
                job_id = job_data.get("id")
                
                if not job_id:
                    raise RuntimeError(f"Failed to get job ID from RunPod: {job_data}")
                
                logger.info(f"Streaming job submitted: {job_id}")

                # 2. Poll /stream endpoint
                stream_url = f"{stream_base_url}/{job_id}"
                is_finished = False
                
                while not is_finished:
                    async with httpx.AsyncClient(timeout=30.0) as stream_client:
                        s_resp = await stream_client.get(stream_url, headers=self.headers)
                        s_resp.raise_for_status()
                        data = s_resp.json()
                        
                        status = data.get("status")
                        stream_data = data.get("stream", [])
                        
                        for item in stream_data:
                            # The item might be the token dict directly or wrapped in 'output'
                            # Based on verify script, it's wrapped in 'output'
                            output = item.get('output') if isinstance(item, dict) and 'output' in item else item
                            
                            if not isinstance(output, dict):
                                continue

                            if output.get('status') == 'complete':
                                logger.info(f"RunPod stream completed for job {job_id}")
                                is_finished = True
                                break
                            
                            tokens = output.get('tokens')
                            embedding = output.get('embedding')
                            
                            if tokens is not None and embedding is not None:
                                audio_bytes = await self._decode_tokens(tokens, embedding)
                                if audio_bytes:
                                    yield audio_bytes
                        
                        if status in ["COMPLETED", "FAILED"]:
                            if status == "FAILED":
                                logger.error(f"RunPod job {job_id} failed: {data.get('error')}")
                            is_finished = True
                        else:
                            # Small delay between polls
                            await asyncio.sleep(0.5)

            except Exception as e:
                logger.error(f"Error during streaming request: {e}")
                raise

    async def _decode_tokens(self, tokens, embedding) -> bytes:
        """
        Decodes LinaCodec tokens using the local SafeLinaCodec instance or subprocess.
        """
        try:
            # Prefer in-process decoding
            if DECODER:
                # SafeLinaCodec.decode returns a Tensor (float32)
                audio_tensor = DECODER.decode(tokens, embedding)
                
                # Convert to numpy
                audio_float = audio_tensor.cpu().numpy() if hasattr(audio_tensor, 'cpu') else audio_tensor
                
                # Convert to int16 PCM (48kHz)
                audio_int16 = (audio_float * 32767).astype(np.int16)
                
                return audio_int16.tobytes()

            # Fallback to subprocess
            process = await asyncio.create_subprocess_exec(
                "python3", "scripts/decode_linacodec.py",
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            input_data = json.dumps({"tokens": tokens, "embedding": embedding}).encode()
            stdout, stderr = await process.communicate(input=input_data)
            
            if process.returncode != 0:
                logger.error(f"LinaCodec decoder script failed with code {process.returncode}: {stderr.decode()}")
                return b""
                
            return stdout
        except Exception as e:
            logger.error(f"Failed to decode tokens: {e}")
            return b""

    async def transcode_stream(self, pcm_stream, output_format: str = "mp3"):
        """
        Transcodes a raw PCM (48kHz, 16-bit, mono) stream to the target format using ffmpeg.
        """
        if output_format == "pcm":
            async for chunk in pcm_stream:
                yield chunk
            return

        # Map format to ffmpeg codec
        codec_map = {
            "mp3": "libmp3lame",
            "opus": "libopus",
            "aac": "aac",
            "flac": "flac",
            "wav": "pcm_s16le" # wav is container, usually pcm inside
        }
        codec = codec_map.get(output_format, "libmp3lame")
        
        format_map = {
            "mp3": "mp3",
            "opus": "opus",
            "aac": "adts", # ADTS for streaming AAC
            "flac": "flac",
            "wav": "wav"
        }
        fmt = format_map.get(output_format, output_format)

        # Start ffmpeg process
        # Input: -f s16le -ar 48000 -ac 1 -i pipe:0
        process = await asyncio.create_subprocess_exec(
            "ffmpeg",
            "-f", "s16le", "-ar", "48000", "-ac", "1", "-i", "pipe:0",
            "-c:a", codec,
            "-f", fmt,
            "pipe:1",
            "-v", "quiet", # Suppress output
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )

        async def writer():
            try:
                async for chunk in pcm_stream:
                    if process.stdin:
                        process.stdin.write(chunk)
                        await process.stdin.drain()
                if process.stdin:
                    process.stdin.close()
            except Exception as e:
                logger.error(f"Error writing to ffmpeg stdin: {e}")
                if process.stdin:
                    process.stdin.close()

        async def reader():
            try:
                while True:
                    chunk = await process.stdout.read(4096)
                    if not chunk:
                        break
                    yield chunk
            except Exception as e:
                logger.error(f"Error reading from ffmpeg stdout: {e}")

        # Run writer in background
        writer_task = asyncio.create_task(writer())
        
        # Yield output from reader
        async for chunk in reader():
            yield chunk
            
        await writer_task
        await process.wait()

    async def _execute_job(self, text: str, voice: str, speed: float) -> bytes:
        # 1. Submit
        payload = {
            "input": {
                "text": text,
                "speaker_voice": self.get_voice_file(voice),
                "parameters": {
                    "num_steps": 40,
                    "cfg_scale_text": 3.0,
                    "cfg_scale_speaker": 8.0,
                    "seed": random.randint(1, 65535),
                    # Speed might need to be implemented in the RunPod worker or pre/post processing
                    # The prompt implies standard RunPod schema.
                    # If the worker doesn't support speed, we might need ffmpeg speed adjustment.
                    # For now, let's assume the worker handles it or we ignore it (MVP).
                    # Actually, we can use ffmpeg in the streaming step to adjust speed if needed, 
                    # but typically TTS engines have a speed param.
                    # Let's assume the worker does NOT strictly assume speed param in `parameters` unless we know the schema.
                    # We will ignore speed in payload for now to match the implementation doc example.
                }
            }
        }
        
        async with httpx.AsyncClient(timeout=settings.RUNPOD_CONNECT_TIMEOUT) as client:
            retries = 3
            last_exception = None
            
            for attempt in range(retries):
                try:
                    logger.info(f"Submitting job to {self.run_endpoint} (Attempt {attempt + 1}/{retries})")
                    resp = await client.post(self.run_endpoint, json=payload, headers=self.headers)
                    resp.raise_for_status()
                    data = resp.json()
                    job_id = data['id']
                    status = data['status']
                    logger.debug(f"Job submitted: {job_id}, status: {status}")
                    
                    # Check if sync return
                    if status == 'COMPLETED':
                        return await self._fetch_result(client, data['output'])
                    
                    # Poll
                    return await self._poll_job(client, job_id)
                    
                except (httpx.ConnectError, httpx.ReadTimeout, httpx.HTTPStatusError) as e:
                    logger.warning(f"RunPod submission error on attempt {attempt + 1}: {e}")
                    last_exception = e
                    if attempt < retries - 1:
                        # Exponential backoff: 2s, 4s, 8s...
                        sleep_time = 2.0 * (2 ** attempt)
                        logger.info(f"Retrying in {sleep_time} seconds...")
                        await asyncio.sleep(sleep_time)
                    else:
                         logger.error(f"All {retries} retry attempts failed for RunPod submission.")

            if last_exception:
                raise last_exception
            raise RuntimeError("Unknown error during RunPod submission retries")

    async def _poll_job(self, client: httpx.AsyncClient, job_id: str) -> bytes:
        status_url = f"{self.base_url}/status/{job_id}"
        start_time = asyncio.get_running_loop().time()
        
        delay = 1.0
        while (asyncio.get_running_loop().time() - start_time) < settings.RUNPOD_JOB_TIMEOUT_SECONDS:
            await asyncio.sleep(delay)
            # Backoff
            delay = min(delay * 1.5, 5.0)
            
            resp = await client.get(status_url, headers=self.headers)
            resp.raise_for_status()
            data = resp.json()
            status = data['status']
            
            if status == 'COMPLETED':
                return await self._fetch_result(client, data['output'])
            elif status == 'FAILED':
                logger.error(f"Job {job_id} failed: {data.get('error')}")
                raise RuntimeError(f"RunPod job failed: {data.get('error')}")
            
        raise TimeoutError(f"Job {job_id} timed out after {settings.RUNPOD_JOB_TIMEOUT_SECONDS}s")

    async def _fetch_result(self, client: httpx.AsyncClient, output: dict) -> bytes:
        # Case 1: Direct Audio (Base64) - e.g., pcm_24
        if isinstance(output, dict) and 'audio' in output:
            import base64
            try:
                return base64.b64decode(output['audio'])
            except Exception as e:
                logger.error(f"Failed to decode base64 audio: {e}")
                return b""

        # Case 2: LinaCodec Tokens - needs local decoding
        if isinstance(output, dict) and 'tokens' in output and 'embedding' in output:
            return await self._decode_tokens(output['tokens'], output['embedding'])

        # Case 3: Legacy URL (S3/Cloud output)
        url = None
        if isinstance(output, str) and output.startswith('http'):
            url = output
        elif isinstance(output, dict):
            url = output.get('audio_url') or output.get('url') or output.get('file_url')
            
        if not url:
            # Fallback scan for URL values
            if isinstance(output, dict):
                for v in output.values():
                    if isinstance(v, str) and v.startswith('http'):
                        url = v
                        break
        
        if not url:
             logger.error(f"No audio found in RunPod output: {output.keys() if isinstance(output, dict) else output}")
             raise ValueError(f"No audio URL or data found in RunPod output")

        logger.info(f"Downloading audio from {url}")
        async with httpx.AsyncClient(timeout=30.0) as dl_client:
            resp = await dl_client.get(url)
            resp.raise_for_status()
            return resp.content

runpod_client = RunPodClient()
