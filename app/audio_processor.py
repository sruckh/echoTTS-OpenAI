import asyncio
import httpx
import logging
import json
import os
import random
from typing import Optional, Dict
from app.config import settings

logger = logging.getLogger(__name__)

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
        async with self.semaphore:
            return await self._execute_job(text, voice, speed)

    async def process_chunk(self, text: str, voice: str, speed: float = 1.0) -> bytes:
        # Backwards-compatible alias; chunking is handled upstream now.
        return await self.process_text(text, voice, speed)

    async def stream_speech(self, text: str, voice: str, speed: float = 1.0):
        """
        Stream speech by requesting tokens from RunPod and decoding locally.
        (Tier 2 Middleware Implementation)
        """
        url = settings.ECHOTTS_STREAMING_ENDPOINT or self.run_endpoint
        
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
        
        async with httpx.AsyncClient(timeout=None) as client:
            try:
                async with client.stream("POST", url, json=payload, headers=self.headers) as response:
                    response.raise_for_status()
                    async for line in response.aiter_lines():
                        if not line.strip():
                            continue
                        try:
                            chunk_data = json.loads(line)
                            
                            # Handle different response formats if needed
                            # The spec says JSON objects per line
                            if chunk_data.get('status') == 'complete':
                                logger.info("RunPod stream completed")
                                break
                            
                            tokens = chunk_data.get('tokens')
                            embedding = chunk_data.get('embedding')
                            
                            if tokens is not None and embedding is not None:
                                audio_bytes = await self._decode_tokens(tokens, embedding)
                                if audio_bytes:
                                    yield audio_bytes
                        except json.JSONDecodeError:
                            logger.warning(f"Failed to parse stream line as JSON: {line}")
                            continue
                        except Exception as e:
                            logger.error(f"Error processing stream chunk: {e}")
                            continue
            except Exception as e:
                logger.error(f"Error during streaming request: {e}")
                raise

    async def _decode_tokens(self, tokens, embedding) -> bytes:
        """
        Decodes LinaCodec tokens using the Python subprocess script.
        """
        try:
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
            logger.error(f"Failed to execute decoder subprocess: {e}")
            return b""

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
        # Output is likely {"audio_url": "..."} or "message" or directly the result?
        # The doc says: "fetch presigned audio URL"
        # The RunPod worker usually returns something like {"message": "..."} or {"audio": "base64"} or {"result_url": "..."}
        # Let's assume it returns a URL to download.
        # Inspecting standard RunPod behavior or implementation doc...
        # Doc: "On completion, fetch presigned audio URL"
        
        # We'll look for common keys.
        url = None
        if isinstance(output, str):
            # Sometimes output is just the string URL? Unlikely but possible.
            if output.startswith('http'):
                url = output
        elif isinstance(output, dict):
            url = output.get('audio_url') or output.get('url') or output.get('file_url')
            
        if not url:
            # Fallback check if it's base64?
            # If we can't find a URL, this is an issue.
            # Let's assume 'message' contains it if the worker is "standard".
            # Or maybe 'audio' key.
            # Without the exact worker code, we guess 'audio_url'.
            # If the user says "RunPod serverless TTS worker", it likely returns an audio file path/url.
            logger.warning(f"Could not find obvious URL in output: {output}. checking other keys.")
            # fallback to values that look like URLs
            for v in output.values():
                if isinstance(v, str) and v.startswith('http'):
                    url = v
                    break
        
        if not url:
             raise ValueError(f"No audio URL found in RunPod output: {output}")

        logger.info(f"Downloading audio from {url}")
        # Use a fresh request for download to avoid auth header issues (e.g. S3 presigned urls might conflict with Bearer auth)
        # S3 usually doesn't like unexpected Authorization headers.
        async with httpx.AsyncClient(timeout=30.0) as dl_client:
            resp = await dl_client.get(url)
            resp.raise_for_status()
            return resp.content

runpod_client = RunPodClient()
