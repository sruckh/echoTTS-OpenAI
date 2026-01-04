#!/usr/bin/env python3
"""
LinaCodec Token Decoder (Tier 2 Component)

Decodes LinaCodec tokens to 48kHz PCM audio for streaming.

Usage:
  echo '{"tokens":[...],"embedding":[...]}' | python3 decode_linacodec.py > audio.pcm

Input: JSON via stdin with tokens and embedding
Output: 16-bit PCM audio to stdout (48kHz, mono, little-endian)

Environment:
  - Runs in Node.js container subprocess (or FastAPI in this case)
  - LinaCodec model cached in ~/.cache/huggingface
"""

import sys
import json
import io

# =============================================================================
# LINACODEC IMPORTS
# =============================================================================
try:
    from linacodec.codec import LinaCodec
    import torch
    import numpy as np
    LINACODEC_AVAILABLE = True
except ImportError:
    LINACODEC_AVAILABLE = False
    print("Error: LinaCodec or dependencies (torch, numpy) not installed.", file=sys.stderr)
    print("Install with: pip install git+https://github.com/ysharma3501/LinaCodec.git torch numpy", file=sys.stderr)
    sys.exit(1)

# =============================================================================
# GLOBAL STATE (CACHED ACROSS CALLS FOR PERFORMANCE)
# =============================================================================
# Note: Each subprocess invocation starts fresh, but model loads from disk cache
_decoder = None

def get_decoder():
    """Get or create LinaCodec decoder instance (cached in process)"""
    global _decoder
    if _decoder is None:
        # We don't want to print to stdout as it's used for binary audio
        # Using stderr for logging
        # print("[Tier 2][Python] Loading LinaCodec model...", file=sys.stderr)
        _decoder = LinaCodec()
        # print("[Tier 2][Python] LinaCodec loaded!", file=sys.stderr)
    return _decoder

def decode_tokens(tokens, embedding):
    """
    Decode LinaCodec tokens to audio

    Args:
        tokens: List or array of LinaCodec tokens
        embedding: Global embedding vector

    Returns:
        bytes: 16-bit PCM audio (48kHz, mono, little-endian)
    """
    decoder = get_decoder()

    if isinstance(tokens, list):
        tokens = np.array(tokens)
    if isinstance(embedding, list):
        embedding = np.array(embedding)

    # Decode tokens to audio (48kHz float32)
    # Ensure tokens are on the correct device if needed, but LinaCodec usually handles it
    audio = decoder.decode(tokens, embedding)

    # Convert float32 to int16 PCM
    audio_float = audio.cpu().numpy() if hasattr(audio, 'cpu') else audio
    audio_int16 = (audio_float * 32767).astype(np.int16)

    # Return as bytes
    return audio_int16.tobytes()

def main():
    """Main entry point for subprocess"""
    try:
        # Read input from stdin (JSON with tokens and embedding)
        input_data = sys.stdin.read()

        if not input_data.strip():
            # print("[Tier 2][Python] Error: No input data", file=sys.stderr)
            sys.exit(1)

        # Parse JSON input
        data = json.loads(input_data)
        tokens = data.get('tokens')
        embedding = data.get('embedding')

        if tokens is None:
            # print("[Tier 2][Python] Error: Missing 'tokens' in input", file=sys.stderr)
            sys.exit(1)

        if embedding is None:
            # print("[Tier 2][Python] Error: Missing 'embedding' in input", file=sys.stderr)
            sys.exit(1)

        # Decode tokens
        # print(f"[Tier 2][Python] Decoding {len(tokens)} tokens...", file=sys.stderr)
        audio_bytes = decode_tokens(tokens, embedding)
        # print(f"[Tier 2][Python] Output {len(audio_bytes)} bytes", file=sys.stderr)

        # Write to stdout (binary PCM)
        sys.stdout.buffer.write(audio_bytes)
        sys.stdout.buffer.flush()

    except json.JSONDecodeError as e:
        print(f"[Tier 2][Python] Error: Invalid JSON input: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"[Tier 2][Python] Error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc(file=sys.stderr)
        sys.exit(1)

if __name__ == '__main__':
    main()
