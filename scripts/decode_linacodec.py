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
import os

# =============================================================================
# LINACODEC IMPORTS
# =============================================================================
try:
    from linacodec.codec import LinaCodec
    from linacodec.model import LinaCodecModel
    from linacodec.module.distill_wavlm import wav2vec2_model
    from linacodec.vocoder.vocos import Vocos
    from linacodec.util import vocode
    from huggingface_hub import snapshot_download
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

class SafeLinaCodec(LinaCodec):
    """
    A wrapper around LinaCodec that safely handles CPU-only environments.
    The original LinaCodec hardcodes .cuda() in __init__ and load_distilled_wavlm.
    """
    def __init__(self):
        # Determine device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Download model (same as original)
        model_path = snapshot_download("YatharthS/LinaCodec")
        
        # Load model with correct device placement
        self.model = LinaCodecModel.from_pretrained(
            config_path=f"{model_path}/config.yaml", 
            weights_path=f'{model_path}/model.safetensors'
        ).eval().to(self.device)
        
        # Manual implementation of load_distilled_wavlm to avoid .cuda()
        # Original: self.model.load_distilled_wavlm(f"{model_path}/wavlm_encoder.pth")
        ckpt = torch.load(f"{model_path}/wavlm_encoder.pth", map_location=self.device)
        w_model = wav2vec2_model(**ckpt["config"])
        w_model.load_state_dict(ckpt["state_dict"], strict=False)
        self.model.wavlm_model = w_model.to(self.device)
        self.model.distilled_layers = [6, 8]

        # Load vocoder
        self.vocos = Vocos.from_hparams(f'{model_path}/vocoder/config.yaml').to(self.device)
        self.vocos.load_state_dict(torch.load(f'{model_path}/vocoder/pytorch_model.bin', map_location=self.device))

    def decode(self, content_tokens, global_embedding):
        """decodes tokens and embedding into 48khz waveform"""
        if not isinstance(content_tokens, torch.Tensor):
            content_tokens = torch.tensor(content_tokens)
        if not isinstance(global_embedding, torch.Tensor):
            global_embedding = torch.tensor(global_embedding)

        content_tokens = content_tokens.to(self.device)
        global_embedding = global_embedding.to(self.device)

        ## decode tokens and embedding to mel spectrogram
        with torch.no_grad():
            # Use autocast only if on CUDA
            if self.device.type == 'cuda':
                with torch.autocast(device_type='cuda', dtype=torch.float16):
                    mel_spectrogram = self.model.decode(content_token_indices=content_tokens, global_embedding=global_embedding)
                    waveform = vocode(self.vocos, mel_spectrogram.unsqueeze(0))
            else:
                mel_spectrogram = self.model.decode(content_token_indices=content_tokens, global_embedding=global_embedding)
                waveform = vocode(self.vocos, mel_spectrogram.unsqueeze(0))
                
        return waveform

def get_decoder():
    """Get or create LinaCodec decoder instance (cached in process)"""
    global _decoder
    if _decoder is None:
        _decoder = SafeLinaCodec()
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
            sys.exit(1)

        # Parse JSON input
        data = json.loads(input_data)
        tokens = data.get('tokens')
        embedding = data.get('embedding')

        if tokens is None:
            sys.exit(1)

        if embedding is None:
            sys.exit(1)

        # Decode tokens
        audio_bytes = decode_tokens(tokens, embedding)

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