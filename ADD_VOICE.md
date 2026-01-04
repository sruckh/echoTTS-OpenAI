Dynamic Voices: Required Changes for `echoTTS-OpenAI` (Bridge Component)

This repository is the OpenAI-compatible TTS *bridge* that forwards `/v1/audio/speech` requests to a RunPod serverless Echo-TTS worker. It currently supports voice selection via a static mapping loaded at process start (`VOICE_MAP` or `VOICE_MAP_FILE`).

This document defines what must change in *this component only* to support dynamically adding new voices at runtime (no container rebuild/restart), and the contract between this bridge and the other system components.

---

## 1) Current Behavior (Baseline)

- The bridge maps `request.voice` → `speaker_voice` filename (sent to RunPod).
- The mapping is loaded once at startup in `app/audio_processor.py` from:
  - `VOICE_MAP_FILE` (JSON file), else
  - `VOICE_MAP` (env, JSON string or comma-separated pairs).
- `/v1/audio/speech` rejects voices not present in the loaded map (`app/main.py` validates).
- `docker-compose.yml` already bind-mounts `./voice_map.json` → `/app/voice_map.json`, but edits require a restart today because the mapping is only loaded at startup.

---

## 2) Target Behavior (Dynamic Voices)

When a new voice is added by the “voice management” component (separate project), the bridge must:

- Accept the new voice without restart.
- Expose a machine-readable voice list for the UI/admin component.
- Keep the OpenAI-compatible behavior unchanged for `/v1/audio/speech`.
- Continue running entirely inside containers (nothing installed on the host/server).

---

## 3) Cross-Component Contract (What the Other Components Must Provide)

The bridge only forwards a filename to the RunPod worker. Therefore:

1) RunPod worker contract
- For every voice entry, the referenced `speaker_voice` filename must exist where the worker expects it (inside the worker image or on a volume mounted into the worker).
- Filenames are treated as opaque strings by the bridge; case sensitivity depends on the worker filesystem.

2) Voice management component contract
- Owns the “source of truth” voice registry (and any approval workflow).
- Writes the registry into a shared location the bridge can read (recommended: a shared Docker volume).
- Updates must be atomic (write temp file, then rename) to avoid the bridge reading partial JSON.

---

## 4) Voice Registry File (New Canonical Input for This Bridge)

### File location

- Use `VOICE_MAP_FILE` as the canonical registry file path.
- Recommended default (already used in compose): `VOICE_MAP_FILE=/app/voice_map.json`

### Supported formats

To keep backward compatibility while enabling richer metadata later, the bridge should support both:

Format A (current / legacy): simple map
```json
{
  "dorota": "Dorota.ogg",
  "scott": "Scott.ogg"
}
```

Format B (proposed): registry with metadata
```json
{
  "version": 1,
  "updated_at": "2025-12-15T00:00:00Z",
  "voices": [
    {
      "id": "dorota",
      "display_name": "Dorota",
      "speaker_file": "Dorota.ogg",
      "enabled": true
    }
  ]
}
```

### Normalization rules (bridge)

- `id` is case-insensitive; treat `request.voice` as `lower()` for lookups.
- Only `enabled: true` voices are accepted for TTS (Format B).
- Validation errors in the registry file should:
  - keep serving the last known-good mapping, and
  - emit a clear log line with the parse/validation error.

---

## 5) Bridge Changes Required (echoTTS-OpenAI)

### 5.1 Dynamic reload (no restart)

Replace “load once at startup” with one of these strategies:

- Strategy 1 (recommended): mtime-based reload
  - On each request (or at most once per `VOICE_MAP_RELOAD_INTERVAL_SECONDS`), check the registry file mtime; reload only if changed.
  - Keep an in-memory cache of:
    - last successful mapping
    - last loaded mtime
    - last load timestamp/error

- Strategy 2: explicit reload endpoint
  - Voice management component updates the file and then calls a reload endpoint on this bridge.

It’s fine to implement both: mtime-based reload for resilience + explicit reload for “instant” UI updates.

### 5.2 Voice listing endpoint (for UI/admin component)

OpenAI’s TTS API does not define a standard “list voices” endpoint, so this bridge must add a non-OpenAI endpoint used only by your other components:

- `GET /voices`
  - Returns an array of available voice IDs (and optionally display names when using Format B).
  - Should be served from the in-memory cache.

Example response:
```json
{
  "voices": [
    { "id": "dorota", "display_name": "Dorota" },
    { "id": "scott", "display_name": "Scott" }
  ],
  "source": "file:/app/voice_map.json",
  "updated_at": "2025-12-15T00:00:00Z"
}
```

### 5.3 Reload endpoint (optional but useful)

- `POST /voices/reload`
  - Forces a reload from `VOICE_MAP_FILE`.
  - Returns the loaded voice count and any warnings.
  - Protect it using the existing auth mechanism (`REQUIRE_AUTH` + `BRIDGE_TOKEN`) or a dedicated admin token.

### 5.4 TTS path changes (keep compatibility)

- `/v1/audio/speech` continues to accept any string `voice` as long as it exists in the dynamically loaded registry.
- Error behavior remains the same: unknown voice returns HTTP 400 with `invalid_request_error`.
- Mapping behavior remains the same: `request.voice` → `speaker_voice` filename passed to RunPod (`speaker_voice` key in the job payload).

---

## 6) Deployment: Sharing the Registry Between Containers

### Recommended: named Docker volume shared by components

Instead of bind-mounting from the host, use a named volume shared between:
- the voice management component (writer), and
- this bridge (reader).

Conceptual example (adjust to your orchestrator):
```yaml
volumes:
  voice_registry:

services:
  echotts-openai:
    volumes:
      - voice_registry:/app
    environment:
      - VOICE_MAP_FILE=/app/voice_map.json

  voice-manager:
    volumes:
      - voice_registry:/shared
    # voice-manager writes /shared/voice_map.json atomically
```

If you keep the current bind mount (`./voice_map.json:/app/voice_map.json`), the other component must be able to write to that same host file.

---

## 7) Operational Workflow (End-to-End)

1. Place the new voice file on/for the RunPod worker (where it expects speaker files).
2. Update the registry JSON (`voice_map.json`) to add:
   - voice `id` (what clients send as `voice`), and
   - `speaker_file` filename (what the worker expects).
3. Trigger bridge reload:
   - wait for mtime/interval reload, or call `POST /voices/reload`.
4. Verify:
   - `GET /voices` shows the new voice.
   - A test call to `POST /v1/audio/speech` with `voice=<new id>` succeeds.

---

## 8) Notes / Non-Goals (For This Component)

- This bridge does not perform voice training, approval workflows, or user/role management; those belong in the other components.
- This bridge does not upload or store voice audio assets; it only maps `voice` → worker filename and forwards TTS jobs.
