"""Spoken replies for messaging channels, encoded as OGG/Opus voice notes.

Telegram (and most messengers) only show audio as a playable voice note when
it is OGG/Opus, so synthesized WAV is re-encoded with PyAV before sending.
"""

from __future__ import annotations

import io
import logging
import re
import threading
from typing import TYPE_CHECKING, Any, Callable, Optional

if TYPE_CHECKING:
    from openjarvis.core.config import JarvisConfig

logger = logging.getLogger(__name__)

# Long replies take a while to synthesize on CPU and nobody listens to a
# five-minute voice note; the full text is always sent alongside.
_MAX_SPOKEN_CHARS = 1200
_OPUS_RATE = 48000

_FENCED_CODE = re.compile(r"```.*?```", re.DOTALL)
_MD_LINK = re.compile(r"\[([^\]]+)\]\([^)]+\)")
_URL = re.compile(r"https?://\S+")
_MD_MARKUP = re.compile(r"[*_`#>|~]+")
_BULLET = re.compile(r"^\s*(?:[-+]|\d+[.)])\s+", re.MULTILINE)
# Emoji and other symbols outside the Basic Multilingual Plane, which
# speech engines either skip or read out by name.
_ASTRAL = re.compile(r"[\U00010000-\U0010FFFF☀-➿️]")


def speakable_text(text: str, max_chars: int = _MAX_SPOKEN_CHARS) -> str:
    """Strip Markdown, links and emoji, and trim to a sentence boundary."""
    text = _FENCED_CODE.sub(" ", text)
    text = _MD_LINK.sub(r"\1", text)
    text = _URL.sub(" ", text)
    text = _BULLET.sub("", text)
    text = _MD_MARKUP.sub("", text)
    text = _ASTRAL.sub("", text)
    # Line breaks (list items, headings) become sentence breaks so the
    # speech engine pauses instead of running items together.
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    text = " ".join(line if line[-1] in ".!?:;," else f"{line}." for line in lines)
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) <= max_chars:
        return text
    cut = text[:max_chars]
    end = max(cut.rfind(". "), cut.rfind("! "), cut.rfind("? "))
    return cut[: end + 1] if end > max_chars // 2 else cut.rsplit(" ", 1)[0] + "..."


def wav_to_ogg_opus(wav: bytes, *, rate_scale: float = 1.0) -> bytes:
    """Re-encode WAV audio as a mono OGG/Opus voice note.

    ``rate_scale`` reinterprets the sample rate before resampling, which
    shifts pitch the same way ``play_wav(rate_scale=...)`` does locally.
    """
    import av
    import numpy as np
    import soundfile as sf

    data, sample_rate = sf.read(io.BytesIO(wav), dtype="float32")
    if data.ndim > 1:
        data = data.mean(axis=1)
    data = np.ascontiguousarray(data, dtype=np.float32)

    frame = av.AudioFrame.from_ndarray(data[None, :], format="flt", layout="mono")
    frame.rate = max(1, int(round(sample_rate * rate_scale)))
    resampler = av.AudioResampler(format="s16", layout="mono", rate=_OPUS_RATE)

    buf = io.BytesIO()
    with av.open(buf, "w", format="ogg") as container:
        stream = container.add_stream("libopus", rate=_OPUS_RATE)
        stream.layout = "mono"
        # Opus consumes fixed 20 ms frames; a FIFO re-chunks the resampled
        # audio so arbitrarily long input encodes correctly.
        fifo = av.AudioFifo()
        frame_size = stream.codec_context.frame_size or _OPUS_RATE // 50
        for resampled in [*resampler.resample(frame), *resampler.resample(None)]:
            fifo.write(resampled)
            while fifo.samples >= frame_size:
                for packet in stream.encode(fifo.read(frame_size)):
                    container.mux(packet)
        tail = fifo.read()
        if tail is not None:
            for packet in stream.encode(tail):
                container.mux(packet)
        for packet in stream.encode(None):
            container.mux(packet)
    return buf.getvalue()


def make_voice_note_synthesizer(
    config: "JarvisConfig",
) -> Callable[[str], Optional[bytes]]:
    """Build ``text -> OGG/Opus bytes`` using the configured voice settings.

    The TTS backend is resolved on first use and reused; the callable
    returns None when no backend is available or the text has nothing
    speakable.
    """
    from openjarvis.speech._tts_discovery import (
        default_voice_for,
        get_tts_backend,
        voice_pitch,
        voice_preferences,
    )

    preferred, voice_id, speed = voice_preferences(config)
    pitch = voice_pitch(config)
    attempted: set[str] = set()
    state: dict[str, Any] = {"backend": None}
    lock = threading.Lock()

    def synthesize(text: str) -> Optional[bytes]:
        spoken = speakable_text(text)
        if not spoken:
            return None
        with lock:
            while True:
                backend = state["backend"] or get_tts_backend(
                    preferred, attempted=attempted
                )
                if backend is None:
                    return None
                state["backend"] = backend
                active = getattr(backend, "backend_id", "") or ""
                kwargs: dict[str, Any] = {
                    "output_format": "wav",
                    "speed": speed / pitch,
                }
                # Voice IDs are backend-specific; see _tts_discovery.
                vid = voice_id if active == preferred else default_voice_for(active)
                if vid:
                    kwargs["voice_id"] = vid
                try:
                    result = backend.synthesize(spoken, **kwargs)
                except Exception:
                    logger.warning(
                        "TTS backend %r failed; trying the next one",
                        active,
                        exc_info=True,
                    )
                    state["backend"] = None
                    continue
                if not result.audio:
                    return None
                return wav_to_ogg_opus(result.audio, rate_scale=pitch)

    return synthesize


__all__ = ["make_voice_note_synthesizer", "speakable_text", "wav_to_ogg_opus"]
