"""Spoken channel replies: text cleanup, OGG/Opus encoding, synthesizer."""

from __future__ import annotations

import io
import math
import wave
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from openjarvis.core.config import JarvisConfig
from openjarvis.speech.voice_note import make_voice_note_synthesizer, speakable_text


def test_speakable_text_strips_markup_links_and_emoji():
    text = (
        "**NVDA** is up 2% today 🚀\n"
        "- First point\n"
        "- See [the chart](https://example.com/c)\n"
        "```python\nprint('x')\n```\n"
        "Source: https://finance.yahoo.com"
    )
    assert speakable_text(text) == (
        "NVDA is up 2% today. First point. See the chart. Source:"
    )


def test_speakable_text_trims_long_replies_at_a_sentence():
    text = "Short sentence. " * 200
    spoken = speakable_text(text, max_chars=100)
    assert len(spoken) <= 100
    assert spoken.endswith(".")


def _wav(seconds=1.0, rate=24000):
    t = [math.sin(2 * math.pi * 300 * i / rate) for i in range(int(seconds * rate))]
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(
            b"".join(int(v * 9000).to_bytes(2, "little", signed=True) for v in t)
        )
    return buf.getvalue()


@pytest.mark.parametrize("rate_scale", [1.0, 0.8])
def test_wav_to_ogg_opus_produces_playable_voice_note(rate_scale):
    av = pytest.importorskip("av")
    pytest.importorskip("soundfile")
    from openjarvis.speech.voice_note import wav_to_ogg_opus

    ogg = wav_to_ogg_opus(_wav(1.5), rate_scale=rate_scale)

    assert ogg[:4] == b"OggS"
    with av.open(io.BytesIO(ogg)) as container:
        stream = container.streams.audio[0]
        assert stream.codec_context.name == "opus"
        samples = sum(frame.samples for frame in container.decode(stream))
    assert samples / stream.rate == pytest.approx(1.5 / rate_scale, abs=0.05)


def test_synthesizer_applies_voice_settings_and_falls_back():
    config = JarvisConfig()
    config.speech.tts_backend = "kokoro"
    config.speech.voice_id = "am_onyx"
    config.speech.voice_speed = 1.0
    config.speech.voice_pitch = 0.8

    broken = MagicMock(backend_id="kokoro")
    broken.synthesize.side_effect = RuntimeError("model missing")
    working = MagicMock(backend_id="openai_tts")
    working.synthesize.return_value = SimpleNamespace(audio=b"wav")
    backends = iter([broken, working])

    with (
        patch(
            "openjarvis.speech._tts_discovery.get_tts_backend",
            side_effect=lambda *a, **k: next(backends, None),
        ),
        patch(
            "openjarvis.speech.voice_note.wav_to_ogg_opus", return_value=b"OggS"
        ) as encode,
    ):
        synthesize = make_voice_note_synthesizer(config)
        assert synthesize("**Hello** there") == b"OggS"
        assert synthesize("Again") == b"OggS"

    first = broken.synthesize.call_args
    assert first.args == ("Hello there.",)
    assert first.kwargs["voice_id"] == "am_onyx"
    assert first.kwargs["speed"] == pytest.approx(1.25)
    # The fallback backend gets its own default voice, not the kokoro id.
    assert working.synthesize.call_args_list[0].kwargs["voice_id"] == "onyx"
    assert working.synthesize.call_count == 2  # healthy backend is reused
    encode.assert_called_with(b"wav", rate_scale=pytest.approx(0.8))


def test_synthesizer_returns_none_without_backend_or_text():
    with patch("openjarvis.speech._tts_discovery.get_tts_backend", return_value=None):
        synthesize = make_voice_note_synthesizer(JarvisConfig())
        assert synthesize("Hello") is None
        assert synthesize("🚀") is None
