"""Voice input/output helpers for the interactive chat session."""

from __future__ import annotations

from typing import Any, Optional

from rich.markup import escape

VOICE_EXIT = object()

# Backend selection is shared with the API server and lives in
# openjarvis.speech._tts_discovery. Importing that at module level would pull
# the whole speech stack -- numpy included -- into every `jarvis` invocation,
# which tests/cli/test_cli.py guards against, so these aliases resolve lazily.
_LAZY_TTS_NAMES = {
    "_TTS_BACKEND_ORDER": "TTS_BACKEND_ORDER",
    "_BACKEND_DEFAULT_VOICE": "BACKEND_DEFAULT_VOICE",
}


def __getattr__(name: str) -> Any:
    target = _LAZY_TTS_NAMES.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    from openjarvis.speech import _tts_discovery

    return getattr(_tts_discovery, target)


def _terminal_safe_text(value: object) -> str:
    """Remove terminal controls and escape Rich markup from dynamic text."""
    printable = "".join(
        char for char in str(value) if char in ("\n", "\t") or char.isprintable()
    )
    return escape(printable)


class VoiceSession:
    """Cache healthy speech backends for one interactive chat session."""

    def __init__(self, config: object | None = None) -> None:
        self._config = config
        self._stt_resolved = False
        self._stt_backend: Any = None
        self._tts_backend: Any = None
        self._tts_attempted: set[str] = set()
        self._voice_prefs: tuple[str, str, float] | None = None
        self._voice_warned: set[str] = set()
        self._warmed = False
        # Pre-synthesized standby acknowledgement as (wav_bytes, sample_rate).
        self.ack_audio: tuple[bytes, int] | None = None

    def warm_up(self, ack_text: str = "Yes?") -> None:
        """Load the speech models now so the first wake-up has no delay.

        Both models load lazily on first use, which otherwise stalls the
        first acknowledgement for seconds -- long enough for the user to
        finish asking before the assistant starts listening. The spoken
        acknowledgement is synthesized once here and replayed on each wake.
        """
        if self._warmed:
            return
        self._warmed = True
        import io
        import wave

        stt = self.get_stt_backend()
        if stt is not None:
            buf = io.BytesIO()
            with wave.open(buf, "wb") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(16000)
                wf.writeframes(bytes(16000))  # 0.5 s of silence
            try:
                stt.transcribe(
                    buf.getvalue(), format="wav", language=self.get_stt_language()
                )
            except Exception:
                pass  # surfaced on the first real transcription instead

        tts = self.get_tts_backend()
        if tts is not None:
            try:
                result = tts.synthesize(ack_text, **self.synth_kwargs(tts))
                if result.audio:
                    self.ack_audio = (result.audio, result.sample_rate)
            except Exception:
                pass  # speak() falls back to the next backend when used

    def get_stt_backend(self) -> Any:
        """Resolve and health-check STT once, then reuse the loaded backend."""
        if not self._stt_resolved:
            from openjarvis.core.config import load_config
            from openjarvis.speech._discovery import get_speech_backend

            config = self._config if self._config is not None else load_config()
            self._stt_backend = get_speech_backend(config)
            self._stt_resolved = True
        return self._stt_backend

    def get_stt_language(self) -> Optional[str]:
        """Return the configured ``speech.language``, or None to auto-detect."""
        from openjarvis.core.config import load_config

        config = self._config if self._config is not None else load_config()
        return (getattr(config.speech, "language", "") or "").strip() or None

    def get_tts_backend(self) -> Any:
        """Return a cached healthy TTS backend, falling through once per key."""
        if self._tts_backend is not None:
            return self._tts_backend

        from openjarvis.speech._tts_discovery import get_tts_backend

        preferred, _, _ = self.get_voice_preferences()
        self._tts_backend = get_tts_backend(preferred, attempted=self._tts_attempted)
        return self._tts_backend

    def get_voice_preferences(self) -> tuple[str, str, float]:
        """Resolve configured (tts_backend, voice_id, speed), cached per session."""
        if self._voice_prefs is None:
            from openjarvis.core.config import load_config
            from openjarvis.speech._tts_discovery import voice_preferences

            config = self._config if self._config is not None else load_config()
            self._voice_prefs = voice_preferences(config)
        return self._voice_prefs

    def get_voice_pitch(self) -> float:
        """Return the configured playback pitch, clamped to 0.5-1.5."""
        from openjarvis.core.config import load_config
        from openjarvis.speech._tts_discovery import voice_pitch

        config = self._config if self._config is not None else load_config()
        return voice_pitch(config)

    def synth_kwargs(self, backend: Any, console: Any = None) -> dict[str, Any]:
        """Synthesis arguments for *backend*, pre-compensated for pitch.

        Pitch is applied by changing the playback rate, which also changes the
        pace; synthesizing at ``speed / pitch`` cancels that out.
        """
        voice_id, speed = self.voice_for_backend(backend, console)
        kwargs: dict[str, Any] = {
            "output_format": "wav",
            "speed": speed / self.get_voice_pitch(),
        }
        if voice_id:
            kwargs["voice_id"] = voice_id
        return kwargs

    def voice_for_backend(self, backend: Any, console: Any = None) -> tuple[str, float]:
        """Return the voice ID valid for ``backend``, plus the configured speed.

        Voice IDs are not portable across backends, so the configured
        ``speech.voice_id`` is honored only when ``backend`` is the configured
        ``speech.tts_backend``. Otherwise the fallback backend's own default is
        used and the substitution is reported once per session.
        """
        want_backend, voice_id, speed = self.get_voice_preferences()
        active = getattr(backend, "backend_id", "") or ""
        if active == want_backend:
            return voice_id, speed

        from openjarvis.speech._tts_discovery import default_voice_for

        substitute = default_voice_for(active)
        if console is not None and active not in self._voice_warned:
            self._voice_warned.add(active)
            console.print(
                f"[dim yellow]Voice: {want_backend!r} unavailable — using "
                f"{active!r} with its default voice "
                f"({substitute or 'backend default'}).[/dim yellow]"
            )
        return substitute, speed

    def discard_tts_backend(self) -> None:
        """Forget a backend that failed synthesis and allow the next fallback."""
        self._tts_backend = None


def read_voice_input(console: Any, session: VoiceSession) -> Optional[str] | object:
    """Accept a typed command/message, or record after an empty submission."""
    try:
        typed = input("You> [type, or press Enter to speak] ")
    except (EOFError, KeyboardInterrupt):
        return VOICE_EXIT
    typed = typed.strip()
    return typed if typed else record_voice(console, session)


def record_voice(
    console: Any,
    session: VoiceSession | None = None,
) -> Optional[str] | object:
    """Record from mic, transcribe, and return text or a loop sentinel."""
    from openjarvis.speech.voice_io import record_until_silence

    active_session = session or VoiceSession()
    backend = active_session.get_stt_backend()
    if backend is None:
        console.print(
            "[red]No speech-to-text backend available. "
            "Install the voice dependencies with: "
            "pip install 'OpenJarvis[speech]', or configure a healthy "
            "OpenAI/Deepgram backend.[/red]"
        )
        return VOICE_EXIT

    console.print("[dim cyan]Listening… (speak now, stops on silence)[/dim cyan]")
    try:
        audio_bytes = record_until_silence()
    except KeyboardInterrupt:
        return VOICE_EXIT
    except Exception as exc:
        # PortAudio failures are regular Exceptions (and can surface as
        # OSError), so report them without letting terminal control sequences
        # through. SystemExit deliberately continues to propagate, while
        # Ctrl-C maps to the chat loop's graceful-exit sentinel above.
        console.print(f"[red]Mic error: {_terminal_safe_text(exc)}[/red]")
        return VOICE_EXIT

    console.print("[dim]Transcribing…[/dim]")
    try:
        result = backend.transcribe(
            audio_bytes, format="wav", language=active_session.get_stt_language()
        )
        text = result.text.strip()
        if text:
            console.print(f"[bold]You (voice):[/bold] {_terminal_safe_text(text)}")
            return text
        console.print("[dim]Nothing heard — try again.[/dim]")
        return None
    except Exception as exc:
        console.print(f"[red]Transcription error: {_terminal_safe_text(exc)}[/red]")
        return None


_GREETING_WORDS = frozenset({"hey", "hi", "hello", "ok", "okay", "yo", "oh"})
# Transcription often bends a name slightly ("Javis", "Jarvis's"); accept
# close spellings but not different words ("Travis" scores ~0.67).
_WAKE_MATCH_RATIO = 0.8


def extract_wake_request(transcript: str, wake_word: str) -> Optional[str]:
    """Return what was asked of the assistant, or None if it wasn't addressed.

    ``"Hey Jarvis, what time is it?"`` -> ``"what time is it?"``;
    ``"What time is it, Jarvis?"`` -> ``"What time is it"``;
    ``"Hey Jarvis"`` -> ``""`` (addressed, request still to come);
    ``"Pass the salt"`` -> ``None``.
    """
    import difflib
    import re

    target = (re.findall(r"[a-z']+", wake_word.lower()) or [""])[-1]
    if not target:
        return None
    for match in re.finditer(r"[A-Za-z']+", transcript):
        word = match.group().lower().removesuffix("'s")
        if difflib.SequenceMatcher(None, word, target).ratio() < _WAKE_MATCH_RATIO:
            continue
        after = transcript[match.end() :].lstrip(" \t,.!?;:-").rstrip()
        if re.search(r"[A-Za-z0-9]", after):
            return after
        before = transcript[: match.start()]
        words = re.findall(r"[A-Za-z0-9']+", before)
        if all(w.lower() in _GREETING_WORDS for w in words):
            return ""
        return before.strip(" \t,.!?;:-")
    return None


def listen_for_wake_word(
    console: Any,
    session: VoiceSession,
    wake_word: str,
) -> Optional[str] | object:
    """Stay on standby until addressed by *wake_word*, then return the request.

    Every utterance is transcribed locally and discarded unless it mentions
    the wake word. Saying only the wake word gets a spoken acknowledgement
    and a fresh recording for the request itself.
    """
    from openjarvis.speech.voice_io import record_until_silence

    backend = session.get_stt_backend()
    if backend is None:
        console.print(
            "[red]No speech-to-text backend available, so standby mode "
            "cannot listen for the wake word.[/red]"
        )
        return VOICE_EXIT

    if not session._warmed:
        console.print("[dim]Getting ready (loading speech models)...[/dim]")
        session.warm_up()

    announce = True
    while True:
        if announce:
            console.print(
                f"[dim cyan]Standby: say '{_terminal_safe_text(wake_word)}' "
                "to wake me (Ctrl+C to quit).[/dim cyan]"
            )
            announce = False
        try:
            audio_bytes = record_until_silence(wait_for_speech=True)
        except KeyboardInterrupt:
            return VOICE_EXIT
        except Exception as exc:
            console.print(f"[red]Mic error: {_terminal_safe_text(exc)}[/red]")
            return VOICE_EXIT

        try:
            result = backend.transcribe(
                audio_bytes, format="wav", language=session.get_stt_language()
            )
        except Exception as exc:
            console.print(f"[red]Transcription error: {_terminal_safe_text(exc)}[/red]")
            continue

        request = extract_wake_request(result.text or "", wake_word)
        if request is None:
            continue
        if request:
            console.print(f"[bold]You (voice):[/bold] {_terminal_safe_text(request)}")
            return request

        if session.ack_audio is not None:
            from openjarvis.speech.voice_io import play_wav

            try:
                play_wav(
                    session.ack_audio[0],
                    sample_rate=session.ack_audio[1],
                    rate_scale=session.get_voice_pitch(),
                )
            except Exception:
                speak("Yes?", console, session)
        else:
            speak("Yes?", console, session)
        follow_up = record_voice(console, session)
        if follow_up is VOICE_EXIT or follow_up:
            return follow_up
        announce = True  # nothing usable heard; say we're waiting again


def speak(text: str, console: Any, session: VoiceSession | None = None) -> None:
    """Synthesize and play text, reusing a healthy backend for the session."""
    from openjarvis.speech.voice_io import play_wav

    active_session = session or VoiceSession()

    # Resolved before the loop: a bad config value must surface as a config
    # error, not be swallowed by the per-backend fallback handler below.
    try:
        active_session.get_voice_preferences()
    except Exception as exc:
        console.print(f"[red]Invalid speech config: {_terminal_safe_text(exc)}[/red]")
        return

    while (backend := active_session.get_tts_backend()) is not None:
        try:
            result = backend.synthesize(
                text, **active_session.synth_kwargs(backend, console)
            )
            if result.audio:
                play_wav(
                    result.audio,
                    sample_rate=result.sample_rate,
                    rate_scale=active_session.get_voice_pitch(),
                )
            return
        except Exception as exc:
            console.print(
                f"[dim yellow]Voice backend "
                f"{getattr(backend, 'backend_id', '?')!r} failed: "
                f"{_terminal_safe_text(exc)}[/dim yellow]"
            )
            active_session.discard_tts_backend()

    console.print(
        "[dim yellow]No TTS backend available — install kokoro: "
        "pip install kokoro[/dim yellow]"
    )


__all__ = [
    "VOICE_EXIT",
    "VoiceSession",
    "extract_wake_request",
    "listen_for_wake_word",
    "read_voice_input",
    "record_voice",
    "speak",
]
