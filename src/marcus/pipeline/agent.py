"""MarcusAgent — the main pipeline orchestrator.

Wires: AudioCapture → ASR → ConversationManager → LLM → TTS → AudioPlayer

Half-duplex flow (no barge-in):
    LISTENING → user speaks → silence → utterance enqueued
    PROCESSING → ASR → LLM streams tokens
    SPEAKING (mic muted) → TTS → playback
    LISTENING (mic resumed)

Streaming TTS: LLM tokens accumulate until a sentence boundary, then that
sentence is synthesized and played while the LLM continues generating
the next sentence. This is what makes the response feel responsive even
though full generation takes 5-10 seconds.
"""

from __future__ import annotations

import asyncio

from rich.console import Console

from marcus.config import MarcusConfig, load_config
from marcus.models.asr import MarcusASR
from marcus.models.llm import MarcusLLM
from marcus.models.tts import MarcusTTS
from marcus.pipeline.audio_io import AgentState, AudioCapture, AudioPlayer
from marcus.pipeline.conversation import ConversationManager, load_system_prompt
from marcus.ui.feedback import FeedbackCollector

console = Console()

SENTENCE_ENDINGS = {".", "!", "?", ":"}


class MarcusAgent:
    """Real-time Stoic voice agent.

    Usage:
        agent = MarcusAgent()
        agent.preload()                          # Load all 3 models upfront
        asyncio.run(agent.run_streaming())       # voice-to-voice, half-duplex
        asyncio.run(agent.text_chat())           # text-only mode
    """

    def __init__(self, config: MarcusConfig | None = None) -> None:
        self.config = config or load_config()

        self.asr = MarcusASR(self.config.asr)
        self.llm = MarcusLLM(self.config.llm)
        self.tts = MarcusTTS(self.config.tts)

        self.player = AudioPlayer(
            self.config.audio,
            sample_rate=self.config.tts.sample_rate,
        )
        self.capture = AudioCapture(self.config.audio, player=self.player)

        system_prompt = load_system_prompt(self.config.llm.system_prompt_path)
        self.conversation = ConversationManager(
            system_prompt=system_prompt,
            max_turns=10,
        )

        self.feedback = FeedbackCollector()
        self._state = AgentState.IDLE

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def preload(self) -> None:
        """Load all 3 models upfront so the first user utterance isn't slow.

        Without this, the first ASR/LLM/TTS call each pays the full model-load
        cost (5-15s each), while the user is sitting in front of a microphone
        wondering why nothing is happening.

        Also warms the TTS by synthesizing a tiny sample (Kokoro's first-call
        latency is ~3s, but after that it's <300ms per sentence).
        """
        import time
        console.print("[cyan]Loading models...[/cyan]")
        t0 = time.time()

        self.asr._load()
        console.print(f"  [dim]ASR ready ({time.time() - t0:.1f}s)[/dim]")

        t1 = time.time()
        self.llm._load()
        console.print(f"  [dim]LLM ready ({time.time() - t1:.1f}s)[/dim]")

        t2 = time.time()
        self.tts._load()
        # Warm the TTS pipeline by synthesizing one short token
        try:
            self.tts.synthesize("Ready.")
        except Exception:
            pass
        console.print(f"  [dim]TTS ready ({time.time() - t2:.1f}s)[/dim]")

        console.print(f"[green]All models loaded in {time.time() - t0:.1f}s.[/green]\n")

    # ------------------------------------------------------------------
    # Main loops
    # ------------------------------------------------------------------

    async def run_streaming(self) -> None:
        """Half-duplex voice loop with sentence-level TTS streaming."""
        self.preload()

        console.print(
            "[bold cyan]Marcus is ready.[/bold cyan] "
            "Speak, then pause. Wait for him to finish before speaking again.\n"
        )
        self._state = AgentState.LISTENING

        async for utterance_audio in self.capture.listen():
            self._state = AgentState.PROCESSING

            # ASR
            user_text = self.asr.transcribe(
                utterance_audio, self.config.audio.sample_rate
            )
            user_text = self._filter_hallucination(user_text)
            if not user_text:
                self._state = AgentState.LISTENING
                continue

            console.print(f"[green]You:[/green] {user_text}")
            self.conversation.add_user(user_text)

            # Half-duplex: mute mic before TTS playback
            self.capture.pause()
            self._state = AgentState.SPEAKING

            full_response = ""
            sentence_buffer = ""

            for token in self.llm.stream_generate(self.conversation.get_messages()):
                full_response += token
                sentence_buffer += token

                if token.rstrip().endswith(tuple(SENTENCE_ENDINGS)):
                    sentence = sentence_buffer.strip()
                    if sentence:
                        audio = self.tts.synthesize(sentence)
                        self.player.play(
                            audio, sample_rate=self.config.tts.sample_rate
                        )
                    sentence_buffer = ""

            # Flush any final partial sentence
            if sentence_buffer.strip():
                audio = self.tts.synthesize(sentence_buffer.strip())
                self.player.play(audio, sample_rate=self.config.tts.sample_rate)

            if full_response.strip():
                self.conversation.add_assistant(full_response.strip())
                console.print(f"[cyan]Marcus:[/cyan] {full_response.strip()}\n")

            # Resume mic for next utterance
            self.capture.resume()
            self._state = AgentState.LISTENING

    async def run(self) -> None:
        """Non-streaming voice loop (full response synthesized at once).

        Slower perceived latency than run_streaming() but simpler and easier
        to reason about. Kept for debugging.
        """
        self.preload()

        console.print(
            "[bold cyan]Marcus is ready.[/bold cyan] Speak, then pause.\n"
        )
        self._state = AgentState.LISTENING

        async for utterance_audio in self.capture.listen():
            self._state = AgentState.PROCESSING

            user_text = self.asr.transcribe(
                utterance_audio, self.config.audio.sample_rate
            )
            user_text = self._filter_hallucination(user_text)
            if not user_text:
                self._state = AgentState.LISTENING
                continue

            console.print(f"[green]You:[/green] {user_text}")
            self.conversation.add_user(user_text)

            response_text = self.llm.generate(self.conversation.get_messages())
            self.conversation.add_assistant(response_text)
            console.print(f"[cyan]Marcus:[/cyan] {response_text}\n")

            self.capture.pause()
            self._state = AgentState.SPEAKING
            audio = self.tts.synthesize(response_text)
            self.player.play(audio, sample_rate=self.config.tts.sample_rate)
            self.capture.resume()

            self._state = AgentState.LISTENING

    async def text_chat(self) -> None:
        """Text-only mode — no microphone, no TTS, useful for testing the LLM."""
        # Text mode only needs the LLM
        self.llm._load()

        console.print(
            "\n[bold cyan]Marcus (text mode).[/bold cyan] "
            "Type your message. 'quit' to exit.\n"
        )

        while True:
            try:
                user_input = console.input("[green]You:[/green] ").strip()
            except (EOFError, KeyboardInterrupt):
                break

            if not user_input or user_input.lower() in ("quit", "exit", "q"):
                break

            self.conversation.add_user(user_input)
            response = self.llm.generate(self.conversation.get_messages())
            self.conversation.add_assistant(response)
            console.print(f"[cyan]Marcus:[/cyan] {response}\n")

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _filter_hallucination(text: str) -> str:
        """Reject obvious Whisper hallucinations.

        Whisper hallucinates on silence, low-volume speech, and music. Common
        failure patterns:
        - One word repeated 5+ times ("freaking freaking freaking...")
        - Very short utterances that are likely background noise
        - Stock phrases Whisper falls back to ("thanks for watching", etc.)
        """
        text = text.strip()
        if not text:
            return ""

        words = text.split()

        # Too short (likely noise / single barked word)
        if len(words) < 2:
            return ""

        # Word-repetition hallucination ("freaking freaking ...")
        if len(words) >= 5:
            # Count consecutive duplicates
            max_run = 1
            run = 1
            for i in range(1, len(words)):
                if words[i].lower() == words[i - 1].lower():
                    run += 1
                    max_run = max(max_run, run)
                else:
                    run = 1
            if max_run >= 4:
                console.print(
                    f"[dim yellow](filtered hallucination: '{text[:60]}...')[/dim yellow]"
                )
                return ""

        # Stock Whisper fallback phrases on silence/music
        stock = {
            "thanks for watching", "thank you for watching",
            "thanks for watching!", "you", "thank you.", "you.",
            "subtitles by", "♪",
        }
        if text.lower().rstrip(" .!?") in stock:
            console.print(
                f"[dim yellow](filtered Whisper fallback: '{text}')[/dim yellow]"
            )
            return ""

        return text

    def record_feedback(self, thumbs_up: bool) -> None:
        """Record thumbs up/down for the last Marcus response."""
        user_msg = self.conversation.last_user_message
        assistant_msg = self.conversation.last_assistant_message
        if user_msg and assistant_msg:
            self.feedback.record(
                user_message=user_msg,
                assistant_message=assistant_msg,
                thumbs_up=thumbs_up,
            )
