"""MarcusAgent — the main pipeline orchestrator.

Wires: AudioCapture → ASR → ConversationManager → LLM → TTS → AudioPlayer

Two conversation modes:
- run(): Simple blocking mode — full LLM response, then synthesize.
- run_streaming(): Sentence-level streaming — Marcus starts speaking
  before he finishes thinking. Lower perceived latency.

State machine:
    IDLE → LISTENING → PROCESSING → SPEAKING → LISTENING (loop)
                ↑                       |
                └──── (barge-in) ───────┘
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import numpy as np
from rich.console import Console

from marcus.config import MarcusConfig, load_config
from marcus.models.asr import MarcusASR
from marcus.models.llm import MarcusLLM
from marcus.models.tts import MarcusTTS, _split_sentences
from marcus.pipeline.audio_io import AgentState, AudioCapture, AudioPlayer
from marcus.pipeline.conversation import ConversationManager, load_system_prompt
from marcus.ui.feedback import FeedbackCollector

console = Console()

# Sentence-ending punctuation for streaming segmentation
SENTENCE_ENDINGS = {".", "!", "?", ":"}


class MarcusAgent:
    """Real-time Stoic voice agent.

    Usage:
        agent = MarcusAgent()
        asyncio.run(agent.run_streaming())   # voice-to-voice with barge-in
        asyncio.run(agent.text_chat())        # text-only (useful for testing LLM)
    """

    def __init__(self, config: MarcusConfig | None = None) -> None:
        self.config = config or load_config()

        # Models (lazy-loaded on first use)
        self.asr = MarcusASR(self.config.asr)
        self.llm = MarcusLLM(self.config.llm)
        self.tts = MarcusTTS(self.config.tts)

        # Audio I/O
        self.player = AudioPlayer(
            self.config.audio,
            sample_rate=self.config.tts.sample_rate,
        )
        self.capture = AudioCapture(self.config.audio, player=self.player)

        # Conversation
        system_prompt = load_system_prompt(self.config.llm.system_prompt_path)
        self.conversation = ConversationManager(
            system_prompt=system_prompt,
            max_turns=10,
        )

        # Feedback
        self.feedback = FeedbackCollector()

        self._state = AgentState.IDLE

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def run(self) -> None:
        """Simple blocking voice loop (full response before speaking)."""
        self._state = AgentState.IDLE
        console.print("\n[bold cyan]Marcus is ready.[/bold cyan] Speak to begin.\n")
        self._state = AgentState.LISTENING

        async for utterance_audio in self.capture.listen():
            self._state = AgentState.PROCESSING

            # 1. Transcribe
            user_text = self.asr.transcribe(utterance_audio, self.config.audio.sample_rate)
            if not user_text.strip():
                self._state = AgentState.LISTENING
                continue

            console.print(f"[green]You:[/green] {user_text}")

            # 2. LLM
            self.conversation.add_user(user_text)
            response_text = self.llm.generate(self.conversation.get_messages())
            self.conversation.add_assistant(response_text)
            console.print(f"[cyan]Marcus:[/cyan] {response_text}\n")

            # 3. TTS + play
            self._state = AgentState.SPEAKING
            audio = self.tts.synthesize(response_text)
            completed = self.player.play_interruptible(
                audio, sample_rate=self.config.tts.sample_rate
            )
            if not completed:
                console.print("[dim](interrupted)[/dim]")

            self._state = AgentState.LISTENING

    async def run_streaming(self) -> None:
        """Streaming voice loop with barge-in support.

        LLM tokens accumulate until a sentence boundary, then that
        sentence is synthesized and played while the LLM continues
        generating the next sentence. Barge-in stops playback immediately.
        """
        self._state = AgentState.IDLE
        console.print(
            "\n[bold cyan]Marcus is ready.[/bold cyan] "
            "Speak to begin. Interrupt him at any time.\n"
        )
        self._state = AgentState.LISTENING

        async for utterance_audio in self.capture.listen():
            self._state = AgentState.PROCESSING

            user_text = self.asr.transcribe(utterance_audio, self.config.audio.sample_rate)
            if not user_text.strip():
                self._state = AgentState.LISTENING
                continue

            console.print(f"[green]You:[/green] {user_text}")
            self.conversation.add_user(user_text)

            # Streaming LLM → sentence-level TTS
            full_response = ""
            sentence_buffer = ""
            self._state = AgentState.SPEAKING

            for token in self.llm.stream_generate(self.conversation.get_messages()):
                # Barge-in: user started speaking during generation
                if self.player._interrupted:
                    break

                full_response += token
                sentence_buffer += token

                # Flush on sentence boundary
                if token.rstrip().endswith(tuple(SENTENCE_ENDINGS)):
                    sentence = sentence_buffer.strip()
                    if sentence:
                        audio = self.tts.synthesize(sentence)
                        completed = self.player.play_interruptible(
                            audio, sample_rate=self.config.tts.sample_rate
                        )
                        if not completed:
                            console.print("[dim](interrupted mid-sentence)[/dim]")
                            sentence_buffer = ""
                            break
                    sentence_buffer = ""

            # Play any remaining text not ending with punctuation
            if sentence_buffer.strip() and not self.player._interrupted:
                audio = self.tts.synthesize(sentence_buffer.strip())
                self.player.play_interruptible(audio, sample_rate=self.config.tts.sample_rate)

            if full_response.strip():
                self.conversation.add_assistant(full_response.strip())
                console.print(f"[cyan]Marcus:[/cyan] {full_response.strip()}\n")

            self._state = AgentState.LISTENING

    async def text_chat(self) -> None:
        """Text-only conversation mode — no audio required.

        Useful for testing the LLM persona without a microphone or TTS.
        """
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
