"""Marcus Gradio web UI with FastRTC WebRTC voice chat.

Features:
- Real-time voice conversation via WebRTC (no manual recording/uploading)
- FastRTC ReplyOnPause handles voice activity detection and turn-taking
- Feedback buttons (thumbs up/down) after each response
- Live transcript displayed alongside audio

Usage:
    uv run python -m marcus.ui.gradio_app
    # Then open http://localhost:7860
"""

from __future__ import annotations

import numpy as np
from rich.console import Console

console = Console()

# ---------------------------------------------------------------------------
# Lazy imports — gradio/fastrtc are optional dependencies
# ---------------------------------------------------------------------------

def _check_imports() -> None:
    try:
        import gradio  # noqa: F401
        import fastrtc  # noqa: F401
    except ImportError:
        raise ImportError(
            "Gradio and FastRTC required for web UI.\n"
            "Install: uv pip install 'marcus[ui]'"
        )


# ---------------------------------------------------------------------------
# Global agent state (loaded once on startup)
# ---------------------------------------------------------------------------

_agent = None
_config = None


def _get_agent():
    global _agent, _config
    if _agent is None:
        from marcus.config import load_config
        from marcus.pipeline.agent import MarcusAgent
        _config = load_config()
        _agent = MarcusAgent(_config)
        console.print("[green]Marcus agent initialized for web UI.[/green]")
    return _agent


# ---------------------------------------------------------------------------
# FastRTC handler
# ---------------------------------------------------------------------------

def respond_to_speech(audio: tuple[int, np.ndarray]):
    """FastRTC ReplyOnPause handler.

    Receives a complete utterance (after pause detected by FastRTC VAD),
    runs ASR → LLM → TTS, and yields audio chunks for streaming playback.

    Args:
        audio: (sample_rate, audio_array) tuple from FastRTC.

    Yields:
        (sample_rate, audio_chunk) tuples for streaming output.
    """
    agent = _get_agent()
    sample_rate, audio_data = audio

    # Ensure float32 mono
    if audio_data.dtype != np.float32:
        audio_data = audio_data.astype(np.float32)
        if audio_data.max() > 1.0:
            audio_data = audio_data / 32768.0  # int16 → float32

    if audio_data.ndim > 1:
        audio_data = audio_data[:, 0]

    # ASR
    user_text = agent.asr.transcribe(audio_data, sample_rate)
    if not user_text.strip():
        return

    console.print(f"[green]You:[/green] {user_text}")

    # LLM
    agent.conversation.add_user(user_text)

    full_response = ""
    sentence_buffer = ""
    tts_sample_rate = agent.config.tts.sample_rate

    for token in agent.llm.stream_generate(agent.conversation.get_messages()):
        full_response += token
        sentence_buffer += token

        # Yield audio at sentence boundaries
        if token.rstrip().endswith((".", "!", "?", ":")):
            sentence = sentence_buffer.strip()
            if sentence:
                audio_chunk = agent.tts.synthesize(sentence)
                if len(audio_chunk) > 0:
                    yield (tts_sample_rate, audio_chunk)
            sentence_buffer = ""

    # Flush remaining
    if sentence_buffer.strip():
        audio_chunk = agent.tts.synthesize(sentence_buffer.strip())
        if len(audio_chunk) > 0:
            yield (tts_sample_rate, audio_chunk)

    if full_response.strip():
        agent.conversation.add_assistant(full_response.strip())
        console.print(f"[cyan]Marcus:[/cyan] {full_response.strip()}\n")


# ---------------------------------------------------------------------------
# Gradio UI
# ---------------------------------------------------------------------------

def build_app():
    """Build and return the Gradio application."""
    _check_imports()

    import gradio as gr
    from fastrtc import ReplyOnPause, Stream

    stream = Stream(
        handler=ReplyOnPause(respond_to_speech),
        modality="audio",
        mode="send-receive",
    )

    with gr.Blocks(title="Marcus — Stoic Voice Agent", theme=gr.themes.Monochrome()) as demo:
        gr.Markdown(
            "# Marcus Aurelius — Stoic Voice Agent\n"
            "*Speak your struggles. Receive Stoic wisdom.*"
        )

        with gr.Row():
            with gr.Column(scale=2):
                stream.ui  # embeds the FastRTC voice widget

            with gr.Column(scale=1):
                gr.Markdown("### Session Transcript")
                transcript_box = gr.Textbox(
                    label="",
                    lines=15,
                    interactive=False,
                    placeholder="Your conversation will appear here...",
                )

        with gr.Row():
            gr.Markdown("**Was that response helpful?**")
            thumbs_up_btn = gr.Button("👍 Yes", variant="primary", scale=1)
            thumbs_down_btn = gr.Button("👎 No", variant="secondary", scale=1)
            feedback_status = gr.Markdown("", visible=False)

        def give_feedback(thumbs_up: bool):
            agent = _get_agent()
            agent.record_feedback(thumbs_up=thumbs_up)
            count = agent.feedback.count()
            msg = "✓ Feedback recorded. " + (
                f"({count} total — Marcus is learning from you)"
            )
            return gr.Markdown(value=msg, visible=True)

        thumbs_up_btn.click(
            fn=lambda: give_feedback(True),
            outputs=feedback_status,
        )
        thumbs_down_btn.click(
            fn=lambda: give_feedback(False),
            outputs=feedback_status,
        )

        gr.Markdown(
            "---\n"
            "*Built with mlx-lm · mlx-audio · Gradio FastRTC · "
            "Finetuned on Marcus Aurelius' Meditations*"
        )

    return demo


def main():
    """Launch the Gradio web UI."""
    demo = build_app()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
    )


if __name__ == "__main__":
    main()
