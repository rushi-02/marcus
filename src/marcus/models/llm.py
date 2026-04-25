"""Language Model wrapper using mlx-lm (Llama 3.2 3B + LoRA adapter).

Provides both blocking generate() for simplicity and stream_generate()
for the sentence-level streaming pipeline (lower perceived latency).
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

from rich.console import Console

from marcus.config import LLMConfig

console = Console()


class MarcusLLM:
    """MLX-native LLM wrapper with optional LoRA adapter.

    Usage:
        llm = MarcusLLM(config.llm)
        response = llm.generate(messages)
        for token in llm.stream_generate(messages):
            print(token, end="", flush=True)
    """

    def __init__(self, config: LLMConfig) -> None:
        self.config = config
        self._model = None
        self._tokenizer = None
        self._loaded = False

    def _load(self) -> None:
        if self._loaded:
            return
        try:
            from mlx_lm import load
        except ImportError:
            raise ImportError("mlx-lm not installed. Run: uv pip install mlx-lm")

        adapter = self.config.adapter_path
        if adapter and not Path(adapter).exists():
            console.print(
                f"[yellow]Warning:[/yellow] adapter_path '{adapter}' not found — "
                "loading base model without adapter."
            )
            adapter = None

        console.print(f"[cyan]Loading LLM:[/cyan] {self.config.model_id}")
        if adapter:
            console.print(f"  [cyan]+ LoRA adapter:[/cyan] {adapter}")

        self._model, self._tokenizer = load(
            self.config.model_id,
            adapter_path=adapter,
        )
        self._loaded = True
        console.print("[green]LLM loaded.[/green]")

    def _build_prompt(self, messages: list[dict]) -> str:
        """Apply the tokenizer's chat template to produce a prompt string."""
        self._load()
        return self._tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

    def generate(self, messages: list[dict]) -> str:
        """Generate a complete response (blocking).

        Args:
            messages: List of {"role": str, "content": str} dicts.

        Returns:
            Full response string.
        """
        self._load()
        from mlx_lm import generate

        prompt = self._build_prompt(messages)
        response = generate(
            self._model,
            self._tokenizer,
            prompt=prompt,
            max_tokens=self.config.max_tokens,
            temp=self.config.temperature,
            top_p=self.config.top_p,
            verbose=False,
        )
        # mlx-lm generate returns the generated text (not the full sequence)
        return response.strip()

    def stream_generate(self, messages: list[dict]) -> Iterator[str]:
        """Yield response tokens one at a time (streaming).

        Use this for sentence-level streaming: accumulate tokens until a
        sentence boundary, then synthesize and play that sentence while
        generation continues.

        Yields:
            Individual token strings.
        """
        self._load()
        from mlx_lm import stream_generate

        prompt = self._build_prompt(messages)

        for response in stream_generate(
            self._model,
            self._tokenizer,
            prompt=prompt,
            max_tokens=self.config.max_tokens,
            temp=self.config.temperature,
            top_p=self.config.top_p,
        ):
            # stream_generate yields objects with .text (the new token)
            token = response.text if hasattr(response, "text") else str(response)
            yield token

    def unload(self) -> None:
        """Free model memory."""
        self._model = None
        self._tokenizer = None
        self._loaded = False
        import gc
        gc.collect()
