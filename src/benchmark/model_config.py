"""Model configuration for benchmarks."""

from dataclasses import dataclass, field
from typing import Any


@dataclass
class ModelConfig:
    """
    Configuration for a specific model in benchmarks.

    Contains model identifier, provider settings, and generation parameters.
    """

    name: str  # Model identifier (e.g., "claude-sonnet-4-20250514", "gpt-4")
    provider: str  # Provider name (e.g., "anthropic", "openai")
    temperature: float = 0.0
    max_tokens: int = 4096
    top_p: float | None = None
    top_k: int | None = None
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0
    stop_sequences: list[str] = field(default_factory=list)
    extra_kwargs: dict[str, Any] = field(default_factory=dict)

    @property
    def model_key(self) -> str:
        """Get a unique key for this model configuration."""
        return f"{self.provider}/{self.name}"

    def to_llm_kwargs(self) -> dict[str, Any]:
        """Convert to kwargs suitable for LLM client calls."""
        kwargs = {
            "model": self.name,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }

        if self.top_p is not None:
            kwargs["top_p"] = self.top_p
        if self.top_k is not None:
            kwargs["top_k"] = self.top_k
        if self.frequency_penalty != 0.0:
            kwargs["frequency_penalty"] = self.frequency_penalty
        if self.presence_penalty != 0.0:
            kwargs["presence_penalty"] = self.presence_penalty
        if self.stop_sequences:
            kwargs["stop_sequences"] = self.stop_sequences

        kwargs.update(self.extra_kwargs)
        return kwargs

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ModelConfig":
        """Create ModelConfig from dictionary."""
        return cls(**data)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "provider": self.provider,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "top_p": self.top_p,
            "top_k": self.top_k,
            "frequency_penalty": self.frequency_penalty,
            "presence_penalty": self.presence_penalty,
            "stop_sequences": self.stop_sequences.copy(),
            "extra_kwargs": self.extra_kwargs.copy(),
        }
