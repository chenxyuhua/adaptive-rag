from .base import GenerationResult, LLMClient


def build_llm(provider: str, **kwargs) -> LLMClient:
    """Provider-agnostic factory. Add a branch when a new provider lands."""
    if provider == "gemini":
        from .gemini import GeminiClient

        return GeminiClient(**kwargs)
    raise ValueError(f"Unknown LLM provider: {provider!r}")


__all__ = ["LLMClient", "GenerationResult", "build_llm"]
