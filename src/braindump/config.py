"""Shared configuration for braindump CLI."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Any, cast

DEFAULT_MODEL = "gateway/anthropic:claude-sonnet-4-5"
DEFAULT_EMBEDDING_MODEL = "gateway/openai:text-embedding-3-small"

_model_name: str = DEFAULT_MODEL
_embedding_model_name: str = DEFAULT_EMBEDDING_MODEL


def configure_models(model: str | None = None, embedding_model: str | None = None) -> None:
    """Set the model names used by make_model/make_embedding_model."""
    global _model_name, _embedding_model_name
    if model:
        _model_name = model
    if embedding_model:
        _embedding_model_name = embedding_model


def init_env() -> None:
    """Load .env and configure logfire. Must be called before importing pydantic_ai."""
    from dotenv import load_dotenv

    load_dotenv()

    import logfire

    logfire.configure(send_to_logfire="if-token-present", console=False)
    logfire.instrument_pydantic_ai()
    logfire.instrument_httpx(capture_all=True)


def make_retry_client():
    """Create an httpx AsyncClient with retry transport for rate limits."""
    from httpx import AsyncClient, HTTPStatusError, TransportError
    from pydantic_ai.retries import AsyncTenacityTransport, RetryConfig, wait_retry_after
    from tenacity import retry_if_exception_type, stop_after_attempt, wait_exponential

    return AsyncClient(
        transport=AsyncTenacityTransport(
            config=RetryConfig(
                # TransportError covers httpx connection/read/write failures (httpx.ConnectError
                # is not a builtin ConnectionError, so it used to be dropped without retrying).
                retry=retry_if_exception_type((HTTPStatusError, TransportError, ConnectionError)),
                wait=wait_retry_after(fallback_strategy=wait_exponential(multiplier=1, max=60)),
                stop=stop_after_attempt(5),
                reraise=True,
            ),
            validate_response=lambda r: r.raise_for_status(),
        ),
    )


def _retry_provider_factory(provider: str) -> Any:
    """Create a provider with retry HTTP client, similar to pydantic_ai's infer_provider."""
    http_client = make_retry_client()
    if provider.startswith("gateway/"):
        from pydantic_ai.providers.gateway import gateway_provider

        return gateway_provider(provider.removeprefix("gateway/"), http_client=http_client)  # type: ignore[arg-type]
    elif provider in ("google-vertex", "google-gla"):
        from pydantic_ai.providers.google import GoogleProvider

        return GoogleProvider(vertexai=provider == "google-vertex", http_client=http_client)
    else:
        from pydantic_ai.providers import infer_provider_class

        return infer_provider_class(provider)(http_client=http_client)  # type: ignore[call-arg]


def make_model():
    """Create a pydantic_ai Model from the configured model name, with retry HTTP client."""
    from pydantic_ai.models import infer_model

    if _model_name.startswith(("openai:", "openai-chat:")) and (
        _openai_compat_quirks_enabled() or _reasoning_effort()
    ):
        provider_name, model_name = _model_name.split(":", 1)
        return _make_openai_chat_model(model_name, _retry_provider_factory(provider_name))

    return infer_model(_model_name, provider_factory=_retry_provider_factory)


def _openai_compat_quirks_enabled() -> bool:
    return os.environ.get("BRAINDUMP_OPENAI_COMPAT_QUIRKS", "").lower() in ("1", "true", "yes")


def _reasoning_effort() -> str | None:
    effort = os.environ.get("BRAINDUMP_REASONING_EFFORT", "").strip().lower()
    if not effort:
        return None
    allowed = ("none", "minimal", "low", "medium", "high", "xhigh")
    if effort not in allowed:
        raise ValueError(
            f"BRAINDUMP_REASONING_EFFORT must be one of {'/'.join(allowed)}, got {effort!r}"
        )
    return effort


def _make_openai_chat_model(model_name: str, provider: Any):
    """Build an OpenAIChatModel with optional provider-quirk tolerance and reasoning effort.

    BRAINDUMP_OPENAI_COMPAT_QUIRKS covers OpenAI-compatible providers that:
    - reject the `tool_choice="required"` sent by pydantic-ai's default 'tool' output mode, so we
      default to 'native' structured output (`response_format` JSON schema) instead;
    - return values that fail the strict response validation in `_validate_completion` (e.g.
      `service_tier: "standard"`, which OpenAI's Literal does not allow), which we drop and retry.

    BRAINDUMP_REASONING_EFFORT sets a model-level default reasoning effort, as pydantic-ai sends
    none by default and providers then use their own (typically highest) level.
    """
    from pydantic_ai.models.openai import OpenAIChatModel, OpenAIChatModelSettings

    quirk_tolerant = _openai_compat_quirks_enabled()

    if quirk_tolerant:

        class _QuirkTolerantOpenAIChatModel(OpenAIChatModel):
            def _validate_completion(self, response: Any) -> Any:
                from openai.types.chat import ChatCompletion
                from pydantic import ValidationError

                data = response.model_dump()
                try:
                    return ChatCompletion.model_validate(data)
                except ValidationError as e:
                    if not any(err.get("loc") == ("service_tier",) for err in e.errors()):
                        raise
                    data.pop("service_tier", None)
                    return ChatCompletion.model_validate(data)

        model_cls: Any = _QuirkTolerantOpenAIChatModel
    else:
        model_cls = OpenAIChatModel

    settings: OpenAIChatModelSettings | None = None
    if effort := _reasoning_effort():
        settings = OpenAIChatModelSettings(openai_reasoning_effort=cast(Any, effort))

    model = model_cls(model_name, provider=provider, settings=settings)
    if quirk_tolerant:
        model.profile.default_structured_output_mode = "native"
    return model


def make_embedding_model():
    """Create a pydantic_ai EmbeddingModel from the configured name, with retry HTTP client."""
    from pydantic_ai.embeddings import infer_embedding_model

    return infer_embedding_model(_embedding_model_name, provider_factory=_retry_provider_factory)


async def run_agent_with_retry(
    agent: Any,
    prompt: str,
    *,
    attempts: int = 6,
    base_delay: float = 20.0,
    max_delay: float = 180.0,
) -> Any:
    """Run a pydantic_ai Agent, retrying transient API failures with backoff.

    The httpx transport retry in make_retry_client covers short bursts of 429s/connection resets;
    this layer rides out provider throttling that outlives that budget (observed on long runs:
    a 429 exhausted the transport retries and aborted a whole stage) without killing the stage.
    """
    import asyncio
    import random

    import logfire
    from httpx import HTTPStatusError, TransportError
    from pydantic_ai.exceptions import ModelAPIError

    retryable = (ModelAPIError, HTTPStatusError, TransportError, ConnectionError)
    for attempt in range(1, attempts + 1):
        try:
            return await agent.run(prompt)
        except retryable as exc:
            if attempt >= attempts:
                raise
            delay = min(base_delay * 2 ** (attempt - 1), max_delay) * (0.5 + random.random())
            # Console line as well as logfire: long unattended stages must show retry activity
            # even when Logfire is not configured (no token).
            from braindump.progress import console

            console.print(
                f"[yellow]LLM call failed ({type(exc).__name__}), retry "
                f"{attempt}/{attempts - 1} in {delay:.0f}s[/yellow]"
            )
            logfire.info(
                "retrying failed LLM call",
                attempt=attempt,
                attempts=attempts,
                delay=round(delay, 1),
                error=str(exc)[:200],
            )
            await asyncio.sleep(delay)
    raise AssertionError("unreachable")


def compute_file_hash(path: Path) -> str:
    """Compute SHA-256 hash of a file's contents."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def compute_multi_file_hash(paths: list[Path]) -> str:
    """Compute SHA-256 hash over multiple files (sorted by name for stability)."""
    h = hashlib.sha256()
    for path in sorted(paths, key=lambda p: p.name):
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                h.update(chunk)
    return h.hexdigest()


class RepoConfig:
    """Resolves all data paths from a repo string (owner/repo)."""

    def __init__(self, repo: str) -> None:
        self.repo = repo
        # Find project root (where pyproject.toml lives)
        self._root = Path(__file__).resolve().parent.parent.parent
        self.data_dir = self._root / "data" / repo

    def stage_dir(self, stage: str) -> Path:
        return self.data_dir / stage

    @property
    def pr_data_dir(self) -> Path:
        return self.stage_dir("1-download")

    @property
    def review_comments_dir(self) -> Path:
        return self.pr_data_dir / "review_comments"

    @property
    def extractions_path(self) -> Path:
        return self.stage_dir("2-extract") / "extractions.jsonl"

    @property
    def rules_path(self) -> Path:
        return self.stage_dir("3-synthesize") / "rules.jsonl"

    @property
    def rules_deduped_path(self) -> Path:
        return self.stage_dir("4-dedupe") / "rules.jsonl"

    @property
    def overrides_path(self) -> Path:
        return self.data_dir / "rule_overrides.jsonl"

    @property
    def placements_path(self) -> Path:
        return self.stage_dir("5-place") / "placements.jsonl"

    @property
    def organization_path(self) -> Path:
        return self.stage_dir("6-group") / "organized_rules.json"

    @property
    def output_dir(self) -> Path:
        return self.stage_dir("7-generate")
