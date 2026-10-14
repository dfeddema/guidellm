from __future__ import annotations

from pydantic import Field, model_validator

from guidellm.schemas.data.entrypoints import DataArgs
from guidellm.utils.dict import deep_update

__all__ = ["TraceDataArgs"]


class TraceDataArgs(DataArgs):
    """Abstract class meant to be inherited by a trace format.
    For testing, use `trace_minimal.MinimalTraceFormatArgs` instead."""

    kind: str = Field(
        description="Type identifier for the trace dataset deserializer.",
    )
    source: DataArgs = Field(
        description="Source dataset to read trace data from.",
    )
    timestamp_column: str = Field(
        default="timestamp",
        description="Column name for timestamps in the trace file.",
    )
    prompt_tokens_column: str = Field(
        default="input_length",
        description="Column name for prompt token counts in the trace file.",
    )
    output_tokens_column: str = Field(
        default="output_length",
        description="Column name for output token counts in the trace file.",
    )
    conversation_id_column: str | None = Field(
        default=None,
        description=(
            "Column name for conversation IDs. Required for formats "
            "with conversation-scoped trace data such as hash IDs."
        ),
    )
    max_wait: float | None = Field(
        default=None,
        gt=0,
        description=(
            "Maximum wait in original trace seconds between consecutive requests "
            "within one session. Applied independently per conversation. "
            "Larger gaps are shortened to this value; later requests in that session "
            "shift earlier by the trimmed amount."
        ),
    )
    max_session_wait: float | None = Field(
        default=None,
        gt=0,
        description=(
            "Maximum wait in original trace seconds between consecutive sessions. "
            "If the next session starts more than this many seconds after "
            "the previous session's last request, that session is shifted earlier."
        ),
    )
    min_concurrent_sessions: int | None = Field(
        default=None,
        ge=1,
        description=(
            "Pack sessions so at least this many overlap during steady state. "
            "The first N sessions start together; each later session starts as soon "
            "as session i-N ends, but never later than its original start."
        ),
    )
    time_scale: float = Field(
        default=1.0,
        gt=0,
        description=(
            "Scale factor applied to relative timestamps after wait and pack caps. "
            "1.0 preserves original timing; values above 1.0 stretch intervals; "
            "values below 1.0 compress them."
        ),
    )
    copies: int = Field(
        default=1,
        ge=1,
        description=(
            "Number of sequential full-dataset replays. Pass k+1 starts when "
            "pass k's last request is scheduled. Use wait/pack options to "
            "raise parallelism first; raise copies when the packed pass is "
            "too short for the benchmark. Hash-id formats use a separately "
            "salted global token-block table per pass."
        ),
    )
    copy_offset: float = Field(
        default=1.0,
        ge=0,
        description=(
            "Where the next copy starts relative to the prior copy's wait-capped "
            "span. 0 is the prior start, 1 is the prior end, values between "
            "interpolate, and values above 1 add a gap."
        ),
    )
    reuse_decoded_blocks: bool = Field(
        default=False,
        description=(
            "Decode each hash-id token block to text once, cache it, and compose "
            "each prompt by concatenating the cached block strings, instead of "
            "re-decoding every request's full concatenated token list. On "
            "multi-turn traces with heavy prefix reuse (e.g. WEKA Claude Code) "
            "this decodes only unique blocks rather than total tokens, a large "
            "speedup to DAG construction. Fidelity is preserved: the same hash-id "
            "sequence always maps to the same text, and token counts and "
            "prefix-cache reuse are unchanged. For byte-level BPE tokenizers the "
            "composed prompt is byte-identical to full-list decoding; other "
            "tokenizers may differ by a few characters at 64-token block seams."
        ),
    )

    @model_validator(mode="after")
    def merge_kwargs(self):
        """Merge load_kwargs from this object into the source DataArgs."""
        if self.load_kwargs:
            deep_update(self.source.load_kwargs, self.load_kwargs)

        return self
