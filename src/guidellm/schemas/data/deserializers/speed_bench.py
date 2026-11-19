from __future__ import annotations

from typing import Any, Literal

from pydantic import AliasChoices, ConfigDict, Field

from guidellm.schemas.data.entrypoints import DataArgs

__all__ = [
    "SPEED_BENCH_CONFIGS",
    "SpeedBenchDataArgs",
]

# The seven configs the resolver and nvidia/SPEED-Bench publish.
SPEED_BENCH_CONFIGS: tuple[str, ...] = (
    "qualitative",
    "throughput_1k",
    "throughput_2k",
    "throughput_8k",
    "throughput_16k",
    "throughput_32k",
    "throughput_100k",
)


@DataArgs.register(["speed_bench", "speedbench", "speed-bench"])
class SpeedBenchDataArgs(DataArgs):
    """Arguments for the SPEED-Bench prompt-dataset deserializer.

    SPEED-Bench (``nvidia/SPEED-Bench``) publishes placeholder rows for prompts
    whose source datasets do not permit redistribution. This deserializer
    resolves them by running the vendored upstream prepare script against the
    original source datasets, caching the resolved config locally, then serving
    categories as views over it. A pre-resolved JSONL may instead be supplied
    via ``source`` to skip resolution entirely.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    kind: Literal["speed_bench", "speedbench", "speed-bench"] = Field(  # type: ignore[assignment]
        default="speed_bench",
        description="Type identifier for the SPEED-Bench deserializer.",
    )
    source: Any = Field(
        default=None,
        validation_alias=AliasChoices("source", "src", "from", "path", "file"),
        description="Optional path to a pre-resolved SPEED-Bench JSONL file "
        "(rows with a 'messages' array). When set, resolution is skipped and "
        "the file is read directly; placeholder rows are still rejected.",
    )
    config: str = Field(
        default="qualitative",
        validation_alias=AliasChoices("config", "subset", "hf_subset"),
        description="SPEED-Bench config to resolve (e.g. 'qualitative', "
        "'throughput_1k', ... 'throughput_100k').",
    )
    category: str | None = Field(
        default=None,
        description="When set, only rows whose 'category' matches are loaded "
        "(e.g. 'coding', 'math', 'low_entropy'). Enables per-category runs.",
    )
    multi_turn: bool = Field(
        default=True,
        description="When True, all turns in a row are emitted as a multi-turn "
        "conversation; when False, only the first user turn is used.",
    )
    hf_dataset_name: str = Field(
        default="nvidia/SPEED-Bench",
        description="HuggingFace dataset identifier for the SPEED-Bench manifest.",
    )
    hf_split: str = Field(
        default="test",
        description="SPEED-Bench manifest split; the dataset publishes 'test'.",
    )
    revision: str | None = Field(
        default=None,
        description="Override the pinned SPEED-Bench manifest revision. Leave "
        "unset to use the pinned, reproducible default.",
    )
    samples: int = Field(
        default=-1,
        description="Maximum number of rows to read; -1 reads all matching rows.",
    )
