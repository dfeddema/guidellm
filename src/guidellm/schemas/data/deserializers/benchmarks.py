from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from pydantic import ConfigDict, Field

from guidellm.schemas.data.entrypoints import DataArgs

__all__ = [
    "BENCHMARK_ALIASES",
    "BENCHMARK_PRESETS",
    "BenchmarkDataArgs",
    "BenchmarkPreset",
    "resolve_benchmark_name",
]


@dataclass(frozen=True)
class BenchmarkPreset:
    """Immutable description of how to load one named benchmark prompt dataset.

    :param source: HuggingFace dataset identifier (``load_dataset`` first arg).
    :param split: Dataset split to load.
    :param mode: How a row carries its prompt(s): ``single`` for a flat
        string column, ``turns_list`` for a column holding a list of
        per-turn strings.
    :param text_column: Column holding the prompt (``single``) or the list of
        per-turn prompt strings (``turns_list``).
    :param subset: Optional HuggingFace config/subset name (``name=`` kwarg).
    :param multi_turn: Whether the benchmark is multi-turn by default.
    :param license_note: Human-readable license string, surfaced in docs/errors.
    """

    source: str
    split: str
    mode: Literal["single", "turns_list"]
    text_column: str
    subset: str | None = None
    multi_turn: bool = False
    license_note: str = ""


# Mirrors the datasets AIPerf reads for these benchmarks (its ``spec_al_*``
# public-dataset entries and accuracy benchmark sources). One canonical name per
# benchmark; aliases are mapped in BENCHMARK_ALIASES below.
BENCHMARK_PRESETS: dict[str, BenchmarkPreset] = {
    "gsm8k": BenchmarkPreset(
        source="openai/gsm8k",
        subset="main",
        split="test",
        mode="single",
        text_column="question",
        license_note="MIT",
    ),
    "math500": BenchmarkPreset(
        source="HuggingFaceH4/MATH-500",
        split="test",
        mode="single",
        text_column="problem",
        license_note="MIT (inherited from Hendrycks MATH)",
    ),
    "humaneval": BenchmarkPreset(
        source="openai/openai_humaneval",
        split="test",
        mode="single",
        text_column="prompt",
        license_note="MIT",
    ),
    "mbpp": BenchmarkPreset(
        source="google-research-datasets/mbpp",
        subset="full",
        split="test",
        mode="single",
        text_column="text",
        license_note="CC-BY-4.0",
    ),
    "mt_bench": BenchmarkPreset(
        source="HuggingFaceH4/mt_bench_prompts",
        split="train",
        mode="turns_list",
        text_column="prompt",
        multi_turn=True,
        license_note="Apache-2.0",
    ),
}

# Accepted CLI aliases -> canonical preset key.
BENCHMARK_ALIASES: dict[str, str] = {
    "gsm8k": "gsm8k",
    "math500": "math500",
    "math_500": "math500",
    "humaneval": "humaneval",
    "human_eval": "humaneval",
    "mbpp": "mbpp",
    "mt_bench": "mt_bench",
    "mtbench": "mt_bench",
    "mt-bench": "mt_bench",
}

BenchmarkKind = Literal[
    "gsm8k",
    "math500",
    "math_500",
    "humaneval",
    "human_eval",
    "mbpp",
    "mt_bench",
    "mtbench",
    "mt-bench",
]


def resolve_benchmark_name(kind: str) -> str:
    """Map a CLI ``kind`` (canonical name or alias) to a canonical preset key.

    :param kind: The ``kind`` string supplied on ``--data``.
    :return: The canonical benchmark key present in :data:`BENCHMARK_PRESETS`.
    :raises KeyError: If ``kind`` names no known benchmark.
    """
    return BENCHMARK_ALIASES[kind]


@DataArgs.register(list(BENCHMARK_ALIASES.keys()))
class BenchmarkDataArgs(DataArgs):
    """Arguments for the named benchmark prompt-dataset deserializer.

    Selects one of the self-contained HuggingFace benchmark datasets that
    AIPerf exposes (GSM8K, MATH-500, HumanEval, MBPP, MT-Bench). The
    benchmark is chosen by ``kind``; every source/column detail comes from
    :data:`BENCHMARK_PRESETS`, so ``--data kind=gsm8k`` is sufficient.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    kind: BenchmarkKind = Field(  # type: ignore[assignment]
        default="gsm8k",
        description="Named benchmark to load (canonical name or alias).",
    )
    split: str | None = Field(
        default=None,
        description="Override the preset dataset split (defaults to the "
        "benchmark's standard evaluation split).",
    )
    subset: str | None = Field(
        default=None,
        description="Override the preset HuggingFace config/subset name.",
    )
    multi_turn: bool | None = Field(
        default=None,
        description="Override whether a multi-turn benchmark emits all turns. "
        "Only affects benchmarks whose rows carry multiple turns (MT-Bench).",
    )
    samples: int = Field(
        default=-1,
        description="Maximum number of rows to read; -1 reads the whole split.",
    )
