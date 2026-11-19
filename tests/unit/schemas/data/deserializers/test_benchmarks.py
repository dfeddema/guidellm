"""Tests for the named benchmark DataArgs schema and preset table."""

from __future__ import annotations

import pytest

from guidellm.schemas.data import DataArgs
from guidellm.schemas.data.deserializers.benchmarks import (
    BENCHMARK_ALIASES,
    BENCHMARK_PRESETS,
    BenchmarkDataArgs,
    resolve_benchmark_name,
)
from guidellm.schemas.data.deserializers.speed_bench import (
    SPEED_BENCH_CONFIGS,
    SpeedBenchDataArgs,
)


@pytest.mark.smoke
def test_preset_table_covers_five_hf_benchmarks():
    """The preset table defines exactly the five self-contained HF benchmarks.

    ## WRITTEN BY AI ##
    """
    assert set(BENCHMARK_PRESETS) == {
        "gsm8k",
        "math500",
        "humaneval",
        "mbpp",
        "mt_bench",
    }
    assert BENCHMARK_PRESETS["mt_bench"].multi_turn is True
    assert all(
        not p.multi_turn for name, p in BENCHMARK_PRESETS.items() if name != "mt_bench"
    )


@pytest.mark.smoke
@pytest.mark.parametrize(
    ("alias", "canonical"),
    [
        ("gsm8k", "gsm8k"),
        ("math_500", "math500"),
        ("math500", "math500"),
        ("human_eval", "humaneval"),
        ("mtbench", "mt_bench"),
        ("mt-bench", "mt_bench"),
    ],
)
def test_resolve_benchmark_name_maps_aliases(alias, canonical):
    """Every registered alias resolves to a canonical preset key.

    ## WRITTEN BY AI ##
    """
    assert resolve_benchmark_name(alias) == canonical
    assert canonical in BENCHMARK_PRESETS


@pytest.mark.sanity
@pytest.mark.parametrize("kind", list(BENCHMARK_ALIASES.keys()))
def test_dataargs_polymorphic_routes_benchmark_kind(kind):
    """A bare ``{"kind": <alias>}`` validates to BenchmarkDataArgs.

    ## WRITTEN BY AI ##
    """
    args = DataArgs.model_validate({"kind": kind})
    assert isinstance(args, BenchmarkDataArgs)
    assert args.kind == kind


@pytest.mark.sanity
def test_dataargs_polymorphic_routes_speed_bench():
    """A ``speed_bench`` kind validates to SpeedBenchDataArgs with defaults.

    ## WRITTEN BY AI ##
    """
    args = DataArgs.model_validate({"kind": "speed_bench"})
    assert isinstance(args, SpeedBenchDataArgs)
    assert args.config == "qualitative"
    assert args.multi_turn is True
    assert "qualitative" in SPEED_BENCH_CONFIGS


@pytest.mark.sanity
def test_speed_bench_config_and_source_aliases():
    """``subset``/``hf_subset`` alias ``config``; ``path``/``file`` alias source.

    ## WRITTEN BY AI ##
    """
    assert (
        DataArgs.model_validate(
            {"kind": "speed_bench", "subset": "throughput_1k"}
        ).config
        == "throughput_1k"
    )
    assert (
        DataArgs.model_validate({"kind": "speed_bench", "path": "data/x.jsonl"}).source
        == "data/x.jsonl"
    )
