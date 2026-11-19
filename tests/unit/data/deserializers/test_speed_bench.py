"""Tests for the SPEED-Bench deserializer's local-file path (no network)."""

from __future__ import annotations

import json

import pytest

from guidellm.data.deserializers.speed_bench import (
    TURNS_PLACEHOLDER,
    SpeedBenchDatasetDeserializer,
)
from guidellm.data.schemas import DataNotSupportedError
from guidellm.data.schemas.conversation_graph_data import ConversationGraphData
from guidellm.schemas.data.deserializers.speed_bench import (
    SPEED_BENCH_CATEGORIES,
    SPEED_BENCH_CONFIGS,
    SpeedBenchDataArgs,
    categories_for_config,
)


@pytest.fixture
def deserializer():
    return SpeedBenchDatasetDeserializer()


def _write_jsonl(path, rows):
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
    return path


def _msg_row(question_id, category, contents):
    return {
        "question_id": question_id,
        "category": category,
        "messages": [{"role": "user", "content": c} for c in contents],
    }


@pytest.mark.smoke
def test_local_file_multi_turn_builds_conversation_graph(deserializer, tmp_path):
    """A pre-resolved JSONL becomes multi-turn conversation_turns graphs.

    ## WRITTEN BY AI ##
    """
    path = _write_jsonl(
        tmp_path / "qualitative.jsonl",
        [_msg_row("q1", "coding", ["write a func", "add a test"])],
    )
    result = deserializer(SpeedBenchDataArgs(source=str(path)), None, random_seed=0)
    assert result.column_names == ["conversation_turns"]
    graph = ConversationGraphData.model_validate_json(result["conversation_turns"][0])
    assert [t.columns["text_column"][0] for t in graph.turns] == [
        "write a func",
        "add a test",
    ]


@pytest.mark.sanity
def test_local_file_single_turn_takes_first_message(deserializer, tmp_path):
    """multi_turn=False uses only the first user message as a flat prompt.

    ## WRITTEN BY AI ##
    """
    path = _write_jsonl(
        tmp_path / "qualitative.jsonl",
        [_msg_row("q1", "math", ["first", "second"])],
    )
    result = deserializer(
        SpeedBenchDataArgs(source=str(path), multi_turn=False), None, random_seed=0
    )
    assert result.column_names == ["prompt"]
    assert result["prompt"] == ["first"]


@pytest.mark.sanity
def test_category_filter(deserializer, tmp_path):
    """Only rows matching the requested category are kept.

    ## WRITTEN BY AI ##
    """
    path = _write_jsonl(
        tmp_path / "qualitative.jsonl",
        [
            _msg_row("q1", "coding", ["c-prompt"]),
            _msg_row("q2", "math", ["m-prompt"]),
        ],
    )
    result = deserializer(
        SpeedBenchDataArgs(source=str(path), category="math", multi_turn=False),
        None,
        random_seed=0,
    )
    assert result["prompt"] == ["m-prompt"]


@pytest.mark.sanity
def test_category_filter_no_match_raises(deserializer, tmp_path):
    """A category matching no rows raises a clear error.

    ## WRITTEN BY AI ##
    """
    path = _write_jsonl(
        tmp_path / "qualitative.jsonl", [_msg_row("q1", "coding", ["x"])]
    )
    with pytest.raises(DataNotSupportedError, match="matched none"):
        deserializer(
            SpeedBenchDataArgs(source=str(path), category="nope"),
            None,
            random_seed=0,
        )


@pytest.mark.smoke
def test_placeholder_rows_are_rejected(deserializer, tmp_path):
    """Unresolved placeholder prompts are refused rather than benchmarked.

    ## WRITTEN BY AI ##
    """
    path = _write_jsonl(
        tmp_path / "qualitative.jsonl",
        [_msg_row("q1", "coding", [f"{TURNS_PLACEHOLDER} (hle row)"])],
    )
    with pytest.raises(DataNotSupportedError, match="placeholder"):
        deserializer(SpeedBenchDataArgs(source=str(path)), None, random_seed=0)


@pytest.mark.sanity
@pytest.mark.parametrize("bad", ["../escape", "a/b", "not a config"])
def test_invalid_config_name_rejected(deserializer, bad):
    """A config name that is not a bare token is rejected before any IO.

    ## WRITTEN BY AI ##
    """
    with pytest.raises(DataNotSupportedError):
        SpeedBenchDatasetDeserializer._validate_config_name(bad)


@pytest.mark.sanity
def test_unknown_config_name_rejected(deserializer):
    """A syntactically valid but unknown config is rejected with valid options.

    ## WRITTEN BY AI ##
    """
    with pytest.raises(DataNotSupportedError, match="Valid configs"):
        SpeedBenchDatasetDeserializer._validate_config_name("throughput_4k")


@pytest.mark.sanity
def test_category_enumeration_covers_every_config():
    """Every published config exposes a non-empty category list.

    ## WRITTEN BY AI ##
    """
    assert set(SPEED_BENCH_CATEGORIES) == set(SPEED_BENCH_CONFIGS)
    assert all(SPEED_BENCH_CATEGORIES[c] for c in SPEED_BENCH_CONFIGS)
    assert "coding" in categories_for_config("qualitative")
    assert categories_for_config("throughput_1k") == (
        "low_entropy",
        "mixed",
        "high_entropy",
    )
    assert categories_for_config("unknown_config") == ()


@pytest.mark.smoke
def test_unknown_category_rejected_before_resolve():
    """A category not in the split is rejected early, listing valid categories.

    ## WRITTEN BY AI ##
    """
    with pytest.raises(DataNotSupportedError, match="Valid categories"):
        SpeedBenchDatasetDeserializer._validate_category("qualitative", "nope")


@pytest.mark.sanity
@pytest.mark.parametrize(
    ("config", "category"),
    [("qualitative", "coding"), ("throughput_1k", "mixed")],
)
def test_valid_category_passes_early_validation(config, category):
    """A category that exists in the split passes early validation (no raise).

    ## WRITTEN BY AI ##
    """
    SpeedBenchDatasetDeserializer._validate_category(config, category)


@pytest.mark.sanity
def test_no_category_skips_validation():
    """Omitting the category is always valid (full-split run).

    ## WRITTEN BY AI ##
    """
    SpeedBenchDatasetDeserializer._validate_category("qualitative", None)
