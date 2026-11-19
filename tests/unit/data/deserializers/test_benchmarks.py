"""Tests for the named benchmark HuggingFace deserializer (no network)."""

from __future__ import annotations

import pytest
from datasets import Dataset

from guidellm.data.deserializers import benchmarks as benchmarks_module
from guidellm.data.deserializers.benchmarks import BenchmarkDatasetDeserializer
from guidellm.data.schemas import DataNotSupportedError
from guidellm.data.schemas.conversation_graph_data import ConversationGraphData
from guidellm.schemas.data.deserializers.benchmarks import BenchmarkDataArgs


@pytest.fixture
def deserializer():
    return BenchmarkDatasetDeserializer()


def _patch_load(monkeypatch, dataset: Dataset):
    captured: dict[str, object] = {}

    def fake_load_dataset(source, *, split, **kwargs):
        captured["source"] = source
        captured["split"] = split
        captured.update(kwargs)
        return dataset

    monkeypatch.setattr(benchmarks_module, "load_dataset", fake_load_dataset)
    return captured


@pytest.mark.smoke
def test_single_turn_benchmark_normalizes_to_prompt_column(monkeypatch, deserializer):
    """GSM8K's ``question`` column becomes a single ``prompt`` column.

    ## WRITTEN BY AI ##
    """
    captured = _patch_load(
        monkeypatch,
        Dataset.from_dict({"question": ["2+2?", "3+3?"], "answer": ["4", "6"]}),
    )
    result = deserializer(BenchmarkDataArgs(kind="gsm8k"), None, random_seed=0)

    assert captured["source"] == "openai/gsm8k"
    assert captured["split"] == "test"
    assert captured["name"] == "main"
    assert result.column_names == ["prompt"]
    assert result["prompt"] == ["2+2?", "3+3?"]


@pytest.mark.smoke
def test_math500_uses_problem_column(monkeypatch, deserializer):
    """MATH-500's non-default ``problem`` column is picked up by the preset.

    ## WRITTEN BY AI ##
    """
    _patch_load(monkeypatch, Dataset.from_dict({"problem": ["p1", "p2"]}))
    result = deserializer(BenchmarkDataArgs(kind="math500"), None, random_seed=0)
    assert result["prompt"] == ["p1", "p2"]


@pytest.mark.sanity
def test_samples_caps_rows(monkeypatch, deserializer):
    """``samples`` limits the number of normalized rows.

    ## WRITTEN BY AI ##
    """
    _patch_load(monkeypatch, Dataset.from_dict({"text": ["a", "b", "c", "d"]}))
    result = deserializer(
        BenchmarkDataArgs(kind="mbpp", samples=2), None, random_seed=0
    )
    assert result["prompt"] == ["a", "b"]


@pytest.mark.smoke
def test_mt_bench_multi_turn_emits_conversation_graph(monkeypatch, deserializer):
    """MT-Bench rows become multi-turn conversation_turns graphs.

    ## WRITTEN BY AI ##
    """
    _patch_load(
        monkeypatch,
        Dataset.from_dict({"prompt": [["turn one", "turn two"], ["solo"]]}),
    )
    result = deserializer(BenchmarkDataArgs(kind="mt_bench"), None, random_seed=0)

    assert result.column_names == ["conversation_turns"]
    first = ConversationGraphData.model_validate_json(result["conversation_turns"][0])
    texts = [t.columns["text_column"][0] for t in first.turns]
    assert texts == ["turn one", "turn two"]
    # Turn 1 depends on turn 0 (sequential history).
    assert first.turns[1].parents[0].parent_node_id == first.turns[0].node_id


@pytest.mark.sanity
def test_mt_bench_single_turn_override(monkeypatch, deserializer):
    """multi_turn=False collapses MT-Bench rows to the first user turn.

    ## WRITTEN BY AI ##
    """
    _patch_load(
        monkeypatch,
        Dataset.from_dict({"prompt": [["turn one", "turn two"]]}),
    )
    result = deserializer(
        BenchmarkDataArgs(kind="mt_bench", multi_turn=False), None, random_seed=0
    )
    assert result.column_names == ["prompt"]
    assert result["prompt"] == ["turn one"]


@pytest.mark.sanity
def test_missing_prompt_column_raises(monkeypatch, deserializer):
    """A dataset missing the preset's text column raises a clear error.

    ## WRITTEN BY AI ##
    """
    _patch_load(monkeypatch, Dataset.from_dict({"unexpected": ["x"]}))
    with pytest.raises(DataNotSupportedError):
        deserializer(BenchmarkDataArgs(kind="gsm8k"), None, random_seed=0)
