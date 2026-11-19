from __future__ import annotations

from collections.abc import Callable

from datasets import Dataset, load_dataset
from datasets.exceptions import (
    DataFilesNotFoundError,
    DatasetNotFoundError,
    FileNotFoundDatasetsError,
)
from transformers import PreTrainedTokenizerBase

from guidellm.data.deserializers._benchmark_common import (
    build_multi_turn_dataset,
    build_single_turn_dataset,
    normalize_turns,
)
from guidellm.data.deserializers.deserializer import (
    DataNotSupportedError,
    DatasetDeserializer,
    DatasetDeserializerFactory,
)
from guidellm.schemas.data.deserializers.benchmarks import (
    BENCHMARK_ALIASES,
    BENCHMARK_PRESETS,
    BenchmarkDataArgs,
    BenchmarkPreset,
    resolve_benchmark_name,
)

__all__ = ["BenchmarkDatasetDeserializer"]


@DatasetDeserializerFactory.register(list(BENCHMARK_ALIASES.keys()))
class BenchmarkDatasetDeserializer(DatasetDeserializer):
    """Deserializer for self-contained HuggingFace benchmark prompt datasets.

    Provides the same read-in behavior AIPerf offers for GSM8K, MATH-500,
    HumanEval, MBPP, and MT-Bench: the user names the benchmark via ``kind`` and
    the HuggingFace source, subset, split, and prompt column are supplied by the
    preset. Rows are normalized to the standard GuideLLM column shapes so the
    stock column mapper handles them without any extra configuration.
    """

    def __call__(
        self,
        config: BenchmarkDataArgs,
        processor_factory: Callable[[], PreTrainedTokenizerBase],
        random_seed: int,
    ) -> Dataset:
        _ = (processor_factory, random_seed)

        name = resolve_benchmark_name(config.kind)
        preset = BENCHMARK_PRESETS[name]

        split = config.split or preset.split
        subset = config.subset if config.subset is not None else preset.subset

        dataset = self._load(preset, subset=subset, split=split, config=config)

        if preset.mode == "single":
            return self._build_single_turn(dataset, preset, name, config.samples)

        multi_turn = (
            preset.multi_turn if config.multi_turn is None else config.multi_turn
        )
        return self._build_turns_list(
            dataset, preset, name, config.samples, multi_turn=multi_turn
        )

    @staticmethod
    def _load(
        preset: BenchmarkPreset,
        *,
        subset: str | None,
        split: str,
        config: BenchmarkDataArgs,
    ) -> Dataset:
        load_kwargs = dict(config.load_kwargs)
        if subset is not None:
            load_kwargs.setdefault("name", subset)
        try:
            return load_dataset(preset.source, split=split, **load_kwargs)
        except (
            FileNotFoundDatasetsError,
            DatasetNotFoundError,
            DataFilesNotFoundError,
        ) as err:
            raise DataNotSupportedError(
                f"Failed to load benchmark dataset '{preset.source}' "
                f"(split={split}, subset={subset}). If the dataset is gated, "
                f"authenticate with 'huggingface-cli login' and accept its terms "
                f"on the HuggingFace dataset page. HF error: {err}"
            ) from err

    @staticmethod
    def _within_samples(count: int, samples: int) -> bool:
        return samples < 0 or count < samples

    def _build_single_turn(
        self,
        dataset: Dataset,
        preset: BenchmarkPreset,
        name: str,
        samples: int,
    ) -> Dataset:
        self._require_column(dataset, preset.text_column, preset.source)
        prompts: list[str] = []
        for row in dataset:
            if not self._within_samples(len(prompts), samples):
                break
            value = row.get(preset.text_column)
            if value is None:
                continue
            text = str(value).strip()
            if text:
                prompts.append(text)
        return build_single_turn_dataset(prompts, source_name=name)

    def _build_turns_list(
        self,
        dataset: Dataset,
        preset: BenchmarkPreset,
        name: str,
        samples: int,
        *,
        multi_turn: bool,
    ) -> Dataset:
        self._require_column(dataset, preset.text_column, preset.source)

        if not multi_turn:
            prompts: list[str] = []
            for row in dataset:
                if not self._within_samples(len(prompts), samples):
                    break
                turns = normalize_turns(row.get(preset.text_column) or [])
                if turns:
                    prompts.append(turns[0])
            return build_single_turn_dataset(prompts, source_name=name)

        conversations: list[list[str]] = []
        for row in dataset:
            if not self._within_samples(len(conversations), samples):
                break
            turns = normalize_turns(row.get(preset.text_column) or [])
            if turns:
                conversations.append(turns)
        return build_multi_turn_dataset(conversations, source_name=name)

    @staticmethod
    def _require_column(dataset: Dataset, column: str, source: str) -> None:
        columns = dataset.column_names
        if columns is not None and column not in columns:
            raise DataNotSupportedError(
                f"Benchmark dataset '{source}' has no column '{column}'. "
                f"Available columns: {columns}."
            )
