from __future__ import annotations

import importlib.util
import json
import os
import re
from collections.abc import Callable
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from datasets import Dataset
from datasets import load_dataset as hf_load_dataset
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
from guidellm.schemas.data.deserializers.speed_bench import (
    SPEED_BENCH_CONFIGS,
    SpeedBenchDataArgs,
)

__all__ = ["SpeedBenchDatasetDeserializer"]

# Pinned so the manifest rows -- and therefore the source URLs the vendored
# resolver fetches -- cannot change under a run. Matches the revision AIPerf
# pins; the resolver needs v1.1 source references for HLE/StackSelect prompts.
SPEED_BENCH_REVISION = "fa4e669bee2a4393c56aa8ed05a7c667897f9590"

# Placeholder emitted by nvidia/SPEED-Bench for rows whose source dataset does
# not permit redistribution; present until the resolver fills in the real text.
TURNS_PLACEHOLDER = (
    "FULL BENCHMARK DATA SHOULD BE FETCHED FROM THE SOURCE USING SPECDEC_BENCH"
)

HLE_ACCESS_URL = "https://huggingface.co/datasets/cais/hle"
_GATED_SOURCE = "cais/hle"

# Hosts the published dataset legitimately sources prompt text from. Row
# ``source`` values drive what the vendored resolver fetches, so anything
# outside this set is refused before a request is made.
_ALLOWED_SOURCE_HOSTS = frozenset(
    {
        "huggingface.co",
        "raw.githubusercontent.com",
        "github.com",
        "opencompass.openxlab.space",
        "www.gutenberg.org",
    }
)

_CONFIG_NAME_RE = re.compile(r"[A-Za-z0-9_-]+")


def _cache_root() -> Path:
    """Where resolved SPEED-Bench configs are cached.

    Honors ``GUIDELLM_DATASET_CACHE_DIR`` and otherwise falls back to
    ``~/.cache/guidellm``. The resolved config is large (several GB of source
    downloads land in the HuggingFace cache; the JSONL written here is small).
    """
    base = os.environ.get("GUIDELLM_DATASET_CACHE_DIR")
    root = Path(base) if base else Path.home() / ".cache" / "guidellm"
    return root / "speed-bench" / "v1.1"


@DatasetDeserializerFactory.register(["speed_bench", "speedbench", "speed-bench"])
class SpeedBenchDatasetDeserializer(DatasetDeserializer):
    """Deserializer for nvidia/SPEED-Bench with full source resolution.

    Mirrors AIPerf's SPEED-Bench support: resolve the published placeholder rows
    by running the vendored upstream prepare script against the original source
    datasets, cache the resolved config, filter by category, and emit single- or
    multi-turn conversations. A pre-resolved JSONL can be supplied via ``source``
    to skip resolution.
    """

    def __call__(
        self,
        config: SpeedBenchDataArgs,
        processor_factory: Callable[[], PreTrainedTokenizerBase],
        random_seed: int,
    ) -> Dataset:
        _ = (processor_factory, random_seed)

        if config.source:
            rows = self._read_rows(Path(config.source))
            self._reject_unresolved(str(config.source), rows)
        else:
            self._validate_config_name(config.config)
            cache_path = self.cache_path_for(config.config)
            if not cache_path.exists():
                cache_path = self.resolve_config(
                    config.config,
                    hf_dataset_name=config.hf_dataset_name,
                    hf_split=config.hf_split,
                    revision=config.revision or SPEED_BENCH_REVISION,
                )
            rows = self._read_rows(cache_path)
            self._reject_unresolved(config.config, rows)

        return self._to_dataset(rows, config)

    # -- normalization ------------------------------------------------------

    def _to_dataset(
        self, rows: list[dict[str, Any]], config: SpeedBenchDataArgs
    ) -> Dataset:
        total = 0
        single: list[str] = []
        multi: list[list[str]] = []

        for row in rows:
            total += 1
            if config.category and row.get("category") != config.category:
                continue
            if config.samples >= 0 and max(len(single), len(multi)) >= config.samples:
                break

            turns = normalize_turns(
                message.get("content", "") for message in row.get("messages", [])
            )
            if not turns:
                continue
            if config.multi_turn:
                multi.append(turns)
            else:
                single.append(turns[0])

        if config.category and not single and not multi:
            raise DataNotSupportedError(
                f"SPEED-Bench category {config.category!r} matched none of the "
                f"{total} rows in config {config.config!r}. Verify the category "
                f"exists in this split -- the qualitative and throughput splits "
                f"have different category names."
            )

        source_name = f"speed_bench:{config.config}"
        if config.multi_turn:
            return build_multi_turn_dataset(multi, source_name=source_name)
        return build_single_turn_dataset(single, source_name=source_name)

    # -- resolution ---------------------------------------------------------

    @classmethod
    def cache_path_for(cls, config: str) -> Path:
        cls._validate_config_name(config)
        return _cache_root() / f"{config}.jsonl"

    @classmethod
    def resolve_config(
        cls,
        config: str,
        *,
        hf_dataset_name: str = "nvidia/SPEED-Bench",
        hf_split: str = "test",
        revision: str = SPEED_BENCH_REVISION,
    ) -> Path:
        """Resolve ``config``'s prompt text and cache it; return the cache path.

        :raises DataNotSupportedError: If resolver dependencies are missing, a
            source host is untrusted, resolution fails, or rows remain
            unresolved after running the prepare script.
        """
        prepare = cls._import_resolver()

        cache_path = cls.cache_path_for(config)
        _cache_root().mkdir(parents=True, exist_ok=True)
        tmp_path = cache_path.with_suffix(f".jsonl.{os.getpid()}.partial")

        try:
            dataset = hf_load_dataset(
                hf_dataset_name, config, split=hf_split, revision=revision
            )
            cls._reject_untrusted_sources(dataset)
            dataset = prepare._resolve_external_data(dataset, config)  # noqa: SLF001
            dataset = dataset.map(
                lambda example: {
                    "messages": [
                        {"role": "user", "content": turn} for turn in example["turns"]
                    ]
                },
                remove_columns=["turns"],
            )
            dataset.to_json(str(tmp_path))
            rows = cls._read_rows(tmp_path)
            cls._reject_unresolved(config, rows, delete=tmp_path)
            tmp_path.replace(cache_path)
        except DataNotSupportedError:
            raise
        except Exception as err:
            raise DataNotSupportedError(
                cls._resolution_failed_message(config, err)
            ) from err
        finally:
            tmp_path.unlink(missing_ok=True)

        return cache_path

    @staticmethod
    def _import_resolver() -> Any:
        """Import the vendored resolver, checking its optional dependencies first.

        ``pandas`` and ``tiktoken`` are only needed for SPEED-Bench resolution,
        so they are an optional install rather than a core GuideLLM dependency.
        The vendored module is imported lazily so the rest of GuideLLM never
        pays for it.
        """
        missing = [
            name
            for name in ("numpy", "pandas", "tiktoken")
            if importlib.util.find_spec(name) is None
        ]
        if missing:
            raise DataNotSupportedError(
                f"Resolving SPEED-Bench requires the optional packages "
                f"{', '.join(missing)}. Install them with "
                f"'pip install guidellm[speed-bench]' (or 'pip install pandas "
                f"tiktoken'), or point --data at a pre-resolved JSONL via "
                f"source=<path>."
            )
        from guidellm.data.deserializers.vendor import (  # noqa: PLC0415
            speed_bench_prepare,
        )

        return speed_bench_prepare

    # -- validation ---------------------------------------------------------

    @staticmethod
    def _validate_config_name(config: str) -> None:
        if not _CONFIG_NAME_RE.fullmatch(config):
            raise DataNotSupportedError(
                f"Invalid SPEED-Bench config {config!r}. Expected a bare config "
                f"name such as 'qualitative' or 'throughput_1k' (letters, "
                f"digits, underscores and hyphens only)."
            )
        if config not in SPEED_BENCH_CONFIGS:
            raise DataNotSupportedError(
                f"Unknown SPEED-Bench config {config!r}. Valid configs: "
                f"{', '.join(SPEED_BENCH_CONFIGS)}."
            )

    @classmethod
    def _reject_untrusted_sources(cls, dataset: Any) -> None:
        column_names = getattr(dataset, "column_names", None) or []
        if "source" not in column_names:
            return
        offenders: set[str] = set()
        for source in dataset["source"]:
            if not source:
                continue
            host = urlparse(str(source)).netloc.split("@")[-1].split(":")[0]
            if host and host not in _ALLOWED_SOURCE_HOSTS:
                offenders.add(host)
        if offenders:
            raise DataNotSupportedError(
                f"SPEED-Bench rows reference unexpected source hosts: "
                f"{', '.join(sorted(offenders))}. Resolution fetches prompt text "
                f"from each row's source, so GuideLLM refuses hosts outside the "
                f"published set ({', '.join(sorted(_ALLOWED_SOURCE_HOSTS))})."
            )

    @classmethod
    def _reject_unresolved(
        cls, config: str, rows: list[dict[str, Any]], *, delete: Path | None = None
    ) -> None:
        unresolved = sum(1 for row in rows if cls._has_placeholder(row))
        if not unresolved:
            return
        if delete is not None:
            delete.unlink(missing_ok=True)
        raise DataNotSupportedError(
            f"SPEED-Bench '{config}': {unresolved} of {len(rows)} rows still "
            f"contain placeholder text instead of prompts. SPEED-Bench ships "
            f"without prompt text for rows whose source datasets do not permit "
            f"redistribution, and the prepare step reports success even when a "
            f"source is unreachable. GuideLLM will not benchmark placeholder "
            f"text. One source, {_GATED_SOURCE}, is gated: accept its terms at "
            f"{HLE_ACCESS_URL} (approval is automatic, and granted per user, not "
            f"per organization), then run 'huggingface-cli login' and retry."
        )

    @staticmethod
    def _has_placeholder(row: dict[str, Any]) -> bool:
        return any(
            str(message.get("content", "")).startswith(TURNS_PLACEHOLDER)
            for message in row.get("messages", [])
        )

    @classmethod
    def _resolution_failed_message(cls, config: str, error: Exception) -> str:
        text = str(error)
        if _GATED_SOURCE in text or "gated" in text.lower():
            return (
                f"SPEED-Bench '{config}' needs '{_GATED_SOURCE}', which is gated "
                f"on HuggingFace. Accept the terms at {HLE_ACCESS_URL} (approval "
                f"is automatic, granted per user), run 'huggingface-cli login', "
                f"and retry. Underlying error: {error}"
            )
        return (
            f"Failed to resolve SPEED-Bench '{config}' from its source datasets: "
            f"{error}"
        )

    # -- io -----------------------------------------------------------------

    @staticmethod
    def _read_rows(path: Path) -> list[dict[str, Any]]:
        if not path.exists():
            raise DataNotSupportedError(f"SPEED-Bench file not found: {path}.")
        with path.open(encoding="utf-8") as handle:
            return [json.loads(line) for line in handle if line.strip()]
