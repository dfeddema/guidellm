"""Shared normalization helpers for the named benchmark deserializers.

Both the HuggingFace benchmark presets (GSM8K, MATH-500, HumanEval, MBPP,
MT-Bench) and SPEED-Bench converge on the same two output shapes so the stock
``generative_column_mapper`` consumes them with zero extra configuration:

* single-turn -> a dataset with one ``prompt`` string column
  (mapped as ``text_column`` turn 0), and
* multi-turn  -> a dataset with one ``conversation_turns`` column, each value a
  serialized :class:`ConversationGraphData` (mapped as
  ``conversation_turns_column``), which carries variable-length conversations
  without the uniform-schema constraints of turn-suffixed columns.

Keeping this in one place means the benchmark deserializers hold only their
source-specific loading logic, and nothing here touches the rest of GuideLLM.
"""

from __future__ import annotations

from collections.abc import Iterable

from datasets import Dataset

from guidellm.data.schemas import DataNotSupportedError
from guidellm.data.schemas.conversation_graph_data import (
    ConversationGraphData,
    ConversationParentRef,
    ConversationTurnData,
)

__all__ = [
    "build_multi_turn_dataset",
    "build_single_turn_dataset",
    "conversation_graph_json",
]


def _clean_turns(turns: Iterable[object]) -> list[str]:
    """Stringify, strip, and drop empty entries from a row's turns."""
    cleaned: list[str] = []
    for turn in turns:
        if turn is None:
            continue
        text = str(turn).strip()
        if text:
            cleaned.append(text)
    return cleaned


def conversation_graph_json(user_turns: list[str]) -> str:
    """Serialize sequential user turns into a ConversationGraphData JSON string.

    Each turn depends on the previous one with full history context, which is
    the sequential multi-turn chat protocol (turn N sees turns 0..N-1 and their
    responses) used by MT-Bench and SPEED-Bench.

    :param user_turns: Non-empty, ordered user-turn prompt strings.
    :return: A JSON string parseable by the generative finalizer's
        ``conversation_turns`` handling.
    """
    turns: list[ConversationTurnData] = []
    for index, text in enumerate(user_turns):
        parents = (
            [ConversationParentRef(parent_node_id=f"turn_{index - 1}")]
            if index > 0
            else []
        )
        turns.append(
            ConversationTurnData(
                node_id=f"turn_{index}",
                parents=parents,
                columns={"text_column": [text]},
            )
        )
    return ConversationGraphData(turns=turns).model_dump_json()


def build_single_turn_dataset(prompts: list[str], *, source_name: str) -> Dataset:
    """Build a one-column (``prompt``) dataset from single-turn prompt strings.

    :param prompts: Prompt strings, one per conversation.
    :param source_name: Benchmark/source name for error messages.
    :raises DataNotSupportedError: If no non-empty prompts were produced.
    """
    if not prompts:
        raise DataNotSupportedError(
            f"'{source_name}' produced no non-empty prompts after normalization."
        )
    return Dataset.from_dict({"prompt": prompts})


def build_multi_turn_dataset(
    conversations: list[list[str]], *, source_name: str
) -> Dataset:
    """Build a one-column (``conversation_turns``) dataset for multi-turn rows.

    :param conversations: One list of ordered user-turn strings per conversation.
    :param source_name: Benchmark/source name for error messages.
    :raises DataNotSupportedError: If no non-empty conversations were produced.
    """
    graphs = [conversation_graph_json(turns) for turns in conversations if turns]
    if not graphs:
        raise DataNotSupportedError(
            f"'{source_name}' produced no non-empty conversations after normalization."
        )
    return Dataset.from_dict({"conversation_turns": graphs})


def normalize_turns(turns: Iterable[object]) -> list[str]:
    """Public wrapper around :func:`_clean_turns` for the deserializers."""
    return _clean_turns(turns)
