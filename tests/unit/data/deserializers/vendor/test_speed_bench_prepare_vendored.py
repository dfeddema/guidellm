"""Offline integrity pin for the vendored SPEED-Bench prepare script."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

# sha256 of the verbatim NVIDIA-NeMo/Gym prepare.py at the pinned commit; see
# src/guidellm/data/deserializers/vendor/README.md.
EXPECTED_SHA256 = "640de89c585d6e7a7ad8a5e09ef3621c3471e459e992848d1387f43d7e47ec8a"

VENDORED_PATH = (
    Path(__file__).resolve().parents[5]
    / "src"
    / "guidellm"
    / "data"
    / "deserializers"
    / "vendor"
    / "speed_bench_prepare.py"
)


@pytest.mark.smoke
def test_vendored_prepare_script_is_unmodified():
    """The vendored resolver must stay byte-identical to the pinned upstream.

    Any local edit changes the sha256 and fails here, protecting the
    Apache-2.0 "unmodified" claim and the resolver's output guarantees.

    ## WRITTEN BY AI ##
    """
    assert VENDORED_PATH.is_file(), f"missing vendored file: {VENDORED_PATH}"
    digest = hashlib.sha256(VENDORED_PATH.read_bytes()).hexdigest()
    assert digest == EXPECTED_SHA256
