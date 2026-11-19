# Vendored third-party sources

Files here are checked in **verbatim** from upstream. Do not edit them.

Their job is to produce byte-identical output to upstream. A hand-adaptation
cannot be mechanically verified, so any fix belongs upstream, followed by a
re-vendor and a SHA bump.

## `speed_bench_prepare.py`

| | |
|---|---|
| Upstream | [`NVIDIA-NeMo/Gym`](https://github.com/NVIDIA-NeMo/Gym) `benchmarks/speed-bench/prepare.py` |
| Source URL | `https://raw.githubusercontent.com/NVIDIA-NeMo/Gym/3a9b9eb981ab7a3ea0ebd517cbd4fca1d586768b/benchmarks/speed-bench/prepare.py` |
| Pinned commit | `3a9b9eb981ab7a3ea0ebd517cbd4fca1d586768b` (authored 2026-11-04) |
| Retrieved | 2026-11-04 |
| sha256 | `640de89c585d6e7a7ad8a5e09ef3621c3471e459e992848d1387f43d7e47ec8a` |
| Licence | Apache-2.0, Copyright (c) 2026 NVIDIA CORPORATION (header retained verbatim) |

Unmodified. GuideLLM adds no changes; the Apache-2.0 §4(b) modification notice
does not apply.

Resolves the placeholder rows in `nvidia/SPEED-Bench` by refetching prompt text
from the upstream source datasets. Consumed by
`guidellm.data.deserializers.speed_bench`.

`tests/unit/data/deserializers/vendor/test_speed_bench_prepare_vendored.py` pins
the sha256 offline, so an accidental local edit fails CI.

### Provenance note

This vendored copy is byte-identical (matching sha256) to the same file as
vendored by NVIDIA `aiperf`
(`src/aiperf/dataset/loader/vendor/speed_bench_prepare.py`), which is itself the
verbatim NVIDIA-NeMo/Gym source at the pinned commit above.

The resolver retains one behavior worth knowing: the source dispatch in
`_fetch_all_turns_data` ends in `return example` with no terminal `else`, so an
unrecognised source silently yields placeholder text. GuideLLM detects this
after the fact by counting unresolved rows rather than trusting the exit status
(see `SpeedBenchDatasetDeserializer._reject_unresolved`).
