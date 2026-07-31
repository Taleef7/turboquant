"""Contracts for the local TurboQuant codec and retrieval reproduction."""

from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
import importlib
import json
import math
from pathlib import Path
import subprocess
import sys
import tomllib

import numpy as np
import pytest
import torch

from core.paper_mse import TurboQuantMSEConfig, TurboQuantMSEReference
from core.paper_product import (
    TurboQuantProductConfig,
    TurboQuantProductReference,
)
from scripts.benchmark_schema import validate_v2_artifact


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs" / "benchmarks" / "codec-local-v1.json"
RUNNER_PATH = ROOT / "scripts" / "run_codec_reproduction.py"
EMPTY_SHA256 = hashlib.sha256(b"").hexdigest()


def _runner():
    """Import lazily so a missing runner is a genuine test failure, not collection noise."""
    return importlib.import_module("scripts.run_codec_reproduction")


def _config() -> dict[str, object]:
    return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))


def _expected_row_ids() -> set[str]:
    synthetic = {
        f"syn-{distribution}-d{dimension}-mse-b{bits}-s{pair_id:02d}"
        for distribution in ("sphere", "gaussian", "heavy_tail", "outlier")
        for dimension in (64, 128, 256)
        for bits in (1, 2, 3, 4)
        for pair_id in range(16)
    }
    synthetic.update(
        {
            f"syn-{distribution}-d{dimension}-product-b{bits}-s{pair_id:02d}"
            for distribution in ("sphere", "gaussian", "heavy_tail", "outlier")
            for dimension in (64, 128, 256)
            for bits in (2, 3, 4)
            for pair_id in range(16)
        }
    )
    glove = {
        f"glove200-{codec}-b{bits}-s{pair_id:02d}"
        for codec in ("mse", "product")
        for bits in (2, 4)
        for pair_id in range(16)
    }
    return synthetic | glove


def test_checked_in_config_materializes_independent_seed_schedules() -> None:
    config = _config()
    pairs = config["seed_pairs"]

    assert pairs == [
        {
            "pair_id": pair_id,
            "rotation_seed": 10_000 + pair_id,
            "projection_seed": 1_000_000 + pair_id,
        }
        for pair_id in range(16)
    ]
    assert all(pair["rotation_seed"] != pair["projection_seed"] for pair in pairs)

    fixtures = config["synthetic"]["fixtures"]
    assert len(fixtures) == 12
    assert [fixture["data_seed"] for fixture in fixtures] == list(
        range(3_000_000, 3_000_012)
    )
    assert [fixture["independent_query_seed"] for fixture in fixtures] == list(
        range(4_000_000, 4_000_012)
    )
    assert [
        fixture["high_inner_product_noise_seed"] for fixture in fixtures
    ] == list(range(5_000_000, 5_000_012))


def test_checked_in_config_declares_formulas_and_pinned_glove_input() -> None:
    config = _config()
    definitions = {
        distribution["name"]: distribution["definition"]
        for distribution in config["synthetic"]["distributions"]
    }

    assert definitions == {
        "sphere": "normalized_iid_gaussian_rows",
        "gaussian": "iid_normal",
        "heavy_tail": "iid_student_t",
        "outlier": "(z + 8*s*e_j)/sqrt(65)",
    }
    glove = config["glove200"]
    assert glove["sha256"] == (
        "4839085e5a8bb293434a1a66e1aa0193afc3f07c6797a85f1dbd91656172da20"
    )
    assert glove["size_bytes"] == 962_819_488
    assert glove["datasets"] == {
        "train": {"shape": [1_183_514, 200]},
        "test": {"shape": [10_000, 200]},
        "neighbors": {"shape": [10_000, 100]},
    }


def test_retrieval_dependency_is_optional_and_runtime_certified() -> None:
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    runtime_input = (ROOT / "requirements" / "runtime.in").read_text(
        encoding="utf-8"
    )

    assert project["project"]["optional-dependencies"]["retrieval"] == [
        "h5py>=3.16,<4"
    ]
    assert "h5py\n" in runtime_input


def test_execution_class_records_cpu_timing_and_lock_reproducibility() -> None:
    runner = _runner()
    execution = runner._execution_class(_config())
    extensions = execution["extensions"]

    assert extensions["environment"]["python_version"]
    assert extensions["environment"]["torch_version"]
    assert extensions["h5py_version"] == "3.16.0"
    assert extensions["runtime_lock_sha256"] == hashlib.sha256(
        (ROOT / "requirements" / "runtime.lock").read_bytes()
    ).hexdigest()
    assert extensions["cpu_model"]
    assert extensions["operating_system"]
    assert extensions["torch_intraop_threads"] > 0
    assert extensions["torch_interop_threads"] > 0
    assert extensions["timing_policy"] == {
        "codec_setup_excluded": True,
        "fixture_generation_excluded": True,
        "base_device_transfer_in_retrieval_scoring": True,
        "query_device_transfer_in_retrieval_scoring": False,
    }


def test_runner_is_materialized_before_contract_import() -> None:
    assert RUNNER_PATH.is_file(), "Task 6 runner has not been implemented"


def test_exact_canonical_matrix_has_1408_unique_rows_and_reuses_data() -> None:
    runner = _runner()
    specs = runner.build_row_specs(_config())
    row_ids = [spec["row_id"] for spec in specs]

    assert len(row_ids) == 1_408
    assert len(set(row_ids)) == 1_408
    assert set(row_ids) == _expected_row_ids()
    assert sum(row_id.startswith("syn-") for row_id in row_ids) == 1_344
    assert sum(row_id.startswith("glove200-") for row_id in row_ids) == 64

    fixture_streams: dict[tuple[str, int], set[tuple[int, int, int]]] = defaultdict(
        set
    )
    for spec in specs:
        if spec["dataset"] == "synthetic":
            fixture_streams[(spec["distribution"], spec["dimension"])].add(
                (
                    spec["data_seed"],
                    spec["independent_query_seed"],
                    spec["high_inner_product_noise_seed"],
                )
            )
    assert len(fixture_streams) == 12
    assert all(len(streams) == 1 for streams in fixture_streams.values())


@pytest.mark.parametrize(
    "distribution", ("sphere", "gaussian", "heavy_tail", "outlier")
)
def test_distribution_generators_match_predeclared_moments(
    distribution: str,
) -> None:
    runner = _runner()
    fixture = runner.generate_synthetic_fixture(
        distribution=distribution,
        dimension=64,
        count=20_000,
        data_seed=3_123_456,
        independent_query_seed=4_123_456,
        high_inner_product_noise_seed=5_123_456,
    )
    vectors = fixture["vectors"]
    squared_norms = np.square(vectors, dtype=np.float64).sum(axis=1)

    assert vectors.shape == (20_000, 64)
    assert vectors.dtype == np.float32
    assert abs(float(vectors.mean())) < 0.01
    if distribution == "sphere":
        np.testing.assert_allclose(squared_norms, 1.0, atol=3e-6, rtol=0)
    else:
        assert 0.88 < float(squared_norms.mean()) < 1.12
    if distribution == "heavy_tail":
        gaussian_coordinate_fourth_moment = 3.0 / (64**2)
        assert float(np.mean(vectors[:, 0].astype(np.float64) ** 4)) > (
            2.0 * gaussian_coordinate_fourth_moment
        )
    if distribution == "outlier":
        assert float(np.quantile(np.abs(vectors), 0.999)) > 0.8


def test_query_fixtures_are_unit_length_independent_and_high_ip() -> None:
    runner = _runner()
    fixture = runner.generate_synthetic_fixture(
        distribution="gaussian",
        dimension=64,
        count=4_096,
        data_seed=3_000_003,
        independent_query_seed=4_000_003,
        high_inner_product_noise_seed=5_000_003,
    )
    vectors = fixture["vectors"].astype(np.float64)
    independent = fixture["independent_queries"].astype(np.float64)
    high_ip = fixture["high_inner_product_queries"].astype(np.float64)
    unit_vectors = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)

    np.testing.assert_allclose(
        np.linalg.norm(independent, axis=1), 1.0, atol=2e-6, rtol=0
    )
    np.testing.assert_allclose(
        np.linalg.norm(high_ip, axis=1), 1.0, atol=2e-6, rtol=0
    )
    np.testing.assert_allclose(
        np.sum(unit_vectors * high_ip, axis=1), 0.8, atol=2e-6, rtol=0
    )
    assert abs(float(np.mean(np.sum(unit_vectors * independent, axis=1)))) < 0.01


def test_codec_identity_uses_only_honest_codec_sentinels() -> None:
    runner = _runner()
    identity = runner.codec_identity(CONFIG_PATH)
    config_sha = hashlib.sha256(CONFIG_PATH.read_bytes()).hexdigest()

    assert identity == {
        "model_id": "not_applicable:codec-only",
        "model_revision": "not_applicable:codec-only",
        "weights_sha256": EMPTY_SHA256,
        "profile_id": "codec-local-v1",
        "profile_sha256": config_sha,
        "kv_dtype": "float32",
    }


@pytest.mark.parametrize(
    ("codec_name", "dimension", "bits", "count"),
    (
        ("mse", 64, 1, 3),
        ("mse", 128, 4, 5),
        ("product", 64, 2, 3),
        ("product", 256, 4, 5),
    ),
)
def test_physical_byte_formula_matches_actual_packed_payload(
    codec_name: str, dimension: int, bits: int, count: int
) -> None:
    runner = _runner()
    vectors = torch.linspace(
        -1.0, 1.0, count * dimension, dtype=torch.float32
    ).reshape(count, dimension)
    if codec_name == "mse":
        codec = TurboQuantMSEReference(
            TurboQuantMSEConfig(dimension, bits, 10_000)
        )
    else:
        codec = TurboQuantProductReference(
            TurboQuantProductConfig(dimension, bits, 10_000, 1_000_000)
        )
    encoded = codec.encode(vectors)

    assert codec.physical_nbytes(encoded) == runner.physical_bytes_formula(
        codec_name, dimension, bits, count
    )


def test_error_summary_uses_sample_sd_sem_and_student_t_interval() -> None:
    runner = _runner()
    summary = runner.summarize_signed_errors(np.array([1.0, 2.0, 3.0]))

    assert summary["count"] == 3
    assert summary["rmse"] == pytest.approx(math.sqrt(14.0 / 3.0))
    assert summary["signed_bias"] == 2.0
    assert summary["sample_standard_deviation"] == 1.0
    assert summary["sem"] == pytest.approx(1.0 / math.sqrt(3.0))
    assert summary["student_t_95_ci"] == pytest.approx(
        [2.0 - 4.302652729749 * math.sqrt(1.0 / 3.0),
         2.0 + 4.302652729749 * math.sqrt(1.0 / 3.0)]
    )


@pytest.mark.parametrize(
    "values",
    (
        np.array([1.0]),
        np.array([1.0, np.nan]),
        np.array([1.0, np.inf]),
    ),
)
def test_error_summary_rejects_invalid_student_t_samples(
    values: np.ndarray,
) -> None:
    with pytest.raises(ValueError, match="finite|at least two"):
        _runner().summarize_signed_errors(values)


def test_product_expectation_uses_finite_float32_residual_norm() -> None:
    runner = _runner()
    vectors = torch.tensor([[1.0, 2.0], [-0.5, 0.25]], dtype=torch.float64)
    mse_decoded = torch.tensor([[0.75, 1.75], [-0.25, 0.0]], dtype=torch.float64)
    queries = torch.tensor([[0.6, 0.8], [1.0, 0.0]], dtype=torch.float64)
    residual = vectors - mse_decoded
    stored = torch.linalg.vector_norm(residual, dim=-1).to(torch.float32)
    expected_vectors = mse_decoded + residual * (
        stored.to(torch.float64)
        / torch.linalg.vector_norm(residual, dim=-1)
    ).unsqueeze(-1)

    actual = runner.product_float32_residual_expectation(
        vectors, mse_decoded, stored, queries
    )

    torch.testing.assert_close(
        actual, (queries * expected_vectors).sum(dim=-1), atol=0, rtol=0
    )


def test_product_expectation_handles_exact_zero_residual() -> None:
    runner = _runner()
    vectors = torch.tensor([[1.0, -2.0]], dtype=torch.float64)
    query = torch.tensor([[0.6, 0.8]], dtype=torch.float64)

    actual = runner.product_float32_residual_expectation(
        vectors,
        vectors.clone(),
        torch.zeros(1, dtype=torch.float32),
        query,
    )

    torch.testing.assert_close(actual, (query * vectors).sum(dim=-1))


@pytest.mark.parametrize(
    ("states", "dirty", "expected"),
    (
        (["passed"], False, "qualified"),
        (["passed"], True, "partial"),
        (["passed", "unsupported"], False, "partial"),
        (["passed", "unavailable"], False, "partial"),
        (["passed", "failed"], False, "failed"),
        (["passed", "oom", "unavailable"], False, "failed"),
        (["passed", "invalid", "failed", "unavailable"], False, "invalid"),
    ),
)
def test_artifact_status_uses_complete_precedence(
    states: list[str], dirty: bool, expected: str
) -> None:
    assert _runner().artifact_status(states, dirty_source=dirty) == expected


def test_resume_requires_terminal_state_and_exact_complete_fingerprint() -> None:
    runner = _runner()
    fingerprint = {
        "row_spec_sha256": "1" * 64,
        "config_sha256": "2" * 64,
        "input_sha256": {"dataset": "3" * 64},
    }
    row = {"status": "passed", "extensions": {"resume_fingerprint": fingerprint}}

    assert runner.can_resume_row(row, fingerprint)
    for status in ("failed", "unsupported"):
        assert runner.can_resume_row({**row, "status": status}, fingerprint)
    for status in ("invalid", "oom", "unavailable"):
        assert not runner.can_resume_row({**row, "status": status}, fingerprint)
    changed = json.loads(json.dumps(fingerprint))
    changed["input_sha256"]["dataset"] = "4" * 64
    assert not runner.can_resume_row(row, changed)


def test_atomic_rows_and_rebuild_are_deterministic_with_missing_placeholder(
    tmp_path: Path,
) -> None:
    runner = _runner()
    row_dir = tmp_path / "rows"
    identity = runner.codec_identity(CONFIG_PATH)
    first = {
        "row_id": "row-b",
        "status": "passed",
        "reason": None,
        "identity": identity,
        "metrics": {"mse": 0.25},
        "sidecars": [],
        "extensions": {},
    }
    runner.atomic_write_json(row_dir / "row-b.json", first)
    specs = [{"row_id": "row-a"}, {"row_id": "row-b"}]

    rows = runner.collect_rows(row_dir, specs, identity)
    assert [row["row_id"] for row in rows] == ["row-a", "row-b"]
    assert rows[0]["status"] == "invalid"
    assert rows[0]["reason"] == "missing expected row file"
    assert not list(tmp_path.rglob("*.tmp"))


def test_malformed_row_file_is_preserved_as_an_invalid_placeholder(
    tmp_path: Path,
) -> None:
    runner = _runner()
    row_dir = tmp_path / "rows"
    row_dir.mkdir()
    (row_dir / "row-a.json").write_text('{"row_id":"row-a",', encoding="utf-8")
    identity = runner.codec_identity(CONFIG_PATH)

    rows = runner.collect_rows(row_dir, [{"row_id": "row-a"}], identity)

    assert rows[0]["status"] == "invalid"
    assert rows[0]["reason"].startswith("malformed row file:")

    rows_jsonl = tmp_path / "rows.jsonl"
    runner.atomic_write_jsonl(rows_jsonl, rows)
    first_bytes = rows_jsonl.read_bytes()
    runner.atomic_write_jsonl(rows_jsonl, reversed(rows))
    assert rows_jsonl.read_bytes() == first_bytes
    assert not list(tmp_path.rglob("*.tmp"))


def test_glove_selection_ids_have_pinned_little_endian_hashes() -> None:
    runner = _runner()
    base_ids, query_ids = runner.glove_selection_ids(_config()["glove200"])

    assert base_ids.dtype == np.dtype("<u4")
    assert query_ids.dtype == np.dtype("<u4")
    assert base_ids.shape == (100_000,)
    assert query_ids.shape == (10_000,)
    assert np.all(base_ids[:-1] < base_ids[1:])
    assert hashlib.sha256(base_ids.tobytes()).hexdigest() == (
        "4465179f0175fbc0a43621e61c5472bff7c70361a7512b34245ff3e47b135c40"
    )
    assert hashlib.sha256(query_ids.tobytes()).hexdigest() == (
        "9140e019602b8628f6f4a6aac3658bf206e332a92943eb113fb2b465fecc55d6"
    )


def test_glove_validation_checks_digest_size_shapes_and_preserves_file(
    tmp_path: Path,
) -> None:
    h5py = pytest.importorskip("h5py")
    runner = _runner()
    path = tmp_path / "fixture.hdf5"
    with h5py.File(path, "w") as handle:
        handle.create_dataset("train", shape=(7, 3), dtype="f4")
        handle.create_dataset("test", shape=(5, 3), dtype="f4")
        handle.create_dataset("neighbors", shape=(5, 2), dtype="i4")
        handle.attrs["distance"] = "angular"
    before = path.read_bytes()
    expected = {
        "sha256": hashlib.sha256(before).hexdigest(),
        "size_bytes": len(before),
        "datasets": {
            "train": {"shape": [7, 3]},
            "test": {"shape": [5, 3]},
            "neighbors": {"shape": [5, 2]},
        },
    }

    metadata = runner.validate_glove_file(path, expected)

    assert metadata["datasets"]["train"]["dtype"] == "float32"
    assert metadata["attributes"]["distance"] == "angular"
    assert path.read_bytes() == before
    with pytest.raises(ValueError, match="SHA-256"):
        runner.validate_glove_file(path, {**expected, "sha256": "0" * 64})
    with pytest.raises(ValueError, match="size mismatch"):
        runner.validate_glove_file(path, {**expected, "size_bytes": len(before) + 1})
    wrong_shape = json.loads(json.dumps(expected))
    wrong_shape["datasets"]["train"]["shape"] = [8, 3]
    with pytest.raises(ValueError, match="train shape mismatch"):
        runner.validate_glove_file(path, wrong_shape)

    missing_path = tmp_path / "missing-dataset.hdf5"
    with h5py.File(missing_path, "w") as handle:
        handle.create_dataset("train", shape=(7, 3), dtype="f4")
        handle.create_dataset("test", shape=(5, 3), dtype="f4")
    missing_bytes = missing_path.read_bytes()
    missing_expected = {
        **expected,
        "sha256": hashlib.sha256(missing_bytes).hexdigest(),
        "size_bytes": len(missing_bytes),
    }
    with pytest.raises(ValueError, match="dataset is missing: neighbors"):
        runner.validate_glove_file(missing_path, missing_expected)


def test_one_at_k_is_true_top_one_membership_in_approximate_prefix() -> None:
    runner = _runner()
    true_top_one = np.array([7, 3, 9, 4], dtype=np.int64)
    approximate = np.array(
        [
            [7, 1, 2, 3],
            [8, 3, 2, 1],
            [8, 7, 6, 9],
            [0, 1, 2, 3],
        ],
        dtype=np.int64,
    )

    assert runner.one_at_k(true_top_one, approximate, (1, 2, 4)) == {
        "1@1": 0.25,
        "1@2": 0.5,
        "1@4": 0.75,
    }


def test_retrieval_scoring_materializes_only_declared_base_chunks() -> None:
    runner = _runner()
    base = np.eye(5, dtype=np.float32)
    queries = np.eye(5, dtype=np.float32)[:3]
    observed_block_sizes: list[int] = []

    def identity_transform(block: np.ndarray) -> np.ndarray:
        observed_block_sizes.append(len(block))
        return block

    ranked, _ = runner._stream_top_k(
        base,
        queries,
        k=2,
        chunk_size=2,
        device="cpu",
        transform=identity_transform,
    )

    assert observed_block_sizes == [2, 2, 1]
    assert ranked.shape == (3, 2)


def test_minimal_manifest_from_rows_is_valid_schema_v2(tmp_path: Path) -> None:
    runner = _runner()
    config = _config()
    identity = runner.codec_identity(CONFIG_PATH)
    spec = runner.build_row_specs(config)[0]
    row = {
        "row_id": spec["row_id"],
        "status": "passed",
        "reason": None,
        "identity": identity,
        "metrics": {"mse": 0.1},
        "sidecars": [],
        "extensions": {},
    }
    manifest = runner.build_manifest(
        config=config,
        config_path=CONFIG_PATH,
        rows=[row],
        expected_row_ids=[spec["row_id"]],
        source={"git_commit": "a" * 40, "dirty": False},
        execution_class={
            "name": "local_rtx5070ti_12gb",
            "gpu_name": "NVIDIA GeForce RTX 5070 Ti",
            "total_vram_bytes": 12_884_901_888,
            "driver_version": "576.88",
            "cuda_runtime": "12.8",
        },
        paper={
            "paper_id": "arxiv:2504.19874v1",
            "source_url": "https://arxiv.org/pdf/2504.19874v1",
            "retrieved_at": "2026-07-31T06:23:40Z",
            "sha256": "4" * 64,
        },
    )

    result = validate_v2_artifact(manifest, repo_root=tmp_path)

    assert result.valid, result.errors
    assert manifest["pass_state_counts"] == dict(
        Counter(
            {
                "passed": 1,
                "failed": 0,
                "invalid": 0,
                "oom": 0,
                "unsupported": 0,
                "unavailable": 0,
            }
        )
    )
    assert manifest["claims"] == []


def test_paired_seed_block_bootstrap_is_deterministic_and_uses_seed_means() -> None:
    runner = _runner()
    mse_seed_means = np.linspace(0.1, 1.6, 16, dtype=np.float64)
    product_seed_means = mse_seed_means + 0.25

    first = runner.paired_seed_block_bootstrap(
        product_seed_means,
        mse_seed_means,
        seed=1729,
        iterations=10_000,
    )
    second = runner.paired_seed_block_bootstrap(
        product_seed_means,
        mse_seed_means,
        seed=1729,
        iterations=10_000,
    )

    assert first == second
    assert first["seed_block_count"] == 16
    assert first["product_minus_mse_mean"] == pytest.approx(0.25)
    assert first["bootstrap_95_ci"] == pytest.approx([0.25, 0.25])

    rows = []
    identity = runner.codec_identity(CONFIG_PATH)
    for pair_id, (mse_value, product_value) in enumerate(
        zip(mse_seed_means, product_seed_means, strict=True)
    ):
        common = {
            "status": "passed",
            "reason": None,
            "identity": identity,
            "sidecars": [],
        }
        rows.append(
            {
                **common,
                "row_id": f"syn-sphere-d64-mse-b2-s{pair_id:02d}",
                "metrics": {
                    "mean_squared_l2_error": float(mse_value),
                    "queries": {
                        "independent": {"rmse": float(mse_value + 1.0)},
                        "high_inner_product": {"rmse": float(mse_value + 2.0)},
                    },
                },
                "extensions": {
                    "row_spec": {
                        "dataset": "synthetic",
                        "distribution": "sphere",
                        "dimension": 64,
                        "codec": "mse",
                        "nominal_total_bits": 2,
                        "pair_id": pair_id,
                    }
                },
            }
        )
        rows.append(
            {
                **common,
                "row_id": f"syn-sphere-d64-product-b2-s{pair_id:02d}",
                "metrics": {
                    "mean_squared_l2_error": float(product_value),
                    "queries": {
                        "independent": {"rmse": float(product_value + 1.0)},
                        "high_inner_product": {
                            "rmse": float(product_value + 2.0)
                        },
                    },
                },
                "extensions": {
                    "row_spec": {
                        "dataset": "synthetic",
                        "distribution": "sphere",
                        "dimension": 64,
                        "codec": "product",
                        "nominal_total_bits": 2,
                        "pair_id": pair_id,
                    }
                },
            }
        )

    aggregates = runner.aggregate_product_comparisons(rows, seed=1729, iterations=10_000)

    assert {aggregate["metric"] for aggregate in aggregates} == {
        "mean_squared_l2_error",
        "queries.independent.rmse",
        "queries.high_inner_product.rmse",
    }
    assert all(aggregate["seed_block_count"] == 16 for aggregate in aggregates)
    assert all(
        aggregate["product_minus_mse_mean"] == pytest.approx(0.25)
        for aggregate in aggregates
    )


def test_glove_subset_is_normalized_and_recomputes_exact_top_one(
    tmp_path: Path,
) -> None:
    h5py = pytest.importorskip("h5py")
    runner = _runner()
    path = tmp_path / "tiny-glove.hdf5"
    train = np.array(
        [[4.0, 0.0], [0.0, 7.0], [3.0, 3.0], [-5.0, 0.0]],
        dtype=np.float32,
    )
    queries = np.array([[2.0, 0.0], [0.0, 9.0]], dtype=np.float32)
    with h5py.File(path, "w") as handle:
        handle.create_dataset("train", data=train)
        handle.create_dataset("test", data=queries)
        handle.create_dataset(
            "neighbors",
            data=np.array([[1, 1], [0, 0]], dtype=np.int32),
        )
    base_ids = np.array([0, 1, 2], dtype="<u4")
    query_ids = np.array([0, 1], dtype="<u4")

    base, query = runner.load_normalized_glove_arrays(
        path, base_ids, query_ids, read_chunk_size=2
    )
    exact = runner.exact_top_one(base, query, chunk_size=2, device="cpu")
    scored = runner.score_glove_codec(
        base,
        query,
        exact,
        codec_name="mse",
        nominal_total_bits=2,
        rotation_seed=10_000,
        projection_seed=1_000_000,
        chunk_size=2,
        top_k=(1, 2),
        device="cpu",
    )

    np.testing.assert_allclose(np.linalg.norm(base, axis=1), 1.0, atol=1e-7)
    np.testing.assert_allclose(np.linalg.norm(query, axis=1), 1.0, atol=1e-7)
    assert exact.tolist() == [0, 1]
    assert exact.tolist() != [1, 0]
    assert scored["ranked_ids"].shape == (2, 2)
    assert scored["metrics"]["retrieval"] == runner.one_at_k(
        exact, scored["ranked_ids"], (1, 2)
    )
    assert scored["metrics"]["retrieval_algorithm"] == (
        "exact_chunked_brute_force_dot_product"
    )
    assert scored["metrics"]["scoring_device"] == "cpu"
    assert scored["metrics"]["full_score_matrix_materialized"] is False
    assert scored["metrics"]["maximum_score_block"] == {
        "rows": 2,
        "columns": 2,
    }
    assert "queries" not in scored["metrics"]
    assert "finite_float32_residual_expectation" not in scored["metrics"]


def _tiny_config(tmp_path: Path) -> Path:
    config = _config()
    config["artifact_id"] = "codec-local-test"
    config["seed_pairs"] = [config["seed_pairs"][0]]
    config["synthetic"]["dimensions"] = [64]
    config["synthetic"]["vector_count"] = 5
    config["synthetic"]["chunk_size"] = 2
    config["synthetic"]["codecs"] = {
        "mse": {"nominal_total_bits": [1]},
        "product": {"nominal_total_bits": [2]},
    }
    config["synthetic"]["distributions"] = [
        config["synthetic"]["distributions"][0]
    ]
    config["synthetic"]["fixtures"] = [config["synthetic"]["fixtures"][0]]
    config["glove200"]["url"] = "file:///definitely-not-present.hdf5"
    config["glove200"]["filename"] = "missing.hdf5"
    config["glove200"]["codecs"] = ["mse"]
    config["glove200"]["nominal_total_bits"] = [2]
    config["glove200"]["datasets"] = {
        "train": {"shape": [4, 64]},
        "test": {"shape": [2, 64]},
        "neighbors": {"shape": [2, 2]},
    }
    config["glove200"]["base_selection"] = {
        "algorithm": "literal_sorted_ids",
        "ids": [0, 1, 2],
        "count": 3,
        "dtype": "little-endian uint32",
        "sha256": hashlib.sha256(
            np.array([0, 1, 2], dtype="<u4").tobytes()
        ).hexdigest(),
    }
    config["glove200"]["query_selection"] = {
        "start": 0,
        "stop": 2,
        "dtype": "little-endian uint32",
        "sha256": hashlib.sha256(
            np.array([0, 1], dtype="<u4").tobytes()
        ).hexdigest(),
    }
    config["glove200"]["top_k"] = [1, 2]
    path = tmp_path / "tiny-codec-config.json"
    path.write_text(json.dumps(config), encoding="utf-8")
    return path


def test_tiny_end_to_end_run_writes_valid_outputs_and_resumes_passed_row(
    tmp_path: Path,
) -> None:
    config_path = _tiny_config(tmp_path)
    output_dir = tmp_path / "artifact"
    cache_dir = tmp_path / "cache"

    command = [
        sys.executable,
        "scripts/run_codec_reproduction.py",
        "--config",
        str(config_path),
        "--output-dir",
        str(output_dir),
        "--dataset-cache",
        str(cache_dir),
        "--no-download",
    ]
    first_completed = subprocess.run(
        command,
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert first_completed.returncode == 0, first_completed.stderr
    first_manifest = json.loads(
        (output_dir / "manifest.json").read_text(encoding="utf-8")
    )
    synthetic_id = "syn-sphere-d64-mse-b1-s00"
    product_id = "syn-sphere-d64-product-b2-s00"
    glove_id = "glove200-mse-b2-s00"
    synthetic_path = output_dir / "rows" / f"{synthetic_id}.json"
    glove_path = output_dir / "rows" / f"{glove_id}.json"
    first_mtime = synthetic_path.stat().st_mtime_ns
    first_glove_mtime = glove_path.stat().st_mtime_ns
    first_bytes = synthetic_path.read_bytes()

    second_completed = subprocess.run(
        command,
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert second_completed.returncode == 0, second_completed.stderr
    second_manifest = json.loads(
        (output_dir / "manifest.json").read_text(encoding="utf-8")
    )

    assert synthetic_path.stat().st_mtime_ns == first_mtime
    assert glove_path.stat().st_mtime_ns != first_glove_mtime
    assert synthetic_path.read_bytes() == first_bytes
    assert first_manifest["expected_row_ids"] == [
        glove_id,
        synthetic_id,
        product_id,
    ]
    assert [row["row_id"] for row in first_manifest["rows"]] == [
        glove_id,
        synthetic_id,
        product_id,
    ]
    assert first_manifest["pass_state_counts"]["passed"] == 2
    assert first_manifest["pass_state_counts"]["unavailable"] == 1
    assert first_manifest["status"] == "partial"
    assert second_manifest["status"] == "partial"
    assert (output_dir / "rows.jsonl").is_file()
    assert (output_dir / "manifest.json").is_file()
    assert not list(output_dir.rglob("*.tmp"))

    synthetic = next(
        row for row in first_manifest["rows"] if row["row_id"] == synthetic_id
    )
    assert set(synthetic["metrics"]) >= {
        "mean_squared_l2_error",
        "relative_squared_error",
        "encode_latency_ms",
        "decode_latency_ms",
        "actual_physical_bytes",
        "formula_physical_bytes",
        "effective_physical_bits_per_coordinate",
        "nominal_total_bits_per_coordinate",
        "queries",
    }
    assert synthetic["extensions"]["timing_class"] == (
        "local_reference_cpu_timing"
    )
    product = next(
        row for row in first_manifest["rows"] if row["row_id"] == product_id
    )
    assert product["metrics"]["sample_count"] == 5
    assert product["metrics"]["actual_physical_bytes"] == (
        product["metrics"]["formula_physical_bytes"]
    )
    assert product["metrics"]["effective_physical_bits_per_coordinate"] > (
        product["metrics"]["nominal_total_bits_per_coordinate"]
    )
    assert set(product["metrics"]["finite_float32_residual_expectation"]) == {
        "independent",
        "high_inner_product",
    }
    assert first_manifest["extensions"]["claim_labels"] == _config()[
        "claim_labels"
    ]
    assert first_manifest["extensions"]["retrieval_policy"] == {
        "device": "cpu",
        "base_chunk_size": 2_048,
        "codec_encode_decode_device": "cpu",
        "gpu_indexing_performed": False,
        "retrieval_algorithm": "exact_chunked_brute_force_dot_product",
        "scoring_device": "cpu",
        "full_score_matrix_materialized": False,
        "maximum_score_block": {"rows": 2, "columns": 3},
    }
    result = validate_v2_artifact(first_manifest, repo_root=tmp_path)
    assert result.valid, result.errors


def test_existing_corrupt_glove_file_marks_rows_invalid_not_unavailable(
    tmp_path: Path,
) -> None:
    runner = _runner()
    config_path = _tiny_config(tmp_path)
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    (cache_dir / "missing.hdf5").write_bytes(b"corrupt")

    manifest = runner.execute_reproduction(
        config_path=config_path,
        output_dir=tmp_path / "artifact-corrupt",
        dataset_cache=cache_dir,
        allow_download=False,
    )
    glove = next(
        row for row in manifest["rows"] if row["row_id"].startswith("glove200-")
    )

    assert glove["status"] == "invalid"
    assert "size mismatch" in glove["reason"]
    assert manifest["status"] == "invalid"


def test_resume_fingerprint_binds_protocol_identity_config_spec_input_and_policy() -> None:
    runner = _runner()
    spec = runner.build_row_specs(_config())[0]
    inputs = {"synthetic_fixture_sha256": "1" * 64}

    fingerprint = runner._fingerprint(
        spec,
        CONFIG_PATH,
        inputs,
        retrieval_device="cpu",
        retrieval_chunk_size=2_048,
    )

    assert fingerprint["protocol"] == "codec_reproduction_row_v1"
    assert fingerprint["identity"] == runner.codec_identity(CONFIG_PATH)
    assert fingerprint["config_sha256"] == hashlib.sha256(
        CONFIG_PATH.read_bytes()
    ).hexdigest()
    assert len(fingerprint["row_spec_sha256"]) == 64
    assert fingerprint["input_sha256"] == inputs
    assert len(fingerprint["code_sha256"]) == 64
    assert fingerprint["retrieval_policy"] == {
        "device": "cpu",
        "base_chunk_size": 2_048,
    }
