# TurboQuant Local Foundation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Establish a clean, artifact-v2 local baseline and independently implement the paper-faithful TurboQuant MSE/product reference codecs before the next Triton architecture change.

**Architecture:** Keep the paper oracle as deterministic CPU-first modules that do not depend on the Qwen adapter or packed runtime. Extend the existing validator-first evidence layer to identify paper revisions, execution classes, exact matrices, dirty state, and row-level outcomes. Preserve the optimized one-page Qwen direct path as the systems baseline.

**Tech Stack:** Python 3.12, PyTorch 2.7, Transformers 4.57.3, Triton 3.3, pytest, SciPy/NumPy already locked by the repository.

## Global Constraints

- Public MIT clean-room implementation; do not copy code, constants, kernel structure, comments, or fixtures from `0xsero/turboquant`.
- Canonical first paper target is `https://arxiv.org/abs/2504.19874v1`; every reproduction artifact records its retrieval date and SHA-256.
- Qwen local systems baseline uses pinned Qwen2.5-7B-Instruct NF4 weights and uncompressed FP16 KV; Llama work is a later plan.
- Preserve failed, invalid, OOM, and unsupported rows; no missing row may qualify.
- Keep paper-faithful profiles and Forge systems profiles separately named.
- Never reconstruct or concatenate full historical K/V inside the qualified direct path.
- Use test-first red/green cycles for behavior changes, keep output pristine, and commit each task after focused plus required regression tests pass.
- Do not modify or delete the previous flat `.superpowers/sdd/progress.md`; this plan owns a separate SDD workspace.

---

### Task 1: Restore and certify the local baseline

**Files:**
- Modify: `kernels/packed_attention.py`
- Delete generated: `codebooks/codebook_d256_b8.json`
- Modify: `.github/workflows/ci.yml`
- Modify: `scripts/README.md`
- Modify: `UPDATED_PLAN.md`
- Modify: `tests/test_documentation_claims.py`

**Interfaces:**
- Consumes: current branch `codex/turboquant-forge` at `2b5890b` plus the interrupted two-line launch-tuning diff.
- Produces: a clean, documented implementation baseline matching the best measured one-page fused kernel (`BLOCK_T=16`, `num_warps=4`) and CI that installs the locked runtime before tests.

- [ ] **Step 1: Capture the starting state and restore the measured kernel configuration**

Record `git status --short`, `git diff`, and `git rev-parse HEAD` in the task report. Restore the canonical one-page launch to:

```python
BLOCK_T=16,
num_warps=4,
```

Remove only the unreferenced generated `codebooks/codebook_d256_b8.json`.
Leave `.superpowers/sdd/progress.md` untouched.

- [ ] **Step 2: Write failing documentation/CI behavior tests**

Extend `tests/test_documentation_claims.py` so the documentation corpus includes
`UPDATED_PLAN.md` and `scripts/README.md`. Add a CI contract test that loads
`.github/workflows/ci.yml` and asserts the CPU job installs
`requirements/runtime.lock` before invoking pytest.

```python
def test_all_public_plans_and_script_docs_use_current_claims():
    corpus = public_claim_corpus()
    assert "5.2x" not in corpus
    assert "full through 16K" not in corpus
    assert ">=80% of baseline" not in corpus

def test_ci_installs_locked_runtime_before_pytest():
    steps = load_ci_steps()
    install_index = next(i for i, step in enumerate(steps) if "requirements/runtime.lock" in step.run)
    test_index = next(i for i, step in enumerate(steps) if "pytest" in step.run)
    assert install_index < test_index
```

Run:

```bash
python -m pytest tests/test_documentation_claims.py tests/test_environment.py -q
```

Expected: FAIL because stale documents are newly included and CI omits the
runtime lock installation.

- [ ] **Step 3: Reconcile stale documents and CI minimally**

Update `UPDATED_PLAN.md` into a historical-plan notice that points to the
2026-07-31 design and current `docs/RESULTS.md`. Rewrite `scripts/README.md`
around `run_direct_benchmark.py`, `reproduce_release.py`, and
`validate_artifact.py`, with no unsupported 16K/32K claims. Update CI to install
the runtime lock and dev test tools before running ruff/pytest.

- [ ] **Step 4: Verify the complete baseline**

Run in WSL with the existing pinned environment:

```bash
python -m pytest tests -m "not gpu" -q
python -m pytest tests/test_packed_attention_kernels.py \
  tests/test_packed_attention_hot_path.py \
  tests/test_qwen_fused_page_kernel.py \
  tests/test_qwen_tail_kernel.py -m gpu -q
TURBOQUANT_RUN_MODEL_TESTS=1 python -m pytest tests/test_qwen_adapter_gpu.py -m gpu -q
python -m ruff check .
git diff --check
```

Expected: all commands exit `0`; the model smoke reports direct/reference
greedy-token agreement.

- [ ] **Step 5: Commit the baseline**

```bash
git add .github/workflows/ci.yml kernels/packed_attention.py scripts/README.md \
  UPDATED_PLAN.md tests/test_documentation_claims.py
git commit -m "chore: certify the optimized forge baseline"
```

### Task 2: Introduce artifact schema v2 and a claim ledger

**Files:**
- Modify: `scripts/benchmark_schema.py`
- Modify: `scripts/repro_utils.py`
- Modify: `scripts/validate_artifact.py`
- Add: `scripts/claim_ledger.py`
- Add: `tests/test_artifact_schema_v2.py`
- Add: `tests/test_claim_ledger.py`

**Interfaces:**
- Produces:

```python
class ExecutionClass(TypedDict):
    name: Literal[
        "local_rtx5070ti_12gb",
        "gilbreth_a100_80gb",
        "gilbreth_a100_40gb",
        "gilbreth_a30",
        "gilbreth_a10",
    ]
    gpu_name: str
    total_vram_bytes: int
    driver_version: str
    cuda_runtime: str

def validate_v2_artifact(payload: dict[str, object], *, repo_root: Path) -> ValidationResult: ...
def build_claim_ledger(manifests: Sequence[Path]) -> dict[str, object]: ...
```

- [ ] **Step 1: Write failing schema-v2 tests**

Use hand-written fixtures. Tests must reject unknown execution classes,
dirty-source qualification, missing expected row IDs, duplicated row IDs,
unavailable sidecars, mixed model/profile hashes, and a claimed metric without
an exact artifact/row/query reference.

```python
def test_v2_rejects_qualified_dirty_source(valid_v2_artifact):
    valid_v2_artifact["source"]["dirty"] = True
    result = validate_v2_artifact(valid_v2_artifact, repo_root=ROOT)
    assert not result.valid
    assert "dirty source cannot qualify" in result.errors

def test_claim_ledger_requires_exact_metric_query(valid_manifest):
    valid_manifest["claims"][0].pop("query")
    with pytest.raises(ValueError, match="claim query"):
        build_claim_ledger_from_payloads([valid_manifest])
```

Run:

```bash
python -m pytest tests/test_artifact_schema_v2.py tests/test_claim_ledger.py -q
```

Expected: FAIL because schema v2 and the claim ledger do not exist.

- [ ] **Step 2: Implement schema v2 as an additive version**

Keep schema v1 validation for historical artifacts. Add v2 validation for:
paper identity, source cleanliness, execution class, model/weight/KV dtype,
predeclared row IDs, statistics specification, row sidecars, pass-state counts,
and exact claim queries. Unknown fields remain allowed only inside
`extensions`.

- [ ] **Step 3: Implement deterministic claim-ledger generation**

`build_claim_ledger()` reads compact manifests, verifies referenced rows, and
emits stable JSON sorted by `(claim_id, artifact_id)`. It never opens large raw
outputs unless a manifest query explicitly points to them.

- [ ] **Step 4: Verify old and new artifacts**

Run:

```bash
python -m pytest tests/test_benchmark_schema.py tests/test_artifact_validation.py \
  tests/test_artifact_schema_v2.py tests/test_claim_ledger.py -q
python scripts/validate_artifact.py \
  artifacts/qualified-semantic-final/20260718T074435Z-5c9d4c34b0f0/repro_report.json \
  --no-binding-check
```

Expected: tests pass; the historical v1 artifact remains structurally valid
without being promoted to v2.

- [ ] **Step 5: Commit schema v2**

```bash
git add scripts/benchmark_schema.py scripts/repro_utils.py \
  scripts/validate_artifact.py scripts/claim_ledger.py \
  tests/test_artifact_schema_v2.py tests/test_claim_ledger.py
git commit -m "feat: add evidence schema v2 and claim ledger"
```

### Task 3: Implement deterministic Haar rotations and spherical codebooks

**Files:**
- Add: `core/paper_codebook.py`
- Add: `core/paper_rotation.py`
- Add: `tests/test_paper_codebook.py`
- Add: `tests/test_paper_rotation.py`
- Add: `docs/references/turboquant-2504.19874v1.json`

**Interfaces:**
- Produces:

```python
@dataclass(frozen=True)
class PaperReference:
    paper_id: str
    source_url: str
    retrieved_at: str
    sha256: str

def seeded_haar_rotation(dimension: int, seed: int, *, dtype: torch.dtype = torch.float64) -> torch.Tensor: ...
def sphere_coordinate_density(x: torch.Tensor, dimension: int) -> torch.Tensor: ...
def lloyd_max_sphere_codebook(dimension: int, bits: int, *, iterations: int = 256, tolerance: float = 1e-12) -> torch.Tensor: ...
```

- [ ] **Step 1: Record the paper reference identity**

Download `https://arxiv.org/pdf/2504.19874v1` outside the tracked tree, compute
SHA-256, and write only its identity JSON under `docs/references`. Include
`paper_id="arxiv:2504.19874v1"`, the exact URL, UTC retrieval timestamp, hash,
title, authors, and `CC BY 4.0`.

- [ ] **Step 2: Write failing rotation tests**

Use literal dimensions/seeds. Assert positive QR diagonal convention,
orthogonality to `1e-10`, determinant magnitude one, deterministic equality for
the same seed, and unequal rotations for different seeds.

```python
def test_seeded_haar_rotation_is_deterministic_and_orthogonal():
    rotation = seeded_haar_rotation(16, 1729)
    torch.testing.assert_close(rotation.T @ rotation, torch.eye(16, dtype=torch.float64), atol=1e-10, rtol=0)
    torch.testing.assert_close(rotation, seeded_haar_rotation(16, 1729), atol=0, rtol=0)
```

Run:

```bash
python -m pytest tests/test_paper_rotation.py -q
```

Expected: FAIL because the module does not exist.

- [ ] **Step 3: Implement the seeded Haar rotation**

Generate a CPU `torch.Generator`, sample a float64 Gaussian square matrix, call
`torch.linalg.qr`, and multiply each Q column by the sign of the corresponding
R diagonal so every adjusted diagonal is nonnegative. Reject dimension `< 2`,
non-integer seeds, and non-floating dtypes.

- [ ] **Step 4: Write failing density/codebook tests**

Assert density symmetry, numerical integral one, second moment `1/d`, sorted
symmetric centroids, deterministic output, and monotone quantization MSE for
bits `1..4` at dimensions `64/128/256`.

- [ ] **Step 5: Implement deterministic Lloyd-Max integration**

Use SciPy quadrature for interval probability and first moment. Initialize
thresholds by equal-probability quantiles, preserve exact symmetry, stop only
when max centroid change is at most `tolerance`, and raise if convergence is not
reached in `iterations`.

- [ ] **Step 6: Verify and commit**

```bash
python -m pytest tests/test_paper_rotation.py tests/test_paper_codebook.py -q
python -m ruff check core/paper_rotation.py core/paper_codebook.py \
  tests/test_paper_rotation.py tests/test_paper_codebook.py
git add core/paper_rotation.py core/paper_codebook.py \
  tests/test_paper_rotation.py tests/test_paper_codebook.py \
  docs/references/turboquant-2504.19874v1.json
git commit -m "feat: add paper-faithful rotation and codebooks"
```

### Task 4: Implement the TurboQuant MSE reference codec

**Files:**
- Add: `core/paper_mse.py`
- Modify: `core/packing.py`
- Add: `tests/test_paper_mse.py`
- Modify: `tests/test_packing.py`

**Interfaces:**
- Consumes: `seeded_haar_rotation()` and `lloyd_max_sphere_codebook()`.
- Produces:

```python
@dataclass(frozen=True)
class TurboQuantMSEConfig:
    dimension: int
    bits: int
    rotation_seed: int

@dataclass(frozen=True)
class TurboQuantMSEEncoded:
    indices: torch.Tensor
    norm: torch.Tensor
    config: TurboQuantMSEConfig

class TurboQuantMSEReference:
    def __init__(self, config: TurboQuantMSEConfig) -> None: ...
    def encode(self, vectors: torch.Tensor) -> TurboQuantMSEEncoded: ...
    def decode(self, encoded: TurboQuantMSEEncoded) -> torch.Tensor: ...
    def physical_nbytes(self, encoded: TurboQuantMSEEncoded) -> int: ...
```

- [ ] **Step 1: Write failing behavioral tests**

Tests cover zero vectors, one/batched vectors, dimensions `64/128/256`, bits
`1/2/3/4`, float32/float64 inputs, deterministic indices, norm preservation,
hand-packed physical bytes, and lower empirical sphere MSE as bits increase.
The expected byte counts are literal and include packed-index padding plus
stored float32 norms.

- [ ] **Step 2: Run RED**

```bash
python -m pytest tests/test_paper_mse.py -q
```

Expected: FAIL because `core.paper_mse` does not exist.

- [ ] **Step 3: Extend the shared index packer to one-bit indices**

Add `1` to the generic index-width contract rather than duplicating a private
paper-only bit packer or misusing the signed QJL packer. Extend
`tests/test_packing.py` so one-bit indices round-trip, reject values outside
`[0, 2)`, and zero their final-byte padding for non-byte-aligned widths.

- [ ] **Step 4: Implement minimal encode/decode**

Normalize nonzero rows, rotate unit vectors, choose nearest codebook centroid
per coordinate, and bit-pack indices with the existing generic packer. Decode
centroids and apply the inverse consistently with the row-vector convention
(`unit @ rotation.T` to encode, `centroids @ rotation` to decode), multiply by
stored norms, and preserve zero rows exactly. Single-vector inputs round-trip
as one-dimensional vectors; batched inputs round-trip as two-dimensional
tensors. Store scalar/vector norms as float32 and return decoded reference
values as CPU float64. Keep reference calculations on CPU; reject CUDA inputs
and tensor ranks outside one or two.

- [ ] **Step 5: Verify cross-module properties**

```bash
python -m pytest tests/test_paper_rotation.py tests/test_paper_codebook.py \
  tests/test_paper_mse.py tests/test_packing.py -q
python -m ruff check core/paper_mse.py tests/test_paper_mse.py
```

Expected: all tests pass with no warnings.

- [ ] **Step 6: Commit**

```bash
git add core/paper_mse.py core/packing.py \
  tests/test_paper_mse.py tests/test_packing.py
git commit -m "feat: add TurboQuant MSE reference codec"
```

### Task 5: Implement the TurboQuant product/QJL reference codec

**Files:**
- Add: `core/paper_product.py`
- Add: `tests/test_paper_product.py`

**Interfaces:**
- Consumes: `TurboQuantMSEReference` with `total_bits - 1`.
- Produces:

```python
@dataclass(frozen=True)
class TurboQuantProductConfig:
    dimension: int
    total_bits: int
    rotation_seed: int
    projection_seed: int

@dataclass(frozen=True)
class TurboQuantProductEncoded:
    mse: TurboQuantMSEEncoded
    residual_signs: torch.Tensor
    residual_norm: torch.Tensor
    config: TurboQuantProductConfig

class TurboQuantProductReference:
    def encode(self, vectors: torch.Tensor) -> TurboQuantProductEncoded: ...
    def decode(self, encoded: TurboQuantProductEncoded) -> torch.Tensor: ...
    def estimate_inner_product(self, queries: torch.Tensor, encoded: TurboQuantProductEncoded) -> torch.Tensor: ...
    def physical_nbytes(self, encoded: TurboQuantProductEncoded) -> int: ...
```

`dimension` is an exact built-in integer at least `2`; `total_bits` is an exact
built-in integer in `2..9`, because the composed MSE codec consumes
`total_bits - 1` and supports widths `1..8`. Both seeds use PyTorch's inclusive
generator range `[-2**63, 2**64 - 1]`. Reject booleans, integer subclasses, and
objects that spoof `__class__`.

Encoding accepts finite CPU float32/float64 `[d]` or non-empty `[n, d]`
vectors. A single payload stores packed signs as `[ceil(d/8)]` and a scalar
float32 residual norm; a batch stores `[n, ceil(d/8)]` signs and `[n]` norms.
Every consumer validates the concrete encoded/config types, derived nested MSE
config, CPU dtype/device, rank, byte width, zero padding, finite nonnegative
norms, and single/batch consistency. The physical payload per vector is:

```text
ceil(d * (total_bits - 1) / 8) + ceil(d / 8) + 8 bytes
```

The two four-byte terms are the MSE and residual float32 norms. Shared
codebooks, rotation, projection matrix, and config objects are excluded.

`estimate_inner_product()` accepts finite CPU float32/float64 query shapes
`[d]` or non-empty `[q, d]`. Its retrieval-oriented outputs are scalar for one
query/one encoded vector, `[n]` for one query/encoded batch, `[q]` for a query
batch/one encoded vector, and `[q, n]` for two batches. Results and decoded
vectors are CPU float64.

- [ ] **Step 1: Write failing QJL/product tests**

Use literal small fixtures for sign packing and byte counts, including a
non-byte-aligned dimension. Test that:

- zero residuals encode deterministically with zero residual norm;
- residual signs are physically one bit per dimension;
- direct inner-product estimation equals query dot decoded product vectors;
- an independent Algorithm 2 oracle matches the row-vector implementation;
- exact config, seed, payload, and four query/output shape contracts fail
  closed; and
- across dimensions `64/128` and projection seeds `0..127`, the per-seed
  signed-error mean satisfies the predeclared deterministic smoke gate
  `abs(mean) <= 4 * SEM + 1e-12`, centered on the finite-float32-residual-norm
  expectation.

Compute and expose the ordinary 95% confidence interval for reporting, but do
not require that interval to contain zero: even an unbiased estimator fails
that gate about five percent of the time. Treat “product beats the same-total-
bit MSE codec” as a Task 6 empirical low-bit comparison, not a unit-test
invariant.

- [ ] **Step 2: Run RED**

```bash
python -m pytest tests/test_paper_product.py -q
```

Expected: FAIL because the product codec does not exist.

- [ ] **Step 3: Implement Algorithm 2 exactly**

Generate a square CPU float64 Gaussian projection `S[d, d]` from a fresh
generator seeded by `projection_seed`, without changing global RNG state.
Compute the original-space residual against the physical MSE decode. With row
vectors, encode signs as `sign(residual @ S.T)`, defining zero projections as
`+1`; do not normalize the residual before projection because Algorithm 2 uses
the residual directly. Pack one sign bit per dimension and store the residual
norm as float32 under the same finite-overflow and ordinary-underflow policy as
the MSE norm. Decode with
`sqrt(pi / 2) / dimension * residual_norm * signs @ S`, then add the MSE
decode. Use the same reconstruction in `estimate_inner_product()` so the test
oracle has one mathematical definition.

- [ ] **Step 4: Verify and commit**

```bash
python -m pytest tests/test_paper_mse.py tests/test_paper_product.py -q
python -m ruff check core/paper_product.py tests/test_paper_product.py
git add core/paper_product.py tests/test_paper_product.py
git commit -m "feat: add TurboQuant product reference codec"
```

### Task 6: Build the local codec/retrieval reproduction artifact

**Files:**
- Add: `scripts/run_codec_reproduction.py`
- Add: `tests/test_codec_reproduction.py`
- Add: `configs/benchmarks/codec-local-v1.json`

**Interfaces:**
- Consumes: paper MSE/product codecs and artifact schema v2.
- Produces: a v2 `codec_reproduction` manifest plus row JSONL for synthetic
  distributions and GloVe.

- [ ] **Step 1: Write failing matrix/completeness tests**

The fixed config contains dimensions `64/128/256`; MSE widths `1/2/3/4`;
product total widths `2/3/4`; distributions
`sphere/gaussian/heavy_tail/outlier`; transform/projection seed pairs `0..15`;
and 10,000 vectors per synthetic cell. Product and MSE results at equal total
width are reported side by side, including whether the product estimator
improves the declared low-bit high-inner-product fixture, but the comparison
is never promoted as a theorem or universal pass gate. GloVe uses a pinned
dataset revision, sample IDs, and `1@k` for `k=1/10/100`.

Tests assert exact expected row IDs/counts, deterministic resumption, invalid
status on any missing row, and actual byte accounting.

- [ ] **Step 2: Run RED**

```bash
python -m pytest tests/test_codec_reproduction.py -q
```

Expected: FAIL because the runner/config do not exist.

- [ ] **Step 3: Implement the resumable runner**

Write one JSONL row atomically per cell, skip only rows whose full identity and
hashes match, and rebuild the manifest/summary from rows. Metrics are MSE,
relative MSE, inner-product RMSE, signed bias with 95% CI, encode/decode
latency, physical bytes/effective bits, and GloVe recall.

- [ ] **Step 4: Execute the local synthetic matrix**

```bash
python scripts/run_codec_reproduction.py \
  --config configs/benchmarks/codec-local-v1.json \
  --output-dir artifacts/codec-local-v1
python scripts/validate_artifact.py artifacts/codec-local-v1/manifest.json
```

Expected: complete synthetic rows validate. If GloVe is unavailable, its rows
must be `unavailable` with a persisted reason and the overall artifact remains
`partial`, never `qualified`.

- [ ] **Step 5: Commit runner/config/tests**

```bash
git add scripts/run_codec_reproduction.py tests/test_codec_reproduction.py \
  configs/benchmarks/codec-local-v1.json
git commit -m "feat: add local TurboQuant codec reproduction"
```

### Task 7: Publish the first transparent reproduction report

**Files:**
- Add: `docs/REPRODUCTION.md`
- Add: `docs/CLAIMS.md`
- Modify: `README.md`
- Modify: `docs/PAPER_CLAIMS_STATUS.md`
- Modify: `tests/test_documentation_claims.py`

**Interfaces:**
- Consumes: validated codec artifact and claim ledger.
- Produces: public tables with exact pass/fail/invalid/unavailable counts and
  artifact-backed metric queries.

- [ ] **Step 1: Write failing report-consistency tests**

Tests load the claim ledger and ensure every numeric claim in the generated
result tables references an artifact/query. They reject “paper equivalent,”
“32K qualified,” or Pareto-win text unless the ledger contains a passing claim
with matching scope.

- [ ] **Step 2: Run RED**

```bash
python -m pytest tests/test_documentation_claims.py -q
```

Expected: FAIL until new reports and ledger bindings exist.

- [ ] **Step 3: Write candid reports**

`REPRODUCTION.md` separates matched mathematical results, partial results,
contradictions, unavailable paper details, and Forge-only extensions.
`CLAIMS.md` lists current safe statements and exact evidence queries. README
links both and retains the existing negative 8K systems result until a new
quality-bound optimized artifact supersedes it.

- [ ] **Step 4: Final local-foundation verification**

```bash
python -m pytest tests -m "not gpu" -q
python -m ruff check .
python scripts/validate_artifact.py artifacts/codec-local-v1/manifest.json
git diff --check
git status --short
```

Expected: all code/test/validation commands exit `0`; only ignored artifacts
and the prior plan’s untouched ledger may remain outside committed source.

- [ ] **Step 5: Commit reports**

```bash
git add README.md docs/REPRODUCTION.md docs/CLAIMS.md \
  docs/PAPER_CLAIMS_STATUS.md tests/test_documentation_claims.py
git commit -m "docs: publish TurboQuant reproduction evidence"
```

## Follow-on plans

After this plan passes final review, create and execute these plans in order:

1. `turboquant-parallel-page-attention`: storage-owned page arenas, parallel
   page-state kernel, deterministic state reduction, Qwen2.5 integration, and
   1K/2K/4K/8K/16K systems evidence.
2. `turboquant-local-quality-matrix`: fixed-token replay, qualified NIAH,
   constrained RULER, and constrained LongBench-E under schema v2.
3. `turboquant-llama31-adapter`: local adapter/parity smoke followed by
   Gilbreth A100-80GB paper-scale reproduction.
4. `turboquant-model-promotions`: Qwen3 dense, then Qwen3.5 hybrid, then Gemma,
   each with separate capability and evidence gates.
