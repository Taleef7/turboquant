# TurboQuant Parallel Page Attention Implementation Plan

> **Execution rule:** Use subagent-driven development, test-driven development,
> and independent review for every task. Do not promote a microbenchmark,
> capacity projection, or diagnostic run into a public systems claim.

**Goal:** Replace the Qwen2.5 decode path's launch-per-page and fragmented-cache
overheads with a fixed-capacity packed page arena and deterministic
page-parallel attention, then determine with fresh RTX 5070 Ti evidence whether
the direct path achieves an end-to-end speed and peak-VRAM win.

**Architecture:** Preserve the existing batch-one, greedy-decode,
Qwen2.5-specific Transformers boundary. Each layer owns fixed-capacity
contiguous packed K/V arenas and fixed-stride raw-tail buffers whose pointers
do not change after reservation. One Triton grid computes an independent stable
softmax state for each `(page, query_head)`, one atomic-free kernel merges page
states in chronological order, and a stride-aware tail kernel merges the raw
tail and writes the result. A single session workspace is reused across layers
and reported separately from logical packed payload bytes.

**Why this plan:** The later one-trial 8K diagnostic at commit `6cd3ea5`
improved direct decode p50 from the older claim-bound 727.09 ms result to about
55.36 ms, but still trailed its 47.11 ms baseline and used more peak VRAM. It
consumed exactly 31 packed pages per layer call. The current coordinator launches
one dependent page kernel per page, producing 868 page launches per token across
28 layers. A prior two-pages-per-program experiment regressed, so this plan adds
parallelism across pages instead of increasing per-program register pressure.
These diagnostics guide engineering only; neither is claim-grade evidence.

**YAGNI boundary:** Do not add generalized serving, paging across requests,
dynamic arena growth, batch/beam/speculative decode, vLLM/SGLang integration,
structured Hadamard rotation, adaptive bits, sink/outlier policies, a
multi-model abstraction, or a `(page, KV-head)` seven-query-head fusion unless
the first page-parallel measurements prove it necessary.

---

## Non-negotiable contracts

- The supported fast path remains the pinned Qwen2.5-7B text layout:
  28 query heads, 4 KV heads, head dimension 128, 256-token packed pages,
  8-bit key indices, 6-bit grouped values, batch one, and one decode query.
- The caller declares `maximum_sequence_length` before prefill. Capacity never
  grows, and overflow fails before cache mutation.
- Full-history FP16/BF16 K/V is never reconstructed or retained.
- Arena and tail pointers are stable after reservation.
- A one-token append performs no allocation, clone, concatenation, or
  contiguous copy.
- Static rotation/codebook/boundary tensors are transferred once per layer,
  not retained once per page through payload closures.
- Page-state reduction is chronological and atomic-free. Same-input repeated
  launches must be bit-identical on the pinned GPU/runtime.
- The runtime reports logical used payload, allocated arena capacity, shared
  constants, reusable workspace, raw-tail capacity/usage, padding, and unique
  resident bytes separately.
- Qualified timing uses one cold/JIT run, two warmups, ten interleaved measured
  trials, CUDA synchronization, and no outlier deletion.
- The public Pareto gate is unchanged: qualified quality, at least 1.8x
  physical-storage reduction, throughput-ratio lower 95% bound above 1.0,
  latency-ratio upper 95% bound below 1.0, and direct peak allocated VRAM no
  higher than baseline.

---

### Task 1: Freeze the current performance and allocation baseline

**Files:**
- Add: `scripts/benchmark_parallel_page_attention.py`
- Add: `tests/test_parallel_page_benchmark.py`
- Modify: `scripts/README.md`

**Interfaces:**
- Consumes: current serial-page prepared decode and fixed synthetic Qwen pages.
- Produces: schema-v2 diagnostic rows at page counts `3/7/15/31/63`.

- [ ] **Step 1: Write failing benchmark-contract tests**

Require exact row IDs for current serial and future parallel implementations,
page count, launch counters, warmup/trial counts, CUDA-event p50/p95/p99,
workspace bytes, logical/capacity/runtime bytes, parity outcome, source/GPU
identity, and an explicit `diagnostic_only` label. Reject missing rows, fallback,
unsynchronized timing, one-trial output, or a claimed win without an interval.

- [ ] **Step 2: Run RED**

```bash
python -m pytest tests/test_parallel_page_benchmark.py -q
```

Expected: FAIL because the dedicated benchmark does not exist.

- [ ] **Step 3: Implement only the serial baseline lane**

Generate deterministic canonical packed pages and queries. Benchmark the
existing `PreparedPackedAttentionDecode` with one cold/JIT launch, two warmups,
and ten synchronized trials. Count rotation, page, reduction, tail, fallback,
and allocation events. Do not implement the new kernel in this task.

- [ ] **Step 4: Execute and preserve the baseline diagnostic**

```bash
python scripts/benchmark_parallel_page_attention.py \
  --implementations serial \
  --page-counts 3 7 15 31 63 \
  --output artifacts/parallel-page-baseline/manifest.json
python scripts/validate_artifact.py \
  artifacts/parallel-page-baseline/manifest.json
```

Expected: valid diagnostic artifact; page launch count grows linearly with
page count. Record exact commit and environment.

- [ ] **Step 5: Commit**

```bash
git add scripts/benchmark_parallel_page_attention.py \
  tests/test_parallel_page_benchmark.py scripts/README.md
git commit -m "bench: freeze serial page attention baseline"
```

---

### Task 2: Add the fixed-capacity storage-owned page arena

**Files:**
- Add: `core/packed_page_arena.py`
- Add: `tests/test_packed_page_arena.py`
- Modify: `core/packed_store.py`
- Modify: `core/packed_cache.py`

**Interfaces:**

```python
@dataclass(frozen=True)
class PackedPageArenaLayout:
    page_capacity: int
    page_tokens: int
    num_kv_heads: int
    head_dim: int
    key_packed_width: int
    value_packed_width: int
    value_groups: int
    tail_capacity: int

@dataclass(frozen=True)
class PackedArenaByteBreakdown:
    logical_payload: int
    arena_capacity: int
    tail_logical: int
    tail_capacity: int
    shared_constants: int
    workspace: int
    padding: int
    unique_resident: int

class PackedPageArena:
    @classmethod
    def reserve(
        cls,
        layout: PackedPageArenaLayout,
        *,
        device: torch.device,
        kv_dtype: torch.dtype,
    ) -> "PackedPageArena": ...
```

- [ ] **Step 1: Write failing arena shape/capacity tests**

Test exact shapes/dtypes for every payload and metadata tensor. Test built-in
integer validation, zero/negative/overflow capacity, device/dtype rejection,
and exact byte formulas. Capacity bytes must include unused reserved slots;
logical bytes include only used pages/tail. Unique resident accounting must
deduplicate tensors by storage pointer.

- [ ] **Step 2: Write failing ordering and pointer-stability tests**

Exercise totals around every boundary:
`127/128/255/256/383/384/639/640`. Assert chronological reference output,
unchanged prior page bytes, and stable data pointers. Exact-capacity append
succeeds; one additional token fails before any counter or byte changes.

- [ ] **Step 3: Run RED**

```bash
python -m pytest tests/test_packed_page_arena.py tests/test_packed_store.py -q
```

Expected: FAIL because arena-backed storage does not exist.

- [ ] **Step 4: Implement reservation and mutation-safe state**

Preallocate plain payload tensors, metadata tensors, and K/V tail buffers.
Keep host integers for `used_pages`, `tail_length`, and logical sequence length.
Do not add geometric growth or retain `PackedKeyPayload`/`PackedValuePayload`
objects in history.

- [ ] **Step 5: Integrate an opt-in arena store**

Add an explicit arena-backed construction path that requires
`maximum_sequence_length`. Keep the current block-list store available as the
reference/serial baseline until the new systems artifact passes. Do not switch
the model adapter yet.

- [ ] **Step 6: Verify and commit**

```bash
python -m pytest tests/test_packed_page_arena.py tests/test_packed_store.py \
  tests/test_packed_cache.py -q
python -m ruff check core/packed_page_arena.py core/packed_store.py \
  core/packed_cache.py tests/test_packed_page_arena.py
git diff --check
git add core/packed_page_arena.py core/packed_store.py core/packed_cache.py \
  tests/test_packed_page_arena.py
git commit -m "feat: add fixed packed page arena"
```

---

### Task 3: Eliminate append copies and per-page device constants

**Files:**
- Modify: `core/codec.py`
- Modify: `core/packed_page_arena.py`
- Modify: `core/packed_store.py`
- Modify: `tests/test_codec.py`
- Modify: `tests/test_packed_page_arena.py`
- Add: `tests/test_packed_arena_allocation_contract.py`

**Interfaces:**

```python
@dataclass(frozen=True)
class PreparedKeyCodecDevice:
    rotation: torch.Tensor
    centroids: torch.Tensor
    boundaries: torch.Tensor
    qjl_projection: torch.Tensor | None

class KeyCodec:
    def prepare_device(self, device: torch.device) -> PreparedKeyCodecDevice: ...
    def encode_into(
        self,
        keys: torch.Tensor,
        destination: PackedPageArena,
        page_id: int,
        *,
        prepared: PreparedKeyCodecDevice,
    ) -> None: ...
```

Implement only the smallest `encode_into`/copy-into-final-slot API necessary to
avoid retaining payload closures. A custom fused page-encode kernel is not part
of this task.

- [ ] **Step 1: Write failing constant-residency tests**

Assert one rotation, centroid table, and boundary table per layer/store;
repeated page encodes reuse exact pointers. Stored history contains only arena
tensors and integer state, not closures capturing device tensors.

- [ ] **Step 2: Write failing no-allocation append tests**

After reservation and prefill, monkeypatch/trace `torch.empty`, `clone`,
`torch.cat`, and `contiguous`; a one-token decode append must call none of them.
Crossing 384 tail tokens may encode one page and move the remaining 128 tokens
once, but must not change arena/tail pointers or touch prior pages.

- [ ] **Step 3: Write failing prefill-copy-volume tests**

An 8K-shaped synthetic prefill must copy each input token into final packed or
tail storage once. It must not clone the full prompt or repeatedly copy
shrinking remainders. Instrument destination bytes rather than inferring from
wall time.

- [ ] **Step 4: Run RED**

```bash
python -m pytest tests/test_codec.py tests/test_packed_page_arena.py \
  tests/test_packed_arena_allocation_contract.py -q
```

- [ ] **Step 5: Implement shared constants and direct arena writes**

Cache device constants on the owning codec/store. Encode complete prefill pages
from input slices and copy their final payload once into arena slots. Copy only
the final raw tail. For decode, write the token into the existing tail slot;
flush only at the declared boundary.

- [ ] **Step 6: Verify allocation and byte accounting**

Check that shared constants are counted once, closure-retained duplicates are
zero, and `unique_resident` reconciles all storages without double counting
views.

- [ ] **Step 7: Commit**

```bash
python -m pytest tests/test_codec.py tests/test_packed_page_arena.py \
  tests/test_packed_arena_allocation_contract.py tests/test_packed_store.py -q
python -m ruff check core tests/test_packed_arena_allocation_contract.py
git diff --check
git add core/codec.py core/packed_page_arena.py core/packed_store.py \
  tests/test_codec.py tests/test_packed_page_arena.py \
  tests/test_packed_arena_allocation_contract.py
git commit -m "perf: remove packed cache append copies"
```

---

### Task 4: Implement the page-state Triton kernel behind a benchmark gate

**Files:**
- Add: `kernels/qwen25_parallel_page_attention.py`
- Add: `tests/test_qwen_parallel_page_kernel.py`
- Modify: `scripts/benchmark_parallel_page_attention.py`
- Modify: `tests/test_parallel_page_benchmark.py`

**Interfaces:**

```python
@dataclass(frozen=True)
class ParallelPageWorkspace:
    rotated_query: torch.Tensor
    page_max: torch.Tensor
    page_denominator: torch.Tensor
    page_numerator: torch.Tensor
    reduced_max: torch.Tensor
    reduced_denominator: torch.Tensor
    reduced_numerator: torch.Tensor

def launch_qwen25_page_states(
    query: torch.Tensor,
    arena: PackedPageArena,
    prepared_codec: PreparedKeyCodecDevice,
    workspace: ParallelPageWorkspace,
    *,
    used_pages: int,
    scale: float,
) -> None: ...
```

- [ ] **Step 1: Write failing kernel parity tests**

For `1/2/3/31/63` pages, compare every page's independent `(max, denominator,
numerator)` state with a float32 PyTorch oracle. Include deterministic random,
all-zero, extreme positive/negative score, and final-byte packing fixtures.
Test fp16 and either qualified BF16 parity or explicit BF16 rejection.

- [ ] **Step 2: Write failing launch/allocation tests**

The page-state launch grid must be exactly one program per
`(page, query_head)`, perform no atomics/fallbacks, allocate no hot-path tensor,
and write only preallocated workspace. Launch count stays one as pages grow.

- [ ] **Step 3: Run RED**

```bash
python -m pytest tests/test_qwen_parallel_page_kernel.py -m gpu -q
```

- [ ] **Step 4: Implement from the accepted one-page equations**

Adapt the existing canonical single-page fused math, remove incoming online
state, add page strides, and store independent stable page states. Do not fuse
multiple pages in one program.

- [ ] **Step 5: Run compute-sanitizer**

```bash
compute-sanitizer --tool memcheck --error-exitcode 1 \
  /home/taleef/.venvs/turboquant-forge/bin/python -m pytest \
  tests/test_qwen_parallel_page_kernel.py -m gpu -q
```

- [ ] **Step 6: Apply the microbenchmark stop-gate**

```bash
python scripts/benchmark_parallel_page_attention.py \
  --implementations serial page-state \
  --page-counts 3 7 15 31 63 \
  --output artifacts/parallel-page-state/manifest.json
```

The page-state stage must beat the serial page path clearly at 31 and 63 pages
after accounting for its later reduction budget. If it does not, stop before
adapter integration, retain the artifact, profile the kernel, and amend this
plan from evidence. Do not relax parity or timing protocol.

- [ ] **Step 7: Commit**

```bash
git add kernels/qwen25_parallel_page_attention.py \
  tests/test_qwen_parallel_page_kernel.py \
  scripts/benchmark_parallel_page_attention.py \
  tests/test_parallel_page_benchmark.py
git commit -m "feat: add parallel packed page states"
```

---

### Task 5: Add deterministic reduction, stride-aware tail merge, and workspace reuse

**Files:**
- Modify: `kernels/qwen25_parallel_page_attention.py`
- Add: `core/parallel_page_workspace.py`
- Modify: `tests/test_qwen_parallel_page_kernel.py`
- Add: `tests/test_parallel_page_workspace.py`

**Interfaces:**

```python
def reserve_parallel_page_workspace(
    *,
    page_capacity: int,
    device: torch.device,
) -> ParallelPageWorkspace: ...

def launch_qwen25_page_reduction(
    workspace: ParallelPageWorkspace,
    *,
    used_pages: int,
) -> None: ...

def launch_qwen25_tail_merge(
    query: torch.Tensor,
    tail_keys: torch.Tensor,
    tail_values: torch.Tensor,
    workspace: ParallelPageWorkspace,
    *,
    tail_length: int,
    tail_head_stride: int,
    tail_token_stride: int,
    scale: float,
    output: torch.Tensor,
) -> None: ...
```

- [ ] **Step 1: Write failing chronological-reduction tests**

Compare merged states with a chronological PyTorch oracle for
`0/1/2/3/31/63` pages and adversarial maxima. Assert no atomics and
bit-identical repeated launches on the pinned GPU. A deliberately reversed
page order must differ on a rounding-sensitive fixture so the test is not
vacuous.

- [ ] **Step 2: Write failing fixed-stride tail tests**

Test tail lengths `1/127/128/255/256/320/383` using a partially filled
fixed-capacity buffer. The kernel must honor supplied strides and must not call
`.contiguous()`. Compare final output against the packed reference.

- [ ] **Step 3: Write failing workspace ownership tests**

Reserve once for maximum page capacity. Assert all pointers remain stable,
workspace is reused across 28 sequential layers, hot-path calls allocate
nothing, and workspace bytes are reported separately from cache payload.

- [ ] **Step 4: Run RED**

```bash
python -m pytest tests/test_qwen_parallel_page_kernel.py \
  tests/test_parallel_page_workspace.py -m gpu -q
```

- [ ] **Step 5: Implement two deterministic kernels**

Use one reduction program per query head, looping page IDs in ascending order
with stable online-softmax merge equations. Use the raw tail as the final
chronological segment and normalize directly into the output. No tree
reduction, atomics, or scheduler-dependent order.

- [ ] **Step 6: Verify four-launch invariant**

For any positive page count, one layer decode uses exactly:

1. query rotation;
2. page-state grid;
3. page-state reduction;
4. tail merge/final output.

Handle zero used pages explicitly without inventing empty page work.

- [ ] **Step 7: Sanitizer, benchmark, and commit**

```bash
compute-sanitizer --tool memcheck --error-exitcode 1 \
  /home/taleef/.venvs/turboquant-forge/bin/python -m pytest \
  tests/test_qwen_parallel_page_kernel.py \
  tests/test_parallel_page_workspace.py -m gpu -q
python scripts/benchmark_parallel_page_attention.py \
  --implementations serial parallel \
  --page-counts 3 7 15 31 63 \
  --output artifacts/parallel-page-complete/manifest.json
python scripts/validate_artifact.py \
  artifacts/parallel-page-complete/manifest.json
git add kernels/qwen25_parallel_page_attention.py \
  core/parallel_page_workspace.py \
  tests/test_qwen_parallel_page_kernel.py \
  tests/test_parallel_page_workspace.py
git commit -m "feat: reduce parallel page attention deterministically"
```

---

### Task 6: Integrate the arena and page-parallel path into Qwen2.5

**Files:**
- Modify: `core/qwen_adapter.py`
- Modify: `core/packed_cache.py`
- Modify: `core/packed_store.py`
- Modify: `tests/test_qwen_adapter.py`
- Modify: `tests/test_qwen_adapter_gpu.py`
- Modify: `tests/test_packed_attention_hot_path.py`
- Add: `tests/test_qwen_parallel_page_integration.py`

- [ ] **Step 1: Write failing request-capacity tests**

Extend prefill/session setup with an explicit `maximum_sequence_length` or
equivalent `prompt_length + max_new_tokens` reservation. Test missing/invalid
capacity, exact-capacity success, prefill overflow, decode overflow, and
device/dtype transitions. Every rejection must happen before cache mutation.

- [ ] **Step 2: Write failing dispatch/stability tests**

Standard multi-token prefill still uses SDPA. Single-token decode uses the
parallel-page path at all 28 layers with zero fallback. Arena, tail, prepared
codec, and workspace pointers remain stable across decode and across a
tail-to-page flush. Adding a page within capacity does not rebuild a plan.

- [ ] **Step 3: Write failing end-to-end parity tests**

At synthetic 1K/2K layouts, compare the parallel direct output against packed
reference output and the current serial direct output. Record max/mean error,
top-1 agreement, and repeated-run determinism. Test final-position logit
retention and exact greedy token agreement for a bounded sequence.

- [ ] **Step 4: Run RED**

```bash
python -m pytest tests/test_qwen_adapter.py \
  tests/test_packed_attention_hot_path.py \
  tests/test_qwen_parallel_page_integration.py -q
python -m pytest tests/test_qwen_adapter_gpu.py \
  tests/test_qwen_parallel_page_integration.py -m gpu -q
```

- [ ] **Step 5: Switch only the qualified Qwen2.5 fast path**

Keep the public adapter boundary and reference fallback. Replace
`_prepared_layer_decode()` page snapshots with stable arena/prepared-constant
pointers plus runtime `used_pages` and `tail_length`. Allocate one
session-owned workspace and reuse it across layers. Retain the serial path as
an explicit diagnostic implementation until final evidence passes.

- [ ] **Step 6: Verify no hidden hot-path work**

Assert no per-page launch, page tuple rebuild, score tensor, decoded history,
constant copy, tail compaction, host synchronization, or hot-path allocation.
Launch counters remain constant with page count.

- [ ] **Step 7: Commit**

```bash
python -m pytest tests/test_qwen_adapter.py \
  tests/test_packed_attention_hot_path.py \
  tests/test_qwen_parallel_page_integration.py -q
python -m pytest tests/test_qwen_adapter_gpu.py \
  tests/test_qwen_parallel_page_integration.py -m gpu -q
python -m ruff check core kernels tests
git diff --check
git add core/qwen_adapter.py core/packed_cache.py core/packed_store.py \
  tests/test_qwen_adapter.py tests/test_qwen_adapter_gpu.py \
  tests/test_packed_attention_hot_path.py \
  tests/test_qwen_parallel_page_integration.py
git commit -m "feat: integrate Qwen parallel page attention"
```

---

### Task 7: Qualify fixed-token correctness before systems claims

**Files:**
- Add: `configs/benchmarks/qwen25-fixed-token-local-v1.json`
- Add: `scripts/run_fixed_token_replay.py`
- Add: `tests/test_fixed_token_replay.py`
- Modify: `scripts/README.md`

- [ ] **Step 1: Predeclare the replay matrix**

Use exact 1K and 2K tokenized inputs, fixed teacher-forced continuation tokens,
and the same pinned Qwen2.5 weights/profile across HF baseline, packed
reference, serial direct, and parallel direct. Record final logits, max/mean
logit error, KL divergence, top-1 agreement, greedy agreement, direct dispatch,
fallback count, and physical/runtime bytes.

- [ ] **Step 2: Write and run RED**

```bash
python -m pytest tests/test_fixed_token_replay.py -q
```

- [ ] **Step 3: Implement validator-first replay**

Use schema v2, exact row IDs, clean/dirty source identity, deterministic sidecar
hashes, and explicit invalid/OOM/unsupported rows. Do not silently shorten
tokenized inputs or compare different token streams.

- [ ] **Step 4: Run from a clean reviewed commit**

```bash
python scripts/run_fixed_token_replay.py \
  --config configs/benchmarks/qwen25-fixed-token-local-v1.json \
  --output-dir artifacts/qwen25-fixed-token-local-v1
python scripts/validate_artifact.py \
  artifacts/qwen25-fixed-token-local-v1/manifest.json
```

Expected: every expected row represented, direct/reference mismatch count zero,
parallel path direct-dispatched at all layers, no fallback, and declared logit
and greedy-agreement gates pass. If not, stop before systems qualification.

- [ ] **Step 5: Commit**

```bash
git add configs/benchmarks/qwen25-fixed-token-local-v1.json \
  scripts/run_fixed_token_replay.py tests/test_fixed_token_replay.py \
  scripts/README.md
git commit -m "test: qualify Qwen fixed-token replay"
```

---

### Task 8: Run the local quality and systems evidence ladder

**Files:**
- Modify: `scripts/run_direct_benchmark.py`
- Modify: `scripts/reproduce_release.py`
- Modify: `scripts/benchmark_schema.py`
- Modify: `tests/test_direct_benchmark.py`
- Modify: `tests/test_release_ladder.py`
- Modify: `tests/test_artifact_schema_v2.py`
- Add: `configs/benchmarks/qwen25-systems-local-v1.json`

- [ ] **Step 1: Write failing matrix/accounting tests**

Predeclare 1K/2K/4K/8K/16K systems rows, baseline/serial/parallel
implementations, one cold/JIT run, two warmups, ten interleaved trials, 64
decode tokens, and exact status precedence. Require phase-separated cold/JIT,
prefill, compression, TTFT, p50/p95/p99, throughput, peak
allocated/reserved, physical payload, arena capacity, constants, workspace,
tail, and unique-resident bytes.

- [ ] **Step 2: Write failing Pareto tests**

Compute paired trial ratios and deterministic confidence intervals. The
artifact, not documentation, decides whether the first local Pareto gate passes
at 8K or 16K. A valid regression remains a `failed` result, not `invalid` and
not omitted.

- [ ] **Step 3: Run RED**

```bash
python -m pytest tests/test_direct_benchmark.py \
  tests/test_release_ladder.py tests/test_artifact_schema_v2.py -q
```

- [ ] **Step 4: Implement runtime accounting and matrix execution**

Record allocator state by phase and reset peaks only at declared boundaries.
Capture `nvidia-smi`, competing processes, temperature, power, driver/runtime,
package lock, source, model, tokenizer, profile, and codec identities. Keep
physical used payload separate from reserved capacity and runtime workspace.

- [ ] **Step 5: Execute progressively**

Run 1K/2K first. Continue to 4K/8K only after correctness and no-fallback gates
pass. Attempt 16K only after a measured capacity preflight; preserve OOM or
capacity rejection as an explicit row.

```bash
python scripts/reproduce_release.py \
  --mode parallel-page-local \
  --config configs/benchmarks/qwen25-systems-local-v1.json \
  --output-dir artifacts/parallel-page-local-v1
python scripts/validate_artifact.py \
  artifacts/parallel-page-local-v1/manifest.json
```

- [ ] **Step 6: Independently audit numbers**

Recompute token counts, trial quantiles, confidence intervals, byte totals,
launch counts, pass-state counts, and the Pareto decision from raw rows. Check
that no diagnostic artifact or current-HEAD extrapolation entered the result.

- [ ] **Step 7: Commit runner/config changes**

```bash
git add scripts/run_direct_benchmark.py scripts/reproduce_release.py \
  scripts/benchmark_schema.py tests/test_direct_benchmark.py \
  tests/test_release_ladder.py tests/test_artifact_schema_v2.py \
  configs/benchmarks/qwen25-systems-local-v1.json
git commit -m "bench: qualify parallel page attention locally"
```

---

### Task 9: Publish the result without marketing drift

**Files:**
- Modify: `README.md`
- Modify: `docs/RESULTS.md`
- Modify: `docs/PAPER_CLAIMS_STATUS.md`
- Modify: `docs/BENCHMARK_PROTOCOL.md`
- Modify: `docs/CLAIMS.md`
- Modify: `tests/test_documentation_claims.py`
- Add: `docs/PARALLEL_PAGE_ATTENTION.md`

- [ ] **Step 1: Write failing claim-consistency tests**

Every numeric systems statement must resolve to the accepted schema-v2
artifact/row/query. Reject unqualified “faster,” “VRAM win,” “16K safe,”
“paper equivalent,” or multi-model statements. Require both wins and
regressions, exact pass-state counts, and separate logical/capacity/runtime
memory.

- [ ] **Step 2: Run RED**

```bash
python -m pytest tests/test_documentation_claims.py -q
```

- [ ] **Step 3: Write the architecture and evidence report**

Explain the previous serial-page bottleneck, arena ownership, page-state math,
chronological reduction, tail merge, workspace reuse, scope limitations,
microbenchmark ladder, fixed-token results, context-by-context systems rows,
quality gates, and Pareto decision. If the gate fails, publish the measured
failure and next diagnosed bottleneck without changing thresholds.

- [ ] **Step 4: Final verification**

```bash
python -m pytest tests -m "not gpu" -q
python -m pytest tests/test_qwen_parallel_page_kernel.py \
  tests/test_parallel_page_workspace.py \
  tests/test_qwen_parallel_page_integration.py \
  tests/test_qwen_adapter_gpu.py -m gpu -q
python -m ruff check .
python scripts/validate_artifact.py \
  artifacts/parallel-page-local-v1/manifest.json
git diff --check
git status --short
```

- [ ] **Step 5: Commit**

```bash
git add README.md docs/RESULTS.md docs/PAPER_CLAIMS_STATUS.md \
  docs/BENCHMARK_PROTOCOL.md docs/CLAIMS.md \
  docs/PARALLEL_PAGE_ATTENTION.md tests/test_documentation_claims.py
git commit -m "docs: publish parallel page attention evidence"
```

---

## Final acceptance checklist

- [ ] The fixed-capacity arena never grows or duplicates packed history.
- [ ] One-token appends allocate and copy no historical/tail tensor.
- [ ] Static codec constants exist once per layer and no stored page closure
      retains duplicates.
- [ ] Page-state, reduction, and tail kernels pass independent oracles and
      compute-sanitizer.
- [ ] Reduction is chronological, atomic-free, and repeatably deterministic.
- [ ] Attention launch count is constant with page count.
- [ ] Logical payload, arena capacity, tail, constants, workspace, padding, and
      unique resident bytes reconcile.
- [ ] Fixed-token replay qualifies before systems execution.
- [ ] The 1K/2K/4K/8K/16K matrix represents every expected row and reports all
      status counts.
- [ ] Public speed/VRAM language appears only if the predeclared Pareto gate
      passes; otherwise the measured regression remains visible.
- [ ] Qwen3, Qwen3.5, Gemma, serving, concurrency, and paper-equivalence claims
      remain explicitly unsupported by this plan.
