# TurboQuant Forge Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Deliver a clean-room, evidence-first compressed KV-attention backend for Qwen2.5 that uses physically packed block storage and direct GPU attention on the local RTX 5070 Ti, then prove the portable adapter boundary with Qwen3, Qwen3.5, and Gemma 4 capability manifests.

**Architecture:** A `ModelCapabilities` gate accepts only an explicitly supported Qwen2.5 full-attention greedy-decode configuration. A `PackedKVBlockStore` owns chronological, fixed-size, bit-packed K/V blocks plus an optional bounded fp16 recency window. The authoritative PyTorch codec and blockwise reference attention define correctness; Triton score/value kernels are required to match that reference without reconstructing the complete historical cache. A minimal Transformers attention backend dispatches standard prefill and packed decode through a scoped session. Reproducibility and benchmark runners create self-validating artifacts rather than inferred claims.

**Tech Stack:** Python 3.12, PyTorch CUDA, Triton, Hugging Face Transformers, pytest, NumPy/SciPy, safetensors, Qwen2.5-7B-Instruct (pinned revision), RTX 5070 Ti (SM120).

## Global Constraints

- Keep the existing MIT license and implement every new codec, kernel, adapter, and benchmark clean-room. Do not copy GPL-3.0 code or structures from `0xSero/turboquant`.
- Treat `core/turboquant_cache_v2.py` as a legacy fidelity prototype after the replacement path lands. Never describe its full-history dequantization as a direct attention result.
- First supported runtime is Qwen2.5 text, batch size one, greedy decode, static full attention, and one CUDA device. All other architecture or generation modes must raise `UnsupportedCapabilityError` before cache mutation.
- Store exact physical bytes (`numel() * element_size()` for tensors plus declared alignment padding); never report nominal bits as memory savings.
- A benchmark artifact is invalid on a missing matrix row, OOM, exception, tokenization loss, mutable model revision, or incomplete environment metadata. Invalid artifacts may be retained for diagnosis but cannot support a claim.
- Use `sys.executable` for all Python subprocesses. CPU tests must run without a model download or CUDA runtime. GPU tests are marked `gpu` and skip only when CUDA/Triton requirements are absent.
- Pin runtime versions in a generated lock file only after the RTX 5070 Ti smoke test succeeds. Record package, model, codebook, profile, and source hashes in every qualified artifact.
- Do not introduce a server, vLLM patch, paging/offload service, beam search, speculative/MTP path, distributed path, or multimodal cache path in this milestone.

---

## 1. Establish a reproducible, CPU-testable project contract

**Files:**
- Add: `pyproject.toml`
- Add: `requirements/runtime.in`
- Add: `requirements/dev.in`
- Add: `requirements/runtime.lock`
- Add: `tests/conftest.py`
- Add: `tests/test_environment.py`
- Add: `.github/workflows/ci.yml`
- Modify: `.gitignore`
- Modify: `README.md`
- Modify: `SETUP_CUDA.md`
- Modify: `WSL2_SETUP.md`
- Modify: `scripts/repro_utils.py`

- [ ] **Step 1: Write failing environment and metadata tests.**

  Create `tests/test_environment.py` that asserts the package-test layout, `capture_env_metadata()` returns the active executable, Python version, package versions when importable, deterministic flags, and a stable SHA-256 helper. Do not use a GPU or download a model. Add the `gpu` pytest marker in `pyproject.toml`.

  ```python
  def test_capture_env_metadata_uses_running_interpreter(monkeypatch):
      metadata = capture_env_metadata()
      assert metadata["python_executable"] == sys.executable
      assert metadata["python_version"] == platform.python_version()
      assert "torch_version" in metadata

  def test_sha256_file_is_stable(tmp_path):
      target = tmp_path / "profile.json"
      target.write_text('{"name":"test"}', encoding="utf-8")
      assert sha256_file(target) == sha256_file(target)
  ```

  Run: `python -m pytest tests/test_environment.py -q`  
  Expected: initially fails because metadata helpers and test configuration do not exist.

- [ ] **Step 2: Define minimal packaging and dependency inputs.**

  Add a `pyproject.toml` with project metadata, `requires-python = ">=3.11"`, pytest `testpaths = ["tests"]`, `pythonpath = ["."]`, strict marker registration, and ruff configuration only if ruff is installed by the dev input. Create `requirements/runtime.in` for the runtime libraries (`torch`, `transformers`, `triton`, `accelerate`, `bitsandbytes`, `safetensors`, `numpy`, `scipy`, `einops`) and `requirements/dev.in` that extends it with `pytest` and `ruff`. Keep `requirements.txt` as a compatibility pointer to the generated runtime lock rather than maintaining a competing dependency list.

  Generate `requirements/runtime.lock` from a successful WSL CUDA environment with exact versions and the matching PyTorch CUDA index URL documented in the file header. Record the generation interpreter and `pip --version`; do not hand-author unavailable CUDA wheel versions. The lock must be regenerated only by the documented command.

- [ ] **Step 3: Make reproducibility metadata complete.**

  Expand `scripts/repro_utils.py` with `sha256_file(path)`, `package_version(name)`, `seed_everything(seed, deterministic)`, `capture_cuda_metadata()`, and an explicit `capture_env_metadata()` schema. It must record Python executable/version, operating system, torch/transformers/triton/bitsandbytes/numpy/scipy versions, CUDA runtime, GPU name, SM capability, total VRAM, driver version when available, TF32/determinism settings, and Git commit. Missing optional packages must be explicit `null`, never silently omitted.

- [ ] **Step 4: Create a safe CPU CI lane and repair guides.**

  Add a GitHub Actions workflow that installs the CPU-compatible development dependencies and runs `python -m pytest tests -m "not gpu" -q`. Update setup guides to create the WSL virtual environment, install the generated lock, run an import smoke test, and print `nvidia-smi`; remove references to deleted `scripts/run_turboquant.py` and unpinned nightly installation. Ignore generated benchmark artifacts under `artifacts/` while allowing checked-in fixtures.

  Run: `python -m pytest tests/test_environment.py -q`  
  Run: `python -m compileall core scripts tests`  
  Expected: both exit 0 on CPU; CI configuration only selects non-GPU tests.

- [ ] **Step 5: Commit the reproducibility foundation.**

  ```bash
  git add pyproject.toml requirements README.md SETUP_CUDA.md WSL2_SETUP.md \
      scripts/repro_utils.py tests .github/workflows/ci.yml .gitignore
  git commit -m "build: add reproducible test and environment contract"
  ```

## 2. Add explicit model capabilities and fail-closed profiles

**Files:**
- Add: `core/errors.py`
- Add: `core/capabilities.py`
- Add: `core/profiles.py`
- Add: `configs/profiles/qwen25-7b-direct-v1.json`
- Add: `configs/models/qwen25-7b.json`
- Add: `configs/models/qwen3.json`
- Add: `configs/models/qwen35-9b.json`
- Add: `configs/models/gemma4-e4b.json`
- Add: `tests/test_capabilities.py`
- Add: `tests/test_profiles.py`
- Modify: `core/__init__.py`
- Modify: `README.md`

- [ ] **Step 1: Write capability rejection tests before the implementation.**

  Test a small synthetic Qwen2 configuration, a known Qwen3.5 hybrid configuration, a Gemma 4 configuration, and unsupported variants. Required failures: batch size not one, non-greedy generation, unknown model type, sliding-window layers, linear-only layers, MTP/speculative mode, and an adapter request for a multimodal cache. Verify failure happens before a supplied store's `append()` can be called.

  ```python
  def test_unknown_model_rejects_before_cache_mutation():
      capabilities = derive_model_capabilities(SimpleNamespace(model_type="unknown"))
      with pytest.raises(UnsupportedCapabilityError, match="model_type"):
          require_direct_decode_support(capabilities, DecodeRequest(batch_size=1))

  def test_qwen35_exposes_only_full_attention_as_compressible():
      caps = derive_model_capabilities(qwen35_hybrid_config())
      assert caps.compressible_layer_indices == (3, 7)
      assert caps.native_state_layer_indices == (0, 1, 2, 4, 5, 6)
  ```

  Run: `python -m pytest tests/test_capabilities.py -q`  
  Expected: fails until capability types and rules exist.

- [ ] **Step 2: Implement immutable capability and profile types.**

  Create `UnsupportedCapabilityError`. In `core/capabilities.py`, implement frozen dataclasses `ModelCapabilities`, `DecodeRequest`, and `ModelRevision`; make `derive_model_capabilities(config, revision)` read architecture facts from the pinned config rather than hard-code them in benchmark scripts. Record model type, layer count, layer types, full/sliding/linear layer index sets, Q and KV heads, head dimension, GQA grouping, RoPE settings, multimodal flag, and cache behavior. Implement `require_direct_decode_support()` for the first milestone’s Qwen2.5 full-attention route.

  In `core/profiles.py`, implement a frozen `CompressionProfile` and `load_profile(path)` that validates `schema_version`, profile hash, supported bit widths, group size divisibility, block size, fp16-window bounds, and the target model manifest hash. A profile must never silently default missing fields.

- [ ] **Step 3: Check in model manifests and one direct profile.**

  Add compact JSON manifests containing only architecture and revision fields necessary for validation. The Qwen3.5 manifest must mark its three-linear/one-full pattern, and Gemma 4 must be marked `adapter_status: "manifest-only"` until its dedicated attention adapter has tests. Add `qwen25-7b-direct-v1.json` with schema version 1, fixed block size 256, key bits 8, value bits 6, value group size 32, fp16 window 128, the model revision, and hashes populated by the profile-generation command.

  Run: `python -m pytest tests/test_capabilities.py tests/test_profiles.py -q`  
  Expected: exit 0; every unsupported route includes an actionable error message.

- [ ] **Step 4: Document the model ladder truthfully.**

  Update the README support table to distinguish `direct adapter`, `manifest-only`, and `not supported`. State: Qwen3.5 compresses only `full_attention` KV blocks while linear recurrent state remains native; Gemma 4 requires its own adapter and capacity preflight; no multimodal processor/vision cache participates in KV-memory claims.

- [ ] **Step 5: Commit model gating.**

  ```bash
  git add core configs tests README.md
  git commit -m "feat: add fail-closed model capability profiles"
  ```

## 3. Build the physically packed reference codec and append-only block store

**Files:**
- Add: `core/packing.py`
- Add: `core/codec.py`
- Add: `core/packed_store.py`
- Add: `tests/test_packing.py`
- Add: `tests/test_codec.py`
- Add: `tests/test_packed_store.py`
- Modify: `core/turboquant_simple.py`
- Modify: `core/turboquant_cache_v2.py`
- Modify: `core/__init__.py`

- [ ] **Step 1: Write bit-exact packing tests for supported index widths.**

  `tests/test_packing.py` must parameterize bits `2, 3, 4, 5, 6, 7, 8`, dimensions `64, 128, 256`, and tail dimensions that do not end on a byte boundary. It must establish the exact payload length `ceil(count * bits / 8)`, exact round-trip, zeroed padding bits, rejection of values outside `[0, 2**bits)`, and rejection of invalid dtype/device shapes.

  ```python
  @pytest.mark.parametrize("bits", [2, 3, 4, 5, 6, 7, 8])
  def test_pack_indices_round_trip_and_bytes(bits):
      indices = torch.randint(0, 1 << bits, (3, 129), dtype=torch.uint8)
      packed = pack_indices(indices, bits)
      assert packed.shape[-1] == math.ceil(129 * bits / 8)
      assert torch.equal(unpack_indices(packed, bits, 129), indices)
  ```

  Run: `python -m pytest tests/test_packing.py -q`  
  Expected: fails until generic packers replace the isolated four-bit helpers.

- [ ] **Step 2: Implement pure-PyTorch packing, codecs, and accounting.**

  In `core/packing.py`, implement vectorized `pack_indices`, `unpack_indices`, `pack_sign_bits`, `unpack_sign_bits`, and `tensor_nbytes`. Every function must preserve the caller device and use `uint8` payloads. Do not reuse the legacy `kernels/bitpack_triton.py` as the new API; retain it only until its callers have migrated.

  In `core/codec.py`, introduce immutable layouts and payload dataclasses:

  ```python
  @dataclass(frozen=True)
  class KeyLayout:
      head_dim: int
      bits: int
      norm_dtype: torch.dtype
      qjl_enabled: bool = False

  @dataclass
  class PackedKeyPayload:
      indices: torch.Tensor
      norms: torch.Tensor
      qjl_signs: torch.Tensor | None
      qjl_gamma: torch.Tensor | None
      logical_tokens: int
  ```

  `KeyCodec` must use the existing Lloyd-Max codebook and rotation semantics but output physically packed indices. `ValueCodec` must group-quantize values with packed indices, `mins`, and `scales`. QJL, when selected, must use `bits - 1` MSE bits plus a separately packed one-bit sign payload—not an `int8` sign matrix. Each payload exposes `physical_nbytes()` and `unpack_reference()` for tests only.

- [ ] **Step 3: Test codec numerical behavior and QJL representation.**

  Move and adapt active V2 tests into `tests/test_codec.py`. Test exact discrete unpacking, key/value reconstruction dtype, finite output for zero vectors, group metadata dimensions, and physical bytes that include every metadata tensor. Add a test that a QJL payload’s sign data consumes `ceil(tokens * qjl_dim / 8)` bytes, not one byte per sign.

- [ ] **Step 4: Implement append-only fixed-size blocks.**

  `PackedKVBlockStore` owns a list of `PackedKVBlock` instances. Its public surface is deliberately small:

  ```python
  class PackedKVBlockStore:
      def append(self, keys: torch.Tensor, values: torch.Tensor) -> None: ...
      def iter_blocks(self) -> Iterator[PackedKVBlock]: ...
      def iter_fp16_window(self) -> Iterator[tuple[torch.Tensor, torch.Tensor]]: ...
      def sequence_length(self) -> int: ...
      def physical_nbytes(self) -> PhysicalByteBreakdown: ...
  ```

  Accept only `[1, num_kv_heads, tokens, head_dim]`, preserve incoming fp16/bf16 in the bounded window, and encode complete blocks of 256 tokens. Keep only the final bounded tail unencoded until it fills; flush exactly one complete block at a time. Never concatenate all historical blocks, and never call `torch.cat` on historical storage. `PhysicalByteBreakdown` separates key payload, value payload, metadata, fp16 window, and padding.

- [ ] **Step 5: Prove storage behavior and preserve legacy compatibility boundaries.**

  In `tests/test_packed_store.py`, append randomized token chunks across block boundaries and assert: sequence order is preserved, all stored tensor bytes equal the reported breakdown, only one bounded tail/window exists, append does not mutate prior block payload data pointers, and no helper reconstructs all blocks. Add explicit regression tests that `TurboQuantCacheV2` remains marked legacy and cannot be selected by the new profile loader.

  Run: `python -m pytest tests/test_packing.py tests/test_codec.py tests/test_packed_store.py -q`  
  Expected: exit 0 on CPU.

- [ ] **Step 6: Commit packed storage.**

  ```bash
  git add core tests
  git commit -m "feat: add packed KV codec and block store"
  ```

## 4. Establish blockwise reference attention before GPU kernels

**Files:**
- Add: `core/reference_attention.py`
- Add: `tests/test_reference_attention.py`
- Modify: `core/codec.py`
- Modify: `core/packed_store.py`

- [ ] **Step 1: Write reference-equivalence tests.**

  Compare blockwise compressed reference attention to the same *unpacked quantized representation*, not to original fp16 K/V. Cover GQA (`num_attention_heads / num_key_value_heads`), head dimensions 64/128/256, block tails 1/255/256, fp16 and bf16 inputs, and mixed packed-block plus fp16-window history. The output must agree with `torch.nn.functional.scaled_dot_product_attention` over the quantized reference within declared tolerances.

  ```python
  def test_blockwise_reference_matches_unpacked_quantized_attention():
      output = packed_reference_decode(query, store, layout, scale=1 / math.sqrt(128))
      expected = sdpa_on_unpacked_quantized_history(query, store, layout)
      torch.testing.assert_close(output, expected, rtol=2e-3, atol=2e-3)
  ```

  Run: `python -m pytest tests/test_reference_attention.py -q`  
  Expected: fails before blockwise online-softmax is implemented.

- [ ] **Step 2: Implement direct reference key scoring and value mixing.**

  `packed_reference_decode()` must iterate each packed block, rotate the query using the matching key-codec rotation, unpack only that block, calculate GQA-aware scores, and update the stable online-softmax state `(running_max, running_sum, weighted_value_sum)`. It must consume the bounded fp16 window as a final small segment. It may allocate a single decoded block but must reject any request to concatenate full history.

  The formula for each score block is:

  ```python
  scores = (rotated_query @ decoded_rotated_keys.transpose(-1, -2))
  scores = scores * key_norms.unsqueeze(-2) * attention_scale
  ```

  Update online state with `new_max = maximum(old_max, scores.max(-1))`, rescaled sums, and value-weighted accumulation. Keep the entire operation in fp32 accumulation, then cast to the requested model dtype at the boundary.

- [ ] **Step 3: Add invariants and cross-check accounting.**

  Add an optional debug guard that tracks the largest temporary tensor and asserts it is bounded by one block (plus the query and output). Tests must prove that inserting an extra old block changes attention output but not the maximum temporary decoded-token count. Add reference output/logit drift fixtures for a deterministic tiny attention case; no “exact fp16 equivalence” claim is permitted for the lossy codec.

  Run: `python -m pytest tests/test_reference_attention.py tests/test_packed_store.py -q`  
  Expected: exit 0 on CPU.

- [ ] **Step 4: Commit the reference backend.**

  ```bash
  git add core/reference_attention.py core/codec.py core/packed_store.py tests
  git commit -m "feat: add blockwise packed attention reference"
  ```

## 5. Implement and verify Triton packed score and value-mix kernels

**Files:**
- Add: `kernels/packed_attention.py`
- Add: `tests/test_packed_attention_kernels.py`
- Add: `tests/test_packed_attention_benchmark.py`
- Modify: `kernels/__init__.py`
- Modify: `pyproject.toml`

- [ ] **Step 1: Write GPU-only parity tests first.**

  Tests must skip with an explicit reason only if CUDA or Triton is unavailable. On the RTX 5070 Ti they must parameterize key bits `(6, 8)`, value bits `(4, 6, 8)`, head dimensions `(64, 128, 256)`, KV heads `(1, 2, 8)`, GQA ratios `(1, 4, 8)`, tokens `(1, 17, 255, 256)`, and fp16/bf16 inputs. Compare each kernel output against its direct packed PyTorch reference; record tested tolerance in the test name/result.

  ```python
  @pytest.mark.gpu
  def test_packed_score_kernel_matches_reference(case):
      actual = packed_key_scores(query, packed_keys, norms, rotation, case)
      expected = packed_key_scores_reference(query, packed_keys, norms, rotation, case)
      torch.testing.assert_close(actual, expected, rtol=3e-3, atol=3e-3)
  ```

  Run: `python -m pytest tests/test_packed_attention_kernels.py -m gpu -q`  
  Expected: fails before kernel dispatch exists; skips only on a machine without the GPU runtime.

- [ ] **Step 2: Implement key-score kernel with generic packed decoding.**

  Create a new clean-room `kernels/packed_attention.py` instead of modifying the legacy experimental kernels. Provide `packed_key_scores()` and `packed_key_scores_reference()`. The kernel receives byte-packed index rows, codebook, query rotation, per-token norm, GQA mapping, and an attention scale. Decode indices on demand with bit shifts and masks into registers/tiles; do not materialize decoded keys for an entire block. Use `@triton.jit` and a small explicit autotune configuration set keyed by head dimension and token tile size. Provide a clear PyTorch fallback only for unit/reference use, not silently inside a qualified GPU benchmark.

- [ ] **Step 3: Implement stable value mixing across blocks.**

  Implement `packed_value_mix()` and an online-state kernel boundary that accepts block scores, packed grouped value indices, mins, scales, prior `(max, denominator, numerator)` state, and GQA mapping. Dequantize only the active tile, accumulate fp32, and return updated state. The public `packed_attention_decode()` coordinates scores, online softmax, values, and the fp16 window without an all-history allocation.

- [ ] **Step 4: Benchmark kernel mechanics separately from model claims.**

  `tests/test_packed_attention_benchmark.py` uses CUDA events and `triton.testing.do_bench` where available after warmup. Emit JSON with shapes, data type, GPU metadata, source commit, median/p95 latency, and reference tolerance. It must label results `kernel_microbenchmark`, never an end-to-end throughput result. Add a documented Compute Sanitizer command for CI-independent manual validation.

  Run: `python -m pytest tests/test_packed_attention_kernels.py -m gpu -q`  
  Run: `python -m pytest tests/test_packed_attention_benchmark.py -m gpu -q`  
  Expected: exit 0 on the 5070 Ti; the JSON includes all requested case dimensions and no fallback flag.

- [ ] **Step 5: Commit GPU kernels only after parity passes.**

  ```bash
  git add kernels tests pyproject.toml
  git commit -m "feat: add verified Triton packed attention kernels"
  ```

## 6. Integrate the direct backend with Qwen2.5 greedy decode

**Files:**
- Add: `core/packed_cache.py`
- Add: `core/qwen_adapter.py`
- Add: `tests/test_qwen_adapter.py`
- Add: `tests/test_qwen_adapter_gpu.py`
- Modify: `core/capabilities.py`
- Modify: `core/__init__.py`
- Modify: `README.md`

- [ ] **Step 1: Write adapter contract tests using a local fake attention module.**

  Test prefill (`query_length > 1`) dispatches standard attention while encoding its result cache into the packed store; test the first decode token and all later tokens use `packed_attention_decode`; verify model layers use per-layer rotations; verify batch >1, beam search, `use_cache=False`, unsupported layer types, device moves, and dtype changes fail before cache modification. Use a fake module/config to keep CPU tests download-free.

  ```python
  def test_decode_dispatches_packed_backend_after_standard_prefill(fake_qwen):
      session = install_qwen25_packed_backend(fake_qwen, profile, capabilities)
      session.prefill(prompt_ids)
      session.decode_one()
      assert session.metrics.standard_prefill_calls == 1
      assert session.metrics.packed_decode_calls == 1
  ```

  Run: `python -m pytest tests/test_qwen_adapter.py -q`  
  Expected: fails until cache/session dispatch exists.

- [ ] **Step 2: Implement a narrow Transformers attention-session adapter.**

  In `core/packed_cache.py`, provide `PackedDynamicCache` only for the supported direct profile. It owns one `PackedKVBlockStore` per compressible layer and encodes post-RoPE K/V tensors. It must retain a bounded fp16 prefill/tail segment only and expose no misleading `DynamicCache` compatibility promise.

  In `core/qwen_adapter.py`, add `install_qwen25_packed_backend(model, profile, revision)` and a context-managed `PackedAttentionSession`. Register the custom Transformers attention implementation through its documented attention interface when the pinned Transformers version exposes it. The attention dispatch must use standard causal attention for prompt prefill, and direct packed attention only for `q_len == 1` greedy decode. Scope active cache state with a context variable rather than a process-global mutable singleton. If the installed Transformers version lacks the required interface or Qwen module structure, fail with an exact minimum-version/unsupported-layout message rather than monkey-patching unverified internals.

- [ ] **Step 3: Add real model GPU integration tests.**

  Pin one small Qwen2.5 prompt fixture and model revision. Mark tests GPU and allow explicit environment opt-in `TURBOQUANT_RUN_MODEL_TESTS=1` to avoid accidental downloads. Validate: the adapter consumes packed blocks, no full-history `torch.cat` is invoked (instrument/cache metric), greedy output matches the packed-reference backend token-for-token, model logits are finite, and the lossy profile’s max logit drift from fp16 is emitted rather than hidden. Include a 1K-token synthetic prefill plus 64-token decode smoke test when memory allows.

- [ ] **Step 4: Document exact first-runtime boundaries.**

  README must state that Qwen2.5 direct support is batch-one greedy decode only; prefill is standard attention; direct decode is a measured backend; and Qwen3/Qwen3.5/Gemma 4 are configuration manifests rather than completed adapters until their test suite passes.

  Run: `python -m pytest tests/test_qwen_adapter.py -q`  
  Run: `TURBOQUANT_RUN_MODEL_TESTS=1 python -m pytest tests/test_qwen_adapter_gpu.py -m gpu -q`  
  Expected: CPU contract pass; GPU test either passes with a pinned model snapshot or reports a non-qualified dependency failure (never skips after explicit opt-in).

- [ ] **Step 5: Commit the Qwen2.5 adapter.**

  ```bash
  git add core tests README.md
  git commit -m "feat: integrate packed decode for Qwen2.5"
  ```

## 7. Replace permissive benchmarks with self-validating evidence runs

**Files:**
- Add: `scripts/run_direct_benchmark.py`
- Add: `scripts/validate_artifact.py`
- Add: `scripts/benchmark_schema.py`
- Add: `tests/test_benchmark_schema.py`
- Add: `tests/test_artifact_validation.py`
- Modify: `scripts/test_long_context.py`
- Modify: `scripts/reproduce_release.py`
- Modify: `scripts/repro_utils.py`
- Modify: `configs/profiles/qwen25-7b-direct-v1.json`
- Modify: `.gitignore`

- [ ] **Step 1: Write failing artifact-schema and invalid-run tests.**

  Define typed JSON schemas for `quality_matrix`, `systems_benchmark`, and `qualified_run`. Tests must fail a row that is missing, has `status != "completed"`, lost a needle during tokenization, recorded an OOM/exception, has a changed model/profile/codebook hash, or lacks exact expected matrix dimensions. Test paired bootstrap implementation with a fixed fixture and declared seed.

  ```python
  def test_artifact_rejects_omitted_matrix_row(complete_matrix):
      complete_matrix["rows"].pop()
      result = validate_quality_artifact(complete_matrix)
      assert result.valid is False
      assert "expected row count" in result.errors[0]
  ```

  Run: `python -m pytest tests/test_benchmark_schema.py tests/test_artifact_validation.py -q`  
  Expected: fails until schemas and validators exist.

- [ ] **Step 2: Make the NIAH matrix complete-or-invalid.**

  Refactor `scripts/test_long_context.py` to use repository-root imports and structured rows. Build the expected `(context, depth, seed)` matrix before execution. Persist every row including failed rows; return non-zero on any missing/error/OOM/tokenization-loss row. Require baseline accuracy >= 95%, calculate paired baseline-minus-profile accuracy, and calculate a deterministic 95% bootstrap upper confidence bound from the complete eligible matrix. Do not call a profile retrieval-safe unless the upper bound is <= 2pp.

- [ ] **Step 3: Implement an honest systems runner.**

  `scripts/run_direct_benchmark.py` must run interleaved baseline/packed-reference/Triton-direct trials with fixed prompts, warmups, CUDA events, and explicit synchronization. It must report actual store bytes, per-category physical byte breakdown, `torch.cuda.max_memory_allocated`, `max_memory_reserved`, prefill latency, TTFT, per-token decode latencies, p50/p95, decode tok/s, OOM boundary, and backend dispatch counts. The runner must not call the legacy V2 cache. It writes artifacts under `artifacts/<timestamp>-<commit>/` with captured stdout/stderr and hashes.

- [ ] **Step 4: Replace the release runner with a validator-first command.**

  Make `scripts/reproduce_release.py` call child scripts via `sys.executable`, collect outputs even if a child fails, validate every artifact, and return non-zero unless both quality and systems results are valid. `quick` is a labeled 8K smoke run; `qualified` runs the complete target matrix only after a VRAM preflight. The report must distinguish `qualified`, `invalid`, `failed`, and `inconclusive`; it must not convert a missing row into a passing delta.

- [ ] **Step 5: Execute the evidence ladder on the local GPU.**

  Run a preflight at 1K prompt/64 decode, then 4K and 8K quality and systems matrices. Attempt the configured 32K qualified matrix only after the capacity estimate and 8K pass; retain OOM as an invalid/inconclusive artifact instead of relaxing the gate. Archive raw JSON/Markdown/CSV outputs and command logs under the generated artifact directory.

  Run: `python -m pytest tests/test_benchmark_schema.py tests/test_artifact_validation.py -q`  
  Run: `python scripts/reproduce_release.py --mode quick --output-dir artifacts/quick`  
  Run: `python scripts/reproduce_release.py --mode qualified --output-dir artifacts/qualified`  
  Expected: schema tests exit 0; each release command either produces a validator-approved artifact or exits non-zero with a preserved explanation.

- [ ] **Step 6: Commit benchmark correctness before claiming a result.**

  ```bash
  git add scripts configs tests .gitignore
  git commit -m "feat: add self-validating quality and systems benchmarks"
  ```

## 8. Reconcile documentation, claims, and portability evidence

**Files:**
- Modify: `README.md`
- Modify: `TESTING_RESULTS.md`
- Modify: `docs/PAPER_CLAIMS_STATUS.md`
- Modify: `docs/PAPER_COMPARISON.md`
- Modify: `docs/REPLICATION_CHECKLIST.md`
- Modify: `ISSUES.md`
- Add: `docs/PORTABILITY.md`
- Add: `docs/BENCHMARK_PROTOCOL.md`
- Add: `docs/RESULTS.md`
- Add: `tests/test_documentation_claims.py`

- [ ] **Step 1: Write a claims-consistency test.**

  Add a lightweight test that rejects stale strings such as `5.2x`, `4.5x`, `status: met_for_4k_8k_16k_32k_t6`, deleted script paths, and a 32K success statement that lacks a linked qualified artifact ID. Require README links to the direct profile and benchmark protocol.

  ```python
  def test_docs_do_not_contain_unqualified_legacy_memory_claims():
      corpus = read_docs(("README.md", "TESTING_RESULTS.md", "docs"))
      assert "5.2x compression" not in corpus
      assert "scripts/run_turboquant.py" not in corpus
  ```

- [ ] **Step 2: Publish a candid result and portability surface.**

  Rewrite result documents to separate legacy prototype observations from direct-backend evidence. State physical bytes, peak memory, and latency only from artifact IDs. Include both wins and regressions. `docs/PORTABILITY.md` must document the `ModelCapabilities` rules, Qwen3 dense validation path, Qwen3.5 full-vs-linear handling, Gemma 4 adapter preflight, and the fact that model/vision processing state is excluded from KV-cache claims. `docs/BENCHMARK_PROTOCOL.md` documents artifacts, environments, matrix completeness, statistical gates, and invalid states.

- [ ] **Step 3: Update tracker-style documents from code truth.**

  Mark issues as resolved only where a test and implementation exists. Keep the missing adapter and profile promotions explicitly open. Do not report Qwen3/Qwen3.5/Gemma 4 integration as complete solely because manifests parse.

  Run: `python -m pytest tests/test_documentation_claims.py -q`  
  Run: `git diff --check`  
  Expected: both exit 0; every public performance or memory number has an artifact reference.

- [ ] **Step 4: Perform the final verification sweep and commit.**

  ```bash
  python -m pytest tests -m "not gpu" -q
  python -m pytest tests -m gpu -q
  python scripts/validate_artifact.py artifacts/qualified
  git diff --check
  git status --short
  git add README.md TESTING_RESULTS.md docs ISSUES.md tests
  git commit -m "docs: publish verified TurboQuant Forge evidence"
  ```

  Expected: CPU suite passes; GPU suite passes on the supported environment; artifact validator reports `qualified` only if all gates hold. If any quality or systems gate fails, commit the truthful `invalid`/`inconclusive` artifact and documentation, not a success claim.

## Final acceptance checklist

- [ ] A clean WSL installation from the lock file imports PyTorch, Transformers, Triton, NumPy, SciPy, and pytest and sees the RTX 5070 Ti.
- [ ] CPU packing, codec, store, capability, reference-attention, benchmark-schema, and documentation tests pass without model downloads.
- [ ] GPU kernel parity proves direct packed scores/value mix against the packed PyTorch reference across every supported shape.
- [ ] Qwen2.5 adapter uses standard prefill and direct packed greedy decode without full-history K/V reconstruction.
- [ ] Qualified artifacts include complete quality rows, baseline threshold, bootstrap upper confidence bound, actual physical cache bytes, allocator peaks, timing distribution, environment/model/profile/codebook hashes, and raw command logs.
- [ ] README and result documents distinguish verified direct-backend evidence from legacy prototype results and never extrapolate unsupported model or hardware claims.
- [ ] Qwen3, Qwen3.5, and Gemma 4 portability state is explicitly correct: manifests and fail-closed gates exist; adapter promotion requires its own test/evidence artifact.
