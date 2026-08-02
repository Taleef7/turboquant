# Gilbreth handoff

**Checked:** 2026-08-02
**Purpose:** resume TurboQuant Forge on Purdue Gilbreth without turning a
diagnostic or capacity failure into a qualification claim.

Read this page before starting an agent, copying artifacts, or selecting a
partition. `docs/BENCHMARK_PROTOCOL.md` and the artifact validators remain the
authority for evidence status.

## Exact handoff state

The active worktree is on `codex/turboquant-forge` at `cd502e2` (documentation
publication). The branch is not pushed to `origin`. The only intentionally
dirty path is the pre-existing protected `.superpowers/sdd/progress.md`; do
not stage, overwrite, or reset it. Clean code/evidence is bound to
`eba1c5b0239971ab2b78e1acceadda0885b6cb28`; `c0b922c` and `cd502e2` add the
fused path and its documentation. A new clone will not see this branch until
it is pushed or transferred as a bundle.

```bash
# preferred, after reviewing the protected file and local diff
git push -u origin codex/turboquant-forge

# or transfer without pushing
git bundle create turboquant-forge.bundle main..codex/turboquant-forge
git clone turboquant-forge.bundle turboquant
cd turboquant
git checkout codex/turboquant-forge
```

The `artifacts/` directory is ignored by Git. Copy raw JSON, logs, fixture
manifests, and SHA-256 values separately. A Markdown summary without its raw
artifact is unavailable evidence and must be rerun.

The only direct adapter is the pinned `Qwen/Qwen2.5-7B-Instruct` text model:
batch size one, one CUDA device, full-attention layers, standard prefill, and
greedy one-token decode. Qwen3, Qwen3.5, and Gemma 4 are manifest-only,
fail-closed portability entries, not supported adapters.

## Evidence tracks

The historical release artifact
`qualified-semantic-final/20260718T074435Z-5c9d4c34b0f0` (bound to
`5c9d4c34b0f0c31d015e2166ae7ad401858b202d`) is structurally valid and
qualifies its exact 4K/8K NIAH quality matrix. Its top-level status is
`failed`: 32K admission projected `12,809,961,472` bytes against an allowed
`10,405,439,078`, so no 32K child ran. Its 8K systems comparison is negative
(1.29 versus 33.99 tok/s and higher allocator peaks for direct). It is not a
32K qualification or an end-to-end win.

Latest clean local diagnostics are bound to code commit `eba1c5b` (publication
commit `cd502e2`):

| Raw artifact | Scope and honest result |
| --- | --- |
| `qwen25-fixed-warmup-local-v2-clean-eba1c5b.json` | Qwen2.5 4K/8K; 8/8 rows complete, 2 baseline rows pass strict gates and all 6 packed rows fail full-logit gates. Cache-store savings: 36.9% (4K), 44.8% (8K). |
| `local-page-fused-clean-eba1c5b.json` | Synthetic 3–63 pages; fused is 25.2–79.5% faster than serial, deterministic, and within `1.05e-05` max error. Not model-level evidence. |
| `qwen25-value-ablation-local-v1-clean-eba1c5b.json` | Qwen2.5 v6/v8 values at 4K/8K; both fail strict full-logit gates; v8 is about 12% larger and 49–59.5% slower. |

The warmup p50 is mixed: parallel is 40.31 ms versus 50.10 ms at 4K
(`-19.5%`) but 42.90 ms versus 33.10 ms at 8K (`+29.6%`). Cache-store bytes
and CUDA allocator peaks answer different questions. See
[local long-context](LOCAL_LONG_CONTEXT_RESULTS.md),
[fused](LOCAL_FUSED_PAGE_RESULTS.md), and
[value ablation](LOCAL_VALUE_BIT_ABLATION.md).

The local fixture is snapshot `a09a35458c702b33eeacc393d103063234e8bc28`,
with PyTorch `2.7.0+cu128`, Transformers `4.57.3`, Triton `3.3.0`, and Python
3.12.3. The original GPU was an RTX 5070 Ti Laptop GPU with
`12,820,480,000` reported bytes. These are provenance values, not a promise
that the same CUDA wheel works on Gilbreth.

## Gilbreth operating rules

Gilbreth is Rocky Linux 9 and Slurm-managed, with heterogeneous A10, A30,
A100-40GB, A100-80GB, and H100 nodes. Verify current hardware and account
permissions:

```bash
ssh <user>@gilbreth.rcac.purdue.edu
myquota
slist
sfeatures
module avail
module spider cuda
echo "$RCAC_SCRATCH"
```

Never run a benchmark, model download, compilation, or long test on the login
node. Every job needs an account (`-A`), partition (`-p`), QOS (`--qos`),
resource request, and wall time. The partition selects the GPU type; request
the count with `--gpus-per-node=1` and do not assume a GPU-type `--gres` string.

Short debugging only (replace `ACCOUNT` and verify partition/QOS):

```bash
sinteractive -A ACCOUNT -p a100-40gb --qos=standby \
  --nodes=1 --ntasks=1 --gpus-per-node=1 --cpus-per-task=8 \
  --mem=64G --time=01:00:00
```

For auditable work use a batch script:

```bash
#!/usr/bin/env bash
#SBATCH -A ACCOUNT
#SBATCH -p a100-40gb
#SBATCH --qos=standby
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
set -euo pipefail
module purge
module load cuda
module list
nvidia-smi
python -c 'import torch; print(torch.__version__, torch.version.cuda, torch.cuda.get_device_name(0))'
srun python scripts/benchmark_parallel_page_attention.py \
  --implementations serial page-state parallel \
  --output "$RCAC_SCRATCH/turboquant/$SLURM_JOB_ID/page-fused.json"
```

Create `logs/` and the scratch directory before submission. Inspect jobs with
`squeue`, `scontrol show job JOBID`, and `sacct -j JOBID`. Record job ID,
partition, QOS, GPU, driver, CUDA runtime, commit, and dirty state in every
artifact. Start with one GPU; request NVLink only for a deliberate multi-GPU
A100 experiment.

### Modules, Python, and storage

Use Lmod in job scripts (`module purge`, `module spider`, `module load`). The
current CUDA catalog lists 12.1.1, 12.6.0, and 13.1.0, but it can change. Do
not copy the local `cu128` assumption or an old CUDA 11.8 example without
checking the driver and catalog. Use the lock only if it installs cleanly;
otherwise record replacement versions, `pip freeze`, Python, Torch, Triton,
and `nvidia-smi`.

```bash
module purge
module load conda
conda-env-mod create -n turboquant-forge
module load use.own
module load conda-env/turboquant-forge-py312  # use the generated name
unset PYTHONPATH  # only when diagnosing contamination
mkdir -p "$RCAC_SCRATCH/turboquant/$SLURM_JOB_ID"
export HF_HOME="$RCAC_SCRATCH/huggingface"
export TRANSFORMERS_CACHE="$HF_HOME/transformers"
```

Keep source/manifests in Home/Depot/Fortress. Scratch is fast, not backed up,
and unchanged files can be purged after 60 days. Copy final JSON, logs, hashes,
and environment snapshots out of scratch. Never put tokens or secrets in jobs,
prompts, or artifacts.

## Ordered first run

1. Checkout `codex/turboquant-forge`; verify `git status`, `git rev-parse HEAD`,
   and that the protected progress file is untouched.
2. Read [ISSUES.md](../ISSUES.md), [BENCHMARK_PROTOCOL.md](BENCHMARK_PROTOCOL.md),
   and the three local result pages.
3. Submit a tiny preflight that prints identity, imports Torch, checks CUDA and
   Triton, and runs one synthetic page row.
4. Run `python -m pytest tests -m "not gpu" -q` and
   `python -m pytest tests/test_documentation_claims.py -q`.
5. Run GPU mechanics and synthetic fused-page validation; validate each JSON
   with `python scripts/validate_artifact.py PATH` before copying it.
6. Re-run Qwen2.5 4K/8K only with the exact pinned snapshot and a compatible
   environment. Never silently substitute a model or tokenizer.
7. Only after parity, run 16K/32K with at least three repeats across `baseline`,
   `packed_reference`, `serial_direct`, and `parallel_direct`. Keep quality
   gates unchanged. A 32K capacity failure is a valid failed result.
8. Copy raw artifacts/logs to durable storage, record hashes, and update docs
   only from validator output. Consider 64K or LongBench/RULER only after 32K
   is capacity-admissible and quality-qualified.

For telemetry after the RCAC delay, use
`/apps/external/gpu_util/get_gpu_util --job_id JOBID --save`; retain its CSV
beside the artifact and treat it as utilization evidence, not a billing claim.

## Prompt for the Gilbreth agent

```text
You are working on Purdue Gilbreth, not my local RTX 5070 Ti. Read
docs/GILBRETH_HANDOFF.md, docs/BENCHMARK_PROTOCOL.md, docs/ISSUES.md, and the
local result pages first. Run myquota, slist, sfeatures, module spider cuda,
git status, and git rev-parse HEAD before changing anything. Never run heavy
work on the login node: use Slurm with explicit -A, -p, --qos,
--gpus-per-node, --mem, and --time. Preserve .superpowers/sdd/progress.md and
confirm deletes/overwrites/cancellations. Keep diagnostic and qualified
evidence separate; do not claim 32K, a speedup, or a VRAM win without a fresh
validator-approved artifact. Record GPU/driver/CUDA/Python/PyTorch/Triton,
job ID, source/config/model hashes, raw logs, and artifact SHA-256. Start with
preflight, then 4K/8K parity, then 16K/32K. Retain failed gates instead of
changing them.
```

If controlling the cluster from local Codex, RCAC documents an SSH `rcac-mcp`
bridge and hosted docs MCP. Do not register the SSH bridge on the cluster when
the agent is already running there; use direct shell and `/etc/agents.d/` agent
context. Verify every agent script, module, and artifact.

## Official references checked 2026-08-02

- [Gilbreth overview](https://docs.rcac.purdue.edu/userguides/gilbreth/overview/)
- [Queues/QOS](https://docs.rcac.purdue.edu/userguides/gilbreth/run_jobs/queues/)
- [Submit scripts](https://docs.rcac.purdue.edu/userguides/gilbreth/run_jobs/submit_script/)
- [GPU jobs](https://docs.rcac.purdue.edu/userguides/gilbreth/run_jobs/gpu_jobs/)
- [Interactive jobs](https://docs.rcac.purdue.edu/userguides/gilbreth/run_jobs/interactive_jobs/)
- [Lmod/software](https://docs.rcac.purdue.edu/userguides/gilbreth/software/)
- [CUDA catalog](https://docs.rcac.purdue.edu/software/apps_md/cuda/)
- [Scratch storage](https://docs.rcac.purdue.edu/userguides/gilbreth/storage/scratch_space/)
- [GPU telemetry](https://docs.rcac.purdue.edu/userguides/gilbreth/run_jobs/gpu_usage_monitoring/)
- [Gilbreth AI agents](https://docs.rcac.purdue.edu/userguides/gilbreth/using_ai_agents/)
- [Agent best practices](https://docs.rcac.purdue.edu/agentic-ai/best_practices/)
- [RCAC MCP servers](https://docs.rcac.purdue.edu/agentic-ai/mcp_servers/)

Recheck these pages and account-specific Slurm output at the start of each
session; partitions, QOS limits, modules, and agent interfaces can change.
