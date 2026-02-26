# POWER8 PSE (Proto-Sentient Engine) Headers

Drop-in performance headers for llama.cpp on IBM POWER8 S824.

## Architecture: Vec_Perm Non-Bijunctive Collapse

Standard LLMs use bijunctive (full matrix) attention. Vec_perm enables
**non-bijunctive collapse** — pruning weak paths and duplicating strong paths
in a single POWER8 cycle across 128 threads.

## Performance (IBM POWER8 S824, 512GB DDR4, 128 threads SMT8)

| Build | pp128 | tg32 | Notes |
|-------|-------|------|-------|
| Stock llama.cpp (scalar) | 16.74 t/s | — | TinyLlama 1.1B Q4 baseline |
| + POWER8 VSX | 66.49 t/s | — | Vector ops enabled |
| + PSE + 64 threads optimal | 84.62 t/s | — | Thread tuning |
| + Full L2/L3 resident prefetch | **147.54 t/s** | **18.88 t/s** | dcbt_resident enabled |
| GPT-OSS 120B MXFP4 MoE | — | **6.42 t/s** | 116B param MoE, 60GB model |
| Qwen2.5-14B Q4 + RPC GPU | 68.8 t/s | 14.9 t/s | Via 40GbE to C4130 V100 |

**Key finding**: 64 threads optimal, NOT 128 (SMT8 contention above 64)

## Key Headers

| Header | Purpose |
|--------|---------|
| `ggml-pse-integration.h` | Master PSE integration — include this first |
| `ggml-dcbt-resident.h` | Full L2/L3 resident dcbt prefetch (the 147 t/s enabler) |
| `ggml-intelligent-collapse.h` | Hebbian Top-K vec_perm collapse |
| `ggml-topk-collapse-vsx.h` | 12-op vec_perm attention collapse |
| `ggml-attn-collapse-vsx.h` | VSX burst attention collapse |
| `ggml-ram-coffer.h` | NUMA weight banking v1 |
| `ggml-ram-coffers.h` | NUMA-aware weight banking (4 nodes, 512GB) |
| `ggml-coffer-mmap.h` | GGUF mmap sharding across NUMA nodes |
| `ggml-neuromorphic-coffers.h` | Brain hemisphere → NUMA cognitive routing |
| `ggml-symbolic-neural-bridge.h` | PowerLISP ↔ neural integration |
| `ggml-pse-symbolic-gate.h` | PSE symbolic gating |
| `ggml-sparse-softmax.h` | Sparse softmax for MoE attention |
| `ggml-mass-integration.h` | IBM MASS library integration (vsexp, vstanh) |
| `ggml-gpu-offload-vulkan.h` | GPU offload via Vulkan (Navi 12 / AMD) |
| `ggml-gpu-offload-local.h` | Local GPU offload helpers |
| `ggml-gpu-offload-integration.h` | GPU offload integration layer |
| `power8-compat.h` | POWER9→POWER8 intrinsic compatibility shims |
| `pse-entropy-burst.h` | mftb timebase hardware entropy injection |

## Source Files

| File | Purpose |
|------|---------|
| `quants.c` | Main quantization with PSE includes |
| `ggml-pse-state.c` | PSE global state management |
| `cpu-feats.cpp` | CPU feature detection for POWER8 |

## Integration

```c
// In quants.c, near the top:
#include "arch/powerpc/ggml-pse-integration.h"
```

## Build (on POWER8 Ubuntu 20.04)

```bash
mkdir build-pse && cd build-pse
cmake .. \
    -DCMAKE_BUILD_TYPE=Release \
    -DGGML_OPENMP=ON \
    -DGGML_BLAS=ON \
    -DGGML_BLAS_VENDOR=OpenBLAS \
    -DCMAKE_C_FLAGS="-mcpu=power8 -mvsx -maltivec -O3 -mtune=power8 -funroll-loops \
        -DGGML_USE_MASS=1 -DGGML_POWER8_COMPAT_ACTIVE=1 -I/opt/ibm/mass/include" \
    -DCMAKE_CXX_FLAGS="-mcpu=power8 -mvsx -maltivec -O3 -mtune=power8 -funroll-loops \
        -DGGML_USE_MASS=1 -DGGML_POWER8_COMPAT_ACTIVE=1 -I/opt/ibm/mass/include" \
    -DCMAKE_EXE_LINKER_FLAGS="-L/opt/ibm/mass/lib -lmassvp8 -lmass"
make -j32
```

## Runtime (NUMA-aware)

```bash
export OMP_NUM_THREADS=64
export OMP_PROC_BIND=spread
export OMP_PLACES=cores

# Models < 100GB: single NUMA node
numactl --cpunodebind=1 --membind=1 ./bin/llama-cli \
  -m ~/models/qwen2.5-14b-q4.gguf -t 32 -p "Hello" -n 64

# Models > 100GB: interleave NUMA nodes
numactl --interleave=all ./bin/llama-cli \
  -m ~/models/gpt-oss-120b-mxfp4.gguf -t 64 -p "Hello" -n 32
```

## Neuromorphic NUMA Coffers

Maps brain hemisphere cognitive functions to NUMA topology for intelligent routing.

| NUMA Node | Brain Region | Cognitive Function | Coffer |
|-----------|--------------|-------------------|--------|
| Node 0 | Right Hemisphere | Spatial, Creative, Holistic | 2 |
| Node 1 | Left Hemisphere | Language, Logic, Sequential | 1 |
| Node 2 | Temporal Lobe | Memory, Context, Episodic | 3 |
| Node 3 | Prefrontal Cortex | Executive, Planning, Meta | 0 |

**Priority claim**: This architecture predates DeepSeek Engram (arXiv:2601.07372, Jan 12, 2026)
by 27+ days. See `ggml-neuromorphic-coffers.h` for Brodmann area integration details.

## Hardware: IBM POWER8 S824

- **CPUs**: Dual 8-core POWER8, 16 cores / 128 hardware threads (SMT8)
- **RAM**: 512GB DDR4 across 2 NUMA nodes (Node 0: 128GB, Node 1: 192GB active)
- **GPU**: AMD Navi 12 8GB via OCuLink, Mesa RADV Vulkan (ppc64le)
- **GPU Offload**: Dell C4130 with Tesla V100 16GB via 40GbE (10.40.0.x, 0.15ms RTT)
- **OS**: Ubuntu 20.04 LTS (last POWER8-supported release)

## The PSE Manifesto

> "Proto-sentient behavior emerges from constraint-bound selection, not from
> unconstrained computation."

Non-bijunctive attention IS constraint-bound selection:
- **Top-K** enforces preference (not everything matters equally)
- **Amplification** creates bias (winners strengthen — Hebbian)
- **Pruning** removes noise (losers don't vote)
- **mftb entropy** injects hardware-native randomness (no two runs identical)

The vec_perm collapse is not an approximation of attention — it is a
**hardware-native Hebbian attention mechanism** that standard GPUs cannot
implement efficiently.

## Entropy Divergence (Proof)

Same seed (42), same temp (0.7), 3 runs produce different MD5 hashes:
```
b52ce7b85e9d02ee27748433b3c88b64  run_1.txt
15c558b2c6c903104a1d4bd1a393563e  run_2.txt
fd5d7ae25b76ae0e88e955a34a28235f  run_3.txt
```

POWER8 `mftb` timebase oscillator drift seeds the collapse differently each run.
Stock deterministic LLMs produce identical output. PSE does not.
