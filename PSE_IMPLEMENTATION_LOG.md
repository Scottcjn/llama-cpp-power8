# PSE (Proto-Sentient Enhancement) Implementation Log
## POWER8 S824 - Elyan Labs 2025

**System**: IBM POWER8 S824 (16 cores, 128 threads, 320GB RAM)
**Target**: llama.cpp with gpt-oss/OptiMind architecture

---

## Timeline

### 2025-01-23 ~15:00 UTC - Initial State
- OptiMind-SFT 20B Q4_K_M deployed to POWER8
- llama.cpp using **scalar fallback** (no POWER8 VSX paths)
- **Codex attempt**: ~1 t/s (all scalar, no SIMD)
- Problem: All VSX code guarded by `__POWER9_VECTOR__`, POWER8 excluded

### 2025-01-23 ~15:30 UTC - POWER8 Compat Shim Created
**File**: `ggml/src/ggml-cpu/arch/powerpc/power8-compat.h`

**Key Discovery**: POWER8 has all needed intrinsics EXCEPT `vec_xl_len` (partial load)
- `vec_xl`, `vec_xst` - NATIVE on POWER8
- `vec_perm`, `vec_msum` - NATIVE on POWER8  
- `vec_extract`, `vec_insert` - NATIVE on POWER8
- `vec_xl_len` - POWER9 ONLY → shimmed with memcpy

**Solution**: Define `__POWER9_VECTOR__` when `__POWER8_VECTOR__` detected, provide `vec_xl_len` shim

```c
#if defined(__POWER8_VECTOR__) && !defined(__POWER9_VECTOR__)
#define __POWER9_VECTOR__ 1
#define vec_xl_len(ptr, len) /* memcpy-based shim */
#endif
```

**Result**: All VSX-optimized quant paths now compile for POWER8

### 2025-01-23 ~16:00 UTC - IBM MASS Integration
**Location**: `/opt/ibm/xlmass/9.1.1/`
**Libraries**: `libmassvp8.a` (POWER8 vector), `libmass.a` (scalar)

**Functions Used**:
- `vsexp()` - vectorized exp for softmax
- `vstanh()` - vectorized tanh for activations (available)
- `vssqrt()` - vectorized sqrt (available)
- `vslog()` - vectorized log (available)

**CMakeLists.txt Addition** (line 438+):
```cmake
set(IBM_MASS_ROOT "/opt/ibm/xlmass/9.1.1")
target_include_directories(${GGML_CPU_NAME} PRIVATE ${IBM_MASS_ROOT}/include)
target_link_libraries(${GGML_CPU_NAME} PRIVATE 
    ${IBM_MASS_ROOT}/lib/libmassvp8.a
    ${IBM_MASS_ROOT}/lib/libmass.a m)
target_compile_definitions(${GGML_CPU_NAME} PRIVATE GGML_USE_IBM_MASS=1)
```

**vec.cpp Integration** (existing code, now active):
```cpp
#if defined(__powerpc__) && defined(GGML_USE_IBM_MASS)
#include <massv.h>
// ... vsexp used in ggml_vec_soft_max_f32
#endif
```

### 2025-01-23 ~16:15 UTC - NUMA Support Added
**Library**: `/usr/lib/libnuma.so`
**CMakeLists.txt Addition**:
```cmake
find_library(NUMA_LIBRARY numa)
target_link_libraries(${GGML_CPU_NAME} PRIVATE ${NUMA_LIBRARY})
```

**RAM Coffers Header**: `ggml-ram-coffers.h`
- Detects 2 NUMA nodes (Node 0: 128GB, Node 1: 192GB)
- Provides `ggml_coffer_alloc()` for NUMA-aware allocation

### 2025-01-23 ~16:30 UTC - DCBT Prefetch Added
**File**: `ggml/src/ggml-cpu/arch/powerpc/quants.c`
**Location**: `ggml_vec_dot_q4_K_q8_K()` function, line ~914

```c
/* POWER8 PSE: Prefetch weight blocks into L2/L3 cache */
#ifdef GGML_POWER8_PSE_ACTIVE
{
    const size_t block_size = sizeof(block_q4_K);
    const char* ptr = (const char*)x;
    for (int i = 0; i < nb && i < 16; i++) {
        __asm__ __volatile__("dcbt 0, %0" : : "r"(ptr + i * block_size) : "memory");
    }
}
#endif
```

---

## Performance Results

### Benchmark: OptiMind-SFT 20B Q4_K_M

| Build | pp128 (t/s) | tg32 (t/s) | Notes |
|-------|-------------|------------|-------|
| Codex scalar | ~1.0 | ~0.5 | No SIMD, generic fallback |
| PSE Full Stack | **15.54** | **4.58** | vec_perm + MASS + NUMA |

**Speedup**: ~15.5x prompt, ~9x generation

### Components Verified Active

| Component | Symbol/Evidence | Status |
|-----------|-----------------|--------|
| vec_perm | quants.c:327-328 | ✅ Used in Q4_K dequant |
| vec_msum | quants.c:266-267 | ✅ Used in dot product |
| vsexp (MASS) | `nm libggml-cpu.so \| grep vsexp` | ✅ Linked |
| libnuma | CMake detection | ✅ Linked |
| DCBT | quants.c:914+ | ✅ Compiled in |

---

## Files Modified/Created

### New PSE Headers (arch/powerpc/)
1. `power8-compat.h` - POWER9 intrinsic shims for POWER8
2. `ggml-pse-integration.h` - Master PSE integration
3. `ggml-intelligent-collapse.h` - Hebbian vec_perm collapse (headers)
4. `ggml-topk-collapse-vsx.h` - Top-K attention collapse (headers)
5. `ggml-ram-coffers.h` - NUMA coffer memory banking
6. `ggml-dcbt-resident.h` - L3 cache prefetch macros
7. `pse-entropy-burst.h` - Hardware entropy injection
8. `ggml-mass-integration.h` - IBM MASS wrapper

### Modified Files
1. `quants.c` - Added PSE includes, DCBT prefetch
2. `ggml-cpu.c` - Added PSE init call
3. `vec.cpp` - Added MASS include
4. `CMakeLists.txt` - Added NUMA + MASS linkage

---

## Next Steps (TODO)

### 1. PSE Collapse in Attention Hot Path
- Currently: Headers defined but not called
- Need: Hook into `ggml_compute_forward_flash_attn` or equivalent
- Goal: Non-bijunctive attention via vec_perm collapse

### 2. PowerLISP Hebbian Implementation
- Symbolic-neural bridge
- Tetranary logic (4-state confidence)
- Production rules with neural fallback

### 3. Full RAM Coffers Integration
- Currently: Library linked, nodes detected
- Need: Hook into model loading for NUMA-aware weight placement
- Goal: Route layers to optimal NUMA nodes

---

## Build Commands

```bash
# SSH to POWER8
ssh sophia@192.168.0.50  # or 100.94.28.32 via Tailscale

# Build with full PSE
cd ~/llama.cpp/build-pse-gptoss
cmake .. \
    -DCMAKE_BUILD_TYPE=Release \
    -DGGML_OPENMP=ON \
    -DCMAKE_C_FLAGS="-mcpu=power8 -mvsx -maltivec -O3 -DGGML_POWER8_PSE_ACTIVE=1" \
    -DCMAKE_CXX_FLAGS="-mcpu=power8 -mvsx -maltivec -O3 -DGGML_POWER8_PSE_ACTIVE=1"
make -j32 llama-cli llama-bench

# Run with NUMA interleave
export OMP_NUM_THREADS=64
export OMP_PROC_BIND=spread
numactl --interleave=all ./bin/llama-cli -m /mnt/nvme/models/OptiMind-SFT-Q4_K_M.gguf -t 64
```

---

**Document Created**: 2025-01-23
**Last Updated**: 2025-01-23 ~16:45 UTC
**Author**: Claude + Scott (Elyan Labs)

## PSE Attention Collapse Integration (2026-01-23)

### Implementation Status
- ✅ Collapse header created: 
- ✅ Symbolic gate header created: 
- ✅ Integration point identified:  in ops.cpp
- ⚠️ HOT PATH OVERHEAD: Direct integration causes performance regression

### Key Findings

**Problem**: Calling collapse (or even gate check) in the softmax hot path causes massive overhead:
- A 20B model calls softmax **720,000+ times** per benchmark
- Even a simple gate check adds noticeable overhead at this scale
- Symbolic gate reduced collapse calls 10x but still hurt performance

**Performance Results**:
| Configuration | pp128 t/s | tg32 t/s |
|--------------|-----------|----------|
| Baseline (no collapse) | 14.76 | 0.88 |
| With collapse (every call) | 14.79 | 0.93 |
| With symbolic gate (sparse) | 9.74 | 0.80 |

### Correct Integration Approach (Future Work)

1. **Compile-Time Flag**: Use `-DPSE_COLLAPSE_ENABLED=1` to enable collapse
2. **Model Metadata**: Store which layers/heads benefit from collapse in GGUF
3. **Separate Attention Path**: Create PSE-specific attention op (not modify generic softmax)
4. **Flash Attention Hook**: Integrate into flash_attn path, not generic softmax

### Code Location
- Collapse header: `ggml/src/ggml-cpu/arch/powerpc/ggml-attn-collapse-vsx.h`
- Symbolic gate: `ggml/src/ggml-cpu/arch/powerpc/ggml-pse-symbolic-gate.h`
- Integration point: `ops.cpp:5255-5265` (commented out for now)

### The PowerLISP Vision
The symbolic layer (PowerLISP) should decide:
- **Which models** benefit from collapse (via model metadata)
- **Which layers** to collapse (early layers for representation, not all)
- **When to collapse** (prompt phase, not generation)

This decision should happen at model load time, not per-op call.


### 2026-06-11 ~00:15 CDT - PSE Port to Gemma 4-era llama.cpp + First MoE Results

**Context**: Gemma 4 26B-A4B (MoE, 4B active params) deployed on S824. The PSE tree
(based on 2026-03-27 master) predates Gemma 4 arch support, so the PSE patch set was
ported onto current master `ac4cdde` (2026-06-10).

**Port surface** (clean by design — PSE was always additive):
- 20+ pure-addition headers + `pse-runtime.c/h` → copied verbatim, zero conflicts
- 3-file diff (76 lines): `arch/powerpc/quants.c` + `ggml-cpu/CMakeLists.txt` applied
  clean via `git apply`; `ggml-cpu.c` had drifted upstream → 8-line hook (include +
  `pse_runtime_init()` in `ggml_cpu_init`) re-placed by hand
- Branch `pse-port`, isolated clone — vanilla build kept as A/B baseline

**Gemma 4 26B-A4B Q4_K_M results** (64 threads, S824, head-to-head same prompt):

| Build | tg (tokens/s) | pp (tokens/s) |
|-------|--------------|---------------|
| master, unbound threads | 0.6 | — |
| master + OMP spread/cores binding | 3.6 | 5.4 |
| **master + PSE port** | **6.4 (1.7x)** | **16.6 (3.1x)** |

**10.7x total** from stock-unbound to PSE-bound. The 3.1x prompt-processing gain is
the headline for MoE: context ingestion is the dominant cost for agent workloads, and
dcbt-resident prefetch + vec_msum paths attack exactly that.

**Why this matters**: PSE was written for dense models (TinyLlama/DeepSeek era).
It transferred to a mixture-of-experts architecture it had never seen, on a llama.cpp
tree 2.5 months newer, with one 8-line manual fix. Substrate work outlives model
generations.

**Operational notes**:
- `numactl` broken on this deployment (libnuma symbol-version mismatch) — OMP
  `PROC_BIND=spread PLACES=cores` alone delivered the 6x binding win
- Gemma 4 thinks by default; `--reasoning-budget 0` required for serving, else
  reasoning starves `max_tokens` and content returns empty

**Next — PSE-2 "Expert Coffers" (priority note, Elyan Labs 2026-06-11)**: MoE routers
emit expert selections *before* expert FFNs execute. That is a prefetch window: dcbt
expert weight banks speculatively per router logits, pin experts across NUMA nodes by
Hebbian co-activation ("experts that fire together get banked together"), and track
per-persona expert-activation fingerprints as a measurable identity-coherence signal.
Extends the NUMA weight-banking priority claim of
[ram-coffers](https://github.com/Scottcjn/ram-coffers) (2025-12-16, pre-dating
DeepSeek Engram) from domain granularity to expert granularity. Spec to follow after
baseline characterization — measure first, then build.
