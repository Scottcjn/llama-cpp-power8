# llama.cpp for IBM POWER8

**AltiVec/VSX Optimized LLM Inference for POWER8**

This provides POWER8-specific optimizations for [llama.cpp](https://github.com/ggerganov/llama.cpp), enabling efficient LLM inference on IBM POWER8 servers.

## What's Included

- **power8-compat.h** - POWER9 intrinsics compatibility layer for POWER8
- **ggml-dcbt-resident.h** - Full L2/L3 cache-resident prefetch hints
- **altivec_benchmark.c** - AltiVec/VSX performance benchmark

## Performance

Tested on IBM Power System S824 (dual 8-core POWER8, 576GB RAM):

| Model | pp128 (tokens/s) | tg32 (tokens/s) |
|-------|-----------------|-----------------|
| TinyLlama 1.1B Q4 | ~85 | ~15 |
| Llama-7B Q4 | ~20 | ~5 |
| DeepSeek-33B Q4 | ~5 | ~1 |

## Building llama.cpp for POWER8

### Prerequisites

- Ubuntu 20.04 LTS (last POWER8-supported release)
- GCC with POWER8 support
- CMake 3.14+

### Build Commands

```bash
# Clone llama.cpp
git clone https://github.com/ggerganov/llama.cpp
cd llama.cpp

# Copy POWER8 headers
cp /path/to/powerpc/* ggml/src/ggml-cpu/arch/powerpc/

# Configure for POWER8
mkdir build-power8 && cd build-power8
cmake .. \
    -DCMAKE_BUILD_TYPE=Release \
    -DGGML_OPENMP=ON \
    -DCMAKE_C_FLAGS="-mcpu=power8 -mvsx -maltivec -O3 -mtune=power8 -funroll-loops" \
    -DCMAKE_CXX_FLAGS="-mcpu=power8 -mvsx -maltivec -O3 -mtune=power8 -funroll-loops"

# Build
make -j$(nproc)
```

### With IBM MASS Library (Optional)

IBM Mathematical Acceleration Subsystem (MASS) provides optimized math functions:

```bash
cmake .. \
    -DCMAKE_BUILD_TYPE=Release \
    -DGGML_OPENMP=ON \
    -DCMAKE_C_FLAGS="-mcpu=power8 -mvsx -maltivec -O3 -mtune=power8 -funroll-loops -DGGML_USE_MASS=1 -I/opt/ibm/mass/include" \
    -DCMAKE_CXX_FLAGS="-mcpu=power8 -mvsx -maltivec -O3 -mtune=power8 -funroll-loops -DGGML_USE_MASS=1 -I/opt/ibm/mass/include" \
    -DCMAKE_EXE_LINKER_FLAGS="-L/opt/ibm/mass/lib -lmassvp8 -lmass"
```

## Running Inference

```bash
# Basic inference
./bin/llama-cli -m ~/models/llama-7b-q4.gguf -p "Hello world" -n 64

# With optimal thread count (64 threads is usually best on POWER8)
OMP_NUM_THREADS=64 ./bin/llama-cli -m ~/models/llama-7b-q4.gguf -p "Hello" -n 64

# NUMA-aware (for dual-socket systems)
numactl --interleave=all ./bin/llama-cli -m ~/models/large-model.gguf -p "Test" -n 32

# Benchmark
./bin/llama-bench -m ~/models/tinyllama-1.1b-q4.gguf -t 64 -p 128 -n 32
```

## POWER8 Optimization Notes

### Thread Scaling

64 threads is typically optimal on POWER8 (NOT 128):
- 16 threads: ~40 t/s
- 32 threads: ~65 t/s
- **64 threads: ~85 t/s** (optimal)
- 96 threads: ~75 t/s
- 128 threads: ~65 t/s

### Cache Prefetch

The `ggml-dcbt-resident.h` header provides cache-resident prefetch hints:
- `DCBT_RESIDENT_FULL()` - Keeps data in L2/L3 until explicit eviction
- Critical for weight reuse in attention/matmul

### Memory Alignment

POWER8 prefers 128-byte aligned data for optimal VSX performance.
The `power8-compat.h` handles alignment requirements.

## Files

```
powerpc/
├── power8-compat.h       # POWER9 → POWER8 intrinsic compatibility
└── ggml-dcbt-resident.h  # Cache-resident prefetch hints

altivec_benchmark.c       # VSX/AltiVec performance test
```

## Hardware Tested

- **System**: IBM Power System S824 (8286-42A)
- **CPUs**: Dual 8-core POWER8, 128 threads (SMT8)
- **RAM**: 576 GB DDR3
- **OS**: Ubuntu 20.04 LTS

## Related Projects

- [llama.cpp](https://github.com/ggerganov/llama.cpp) - Main project
- [Claude Code POWER8](https://github.com/Scottcjn/claude-code-power8) - Claude Code for POWER8

## Attribution

**Months of research, tuning, and testing on real POWER8 hardware went into this.**

If you use this project, please give credit:

```
llama.cpp POWER8 Optimizations by Scott (Scottcjn)
https://github.com/Scottcjn/llama-cpp-power8
```

If this helped you, please:
- ⭐ **Star this repo** - It helps others find it
- 📝 **Credit in your project** - Keep the attribution
- 🔗 **Link back** - Share the love

## Credits

- [ggerganov/llama.cpp](https://github.com/ggerganov/llama.cpp) - Original project
- [Elyan Labs](https://github.com/Scottcjn) - POWER8 optimizations

## License

MIT License - Free to use, but please keep the copyright notice and attribution.

---

*"576GB RAM. 128 threads. Your POWER8 was built for AI - it just didn't know it yet."*
