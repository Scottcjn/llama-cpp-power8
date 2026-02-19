# Contributing to llama.cpp POWER8

Thank you for your interest in contributing to POWER8 optimization for llama.cpp!

## How to Contribute

### Adding New Optimizations

1. **Fork** the repository
2. **Create** a branch: `git checkout -b optimize/your-feature`
3. **Add** your optimization to the appropriate header in `powerpc/`
4. **Test** with `altivec_benchmark.c`
5. **Submit** a PR with benchmarks

### Code Style

- Follow existing code patterns in the `powerpc/` headers
- Use VSX/AltiVec intrinsics with `-mcpu=power8 -mvsx -maltivec`
- Add comments explaining the optimization

### Adding New Hardware

This repo focuses on POWER8, but contributions for other PowerPC architectures are welcome:
- POWER9 (requires compatibility layer)
- POWER10
- Older PowerPC (G4/G5)

## Building

```bash
# Standard build
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)

# Run benchmark
./altivec_benchmark
```

## Pull Request Process

1. Include benchmark results comparing before/after
2. Update README.md if adding new optimizations
3. Ensure code compiles on Ubuntu 20.04 with GCC

## License

By contributing, you agree to license under MIT.
