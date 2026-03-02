# Contributing to llama.cpp for IBM POWER8

Thank you for your interest in contributing to llama.cpp for POWER8! This document provides guidelines and instructions for contributing.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [How Can I Contribute?](#how-can-i-contribute)
- [Development Setup](#development-setup)
- [Building](#building)
- [Pull Request Process](#pull-request-process)
- [Style Guidelines](#style-guidelines)

## Code of Conduct

This project follows the [Contributor Covenant](https://www.contributor-covenant.org/) code of conduct. By participating, this project, you agree to uphold this code. Please report unacceptable behavior to the repository maintainers.

## How Can I Contribute?

### Reporting Bugs

Before creating bug reports, please check the issue list. When creating a bug report, include:

- **Clear title and description**
- **Steps to reproduce**
- **Expected vs actual behavior**
- **Environment details** (OS, POWER8 model, memory)
- **Build configuration** (cmake flags, compiler version)

### Suggesting Enhancements

- Check if the enhancement has been suggested
- Describe the feature and its benefits
- Explain how it fits with POWER8 optimization goals

### Pull Requests

- Fork the repository
- Create a feature branch
- Make your changes
- Add tests if applicable
- Submit a pull request

## Development Setup

### Prerequisites

- IBM POWER8 system or simulator
- GCC 11+ or Clang 14+
- CMake 3.14+
- Git

### Setup Steps

1. Fork and clone:
   ```bash
   git clone https://github.com/YOUR_USERNAME/llama-cpp-power8.git
   cd llama-cpp-power8
   ```

2. Create build directory:
   ```bash
   mkdir build && cd build
   ```

3. Configure with cmake:
   ```bash
   cmake .. -DCMAKE_BUILD_TYPE=Release
   ```

4. Build:
   ```bash
   cmake --build . --config Release -j 8
   ```

## Building

### POWER8 Optimizations

This port includes POWER8-specific optimizations:

- **POWER9 intrinsics** via `power8-compat.h`
- **DCBT prefetch hints** via `ggml-dcbt-resident.h`
- **AltiVec/VSX** vectorization support

### Build Flags

```bash
# Enable POWER8 optimizations (default: ON)
cmake .. -DGGML_POWER8=ON

# Enable AltiVec/VSX (default: ON)
cmake .. -DGGML_ALTIVEC=ON

# Enable native instructions
cmake .. -DGGML_NATIVE=ON
```

## Pull Request Process

1. **Branch naming**: Use descriptive names like `feature/add-power9-intrinsics` or `fix/dcbt-hint`
2. **Commit messages**: Follow conventional commits format
3. **Testing**: Include performance benchmarks for changes
4. **Documentation**: Update README as needed
5. **Review**: Address all review comments

## Style Guidelines

### C Code Style

- Follow existing formatting in the file
- Use 4 spaces for indentation
- Keep lines under 120 characters
- Add comments for complex POWER8-specific code

### Commit Messages

- Use the present tense ("Add feature" not "Added feature")
- Use the imperative mood ("Optimize..." not "Optimizes...")
- Reference issues and pull requests

### Performance Testing

When contributing performance optimizations:

1. Include benchmark results
2. Compare against baseline
3. Test on real POWER8 hardware when possible

## License

By contributing, you agree that your contributions will be licensed under the MIT License.

---

🤝 Generated for the POWER8 community
