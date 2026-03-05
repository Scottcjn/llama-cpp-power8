# Contributing to llama-cpp-power8

Thank you for your interest in contributing to llama-cpp-power8!

## How to Contribute

### Reporting Bugs

1. Check if the bug has already been reported
2. Create a detailed bug report with:
   - Clear title and description
   - Steps to reproduce
   - Expected vs actual behavior
   - Environment details

### Suggesting Features

1. Open an issue with a clear feature description
2. Explain why this feature would be useful
3. Provide any relevant examples or references

### Pull Requests

1. Fork the repository
2. Create a branch, e.g. `git checkout -b docs/contributing`
3. Make your changes
4. If you touch build scripts or performance code, include before/after notes.
5. Commit with clear messages
6. Push your branch and open a PR

## Build & Toolchain Notes

This repo contains POWER8-specific headers and benchmarking code intended to be copied into a `llama.cpp` checkout.

### Requirements
- CMake 3.14+
- GCC with POWER8 support
- A POWER8 system (or cross toolchain) if you want to validate VSX/AltiVec changes

### Quick sanity check
- `altivec_benchmark.c` should compile with appropriate POWER8 flags (see README).

## Testing

There's no CI test suite here yet. Please include:
- what you changed
- how you validated it (compile command, bench output, etc.)

### Code Style

- Follow the existing code style
- Add comments for complex logic
- Keep functions focused and small

## Questions?

- Open an issue for questions
- Contact: scott@elyanlabs.ai

## Recognition

Contributors will be recognized in the project README.
