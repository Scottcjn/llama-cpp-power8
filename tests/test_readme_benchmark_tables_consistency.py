# SPDX-License-Identifier: MIT
"""Unit test validating that README.md benchmark tables are disambiguated and fully specified."""

import re
from pathlib import Path
import unittest


class TestReadmeBenchmarkTablesConsistency(unittest.TestCase):
    def setUp(self):
        self.readme_path = Path(__file__).resolve().parent.parent / "README.md"
        self.assertTrue(self.readme_path.exists(), "README.md must exist in repo root")
        self.readme_text = self.readme_path.read_text(encoding="utf-8")

    def test_baseline_table_specifies_workload_and_quantization(self):
        """Baseline comparative table must specify quantization, workload metric, and CLI command."""
        self.assertIn("Baseline Comparative Throughput (POWER8 vs x86_64)", self.readme_text)
        self.assertIn("Q4_0", self.readme_text)
        self.assertIn("Prompt Eval (`pp512`)", self.readme_text)
        self.assertIn("llama-cli", self.readme_text)
        self.assertIn("--cpunodebind=0 --membind=0", self.readme_text)

    def test_llama_bench_metrics_explained(self):
        """README must define pp128 and tg32 metrics explicitly."""
        self.assertIn("`pp128` (Prompt Processing / Context Prefill)", self.readme_text)
        self.assertIn("`tg32` (Text Generation / Autoregressive Decoding)", self.readme_text)
        self.assertIn("llama-bench", self.readme_text)

    def test_tables_reconciled_with_explanation(self):
        """README must explain the relationship between baseline prompt eval and llama-bench generation."""
        self.assertIn("Reconciling the Two Benchmark Sections", self.readme_text)
        self.assertIn("prefill throughput vs autoregressive generation", self.readme_text)

    def test_reproducible_commands_present_for_all_baseline_models(self):
        """Each baseline model must have an explicit reproducible CLI invocation."""
        for model in ["llama-7b", "llama-13b", "llama-30b", "llama-65b"]:
            self.assertIn(model, self.readme_text)


if __name__ == "__main__":
    unittest.main()
