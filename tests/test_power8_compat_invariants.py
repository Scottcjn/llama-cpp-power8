"""
Unit test asserting POWER8/POWER9 compatibility and preprocessor invariants.
Issue Reference: #39
"""

import re
import unittest
from pathlib import Path


class TestPower8CompatInvariants(unittest.TestCase):
    def setUp(self):
        self.root = Path(__file__).parent.parent
        self.readme = self.root / "README.md"
        self.compat_header = self.root / "powerpc" / "power8-compat.h"

    def test_compat_header_guards_power9_conflicts(self):
        """power8-compat.h must guard against __POWER9_VECTOR__ collisions."""
        self.assertTrue(self.compat_header.exists(), "power8-compat.h must exist")
        content = self.compat_header.read_text(encoding="utf-8")
        self.assertIn("__POWER8_VECTOR__", content)
        self.assertIn("!defined(__POWER9_VECTOR__)", content)
        self.assertIn("GGML_POWER8_COMPAT_ACTIVE", content)

    def test_readme_documents_compatibility_matrix(self):
        """README.md must contain the POWER8/POWER9 compatibility section and matrix."""
        content = self.readme.read_text(encoding="utf-8")
        self.assertIn("## POWER8/POWER9 Compatibility & Fallback Matrix", content)
        self.assertIn("POWER8 Native", content)
        self.assertIn("POWER9 (Compatibility)", content)
        self.assertIn("Scalar Fallback", content)


if __name__ == "__main__":
    unittest.main()
