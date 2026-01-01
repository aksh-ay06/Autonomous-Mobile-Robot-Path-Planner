"""
Minimal sanity test to verify test discovery and execution.
Useful for debugging test runner issues.
"""

import unittest

print("[DEBUG] test_minimal.py loaded")


class TestMinimal(unittest.TestCase):
    """Minimal test case for verifying unittest execution."""

    def test_basic(self):
        """Basic arithmetic sanity check."""
        print("[DEBUG] Running TestMinimal.test_basic")
        self.assertEqual(1 + 1, 2)


if __name__ == "__main__":
    print("[DEBUG] Entering __main__ block")
    unittest.main(verbosity=2)
