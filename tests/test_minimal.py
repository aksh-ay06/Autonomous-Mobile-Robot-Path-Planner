import unittest

print("[DEBUG] test_minimal.py loaded")

class TestMinimal(unittest.TestCase):
    def test_basic(self):
        print("[DEBUG] Running test_basic")
        self.assertEqual(1 + 1, 2)

if __name__ == '__main__':
    print("[DEBUG] Entering main block")
    unittest.main()
