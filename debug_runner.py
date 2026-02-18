import unittest
import sys
import os

# Redirect stderr to a file for better debugging
with open('debug_sdk_log.txt', 'w') as f:
    sys.stderr = f
    import test_sdk
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromModule(test_sdk)
    runner = unittest.TextTestRunner(stream=f, verbosity=2)
    result = runner.run(suite)
    
    if result.wasSuccessful():
        print("Tests passed!")
    else:
        print(f"Tests failed with {len(result.errors)} errors and {len(result.failures)} failures.")
