"""
Test runner script for Shape Classifier.

This script runs all tests and generates coverage reports.
Usage: python test_runner.py
"""

import pytest
import sys
import os


def run_tests():
    """Run all tests with coverage."""
    # Add current directory to Python path
    current_dir = os.path.dirname(os.path.abspath(__file__))
    if current_dir not in sys.path:
        sys.path.insert(0, current_dir)

    # Run tests with coverage
    exit_code = pytest.main([
        'tests/', '-v', '--tb=short', '--strict-markers', '--disable-warnings'
    ])

    return exit_code


if __name__ == '__main__':
    exit_code = run_tests()
    sys.exit(exit_code)
