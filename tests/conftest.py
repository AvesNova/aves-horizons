"""
Pytest configuration and global fixtures.

This file makes all fixtures from test_fixtures.py available to all tests
without requiring individual imports.
"""

# Import all fixtures to make them available globally
from .test_fixtures import *
