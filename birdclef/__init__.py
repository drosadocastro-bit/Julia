"""
Julia Bioacoustics — BirdCLEF+ 2026 Competition Module

Adapted from Project Aria's audio classification stack.
Target: Identify wildlife species from 5-second audio windows
        in the Pantanal wetlands (Brazil).

Kaggle constraints:
  - CPU notebook only (≤ 90 min runtime)
  - No internet at submission time
  - Output: submission.csv with per-species probabilities
"""

__version__ = "0.1.0"
