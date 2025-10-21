#!/usr/bin/env python3
"""
Unit tests for find_start function using pytest.

# Run all parameterized tests
pytest extract_bowling_test.py -v

# Run specific test case
pytest extract_bowling_test.py::TestFindStart::test_find_start[exact_minimum_run] -v

# Run with more verbose output
pytest extract_bowling_test.py -vv
"""

import pytest
from typing import List
from extract_action import runs_2sec, window4
import numpy as np

class TestWindow4:
    """Parameterized tests for window4 iterator (4-wide sliding window with padding)."""

    @pytest.mark.parametrize("name,seq,fill,expected", [
        ("empty", [], None, []),
        ("one_elem_none_fill", [1], None, [(1, None, None, None)]),
        ("two_elems_none_fill", [1, 2], None, [(1, 2, None, None), (2, None, None, None)]),
        ("three_elems_none_fill", [1, 2, 3], None, [(1, 2, 3, None), (2, 3, None, None), (3, None, None, None)]),
        ("exact_four_none_fill", [1, 2, 3, 4], None, [(1, 2, 3, 4), (2, 3, 4, None), (3, 4, None, None), (4, None, None, None)]),
        ("five_none_fill", [1, 2, 3, 4, 5], None, [(1, 2, 3, 4), (2, 3, 4, 5), (3, 4, 5, None), (4, 5, None, None), (5, None, None, None)]),
        ("strings_with_zero_fill", ["a", "b", "c"], 0, [("a", "b", "c", 0), ("b", "c", 0, 0), ("c", 0, 0, 0)]),
    ])
    def test_window4(self, name: str, seq: List, fill, expected: List[tuple]):
        got = list(window4(iter(seq), fillvalue=fill))
        assert got == expected, f"Test '{name}' failed: expected {expected}, got {got}"


class TestRuns2Sec:
    """Parameterized tests for runs_2sec function."""

    @pytest.mark.parametrize("name,labelled_frames,fps,want", [
        # Basic functionality tests
        ("no_action_frames", [(False, "f0"), (False, "f1"), (False, "f2")], 60, []),
        ("insufficient_action_run", [(False, "f0"), (True, "f1"), (True, "f2"), (False, "f3")], 60, []),
        ("exact_4_action_frames", [(True, "f0"), (True, "f1"), (True, "f2"), (True, "f3")], 60, [None, "f0", "f1", "f2", "f3"]),
        ("more_than_4_action_frames", [(True, "f0"), (True, "f1"), (True, "f2"), (True, "f3"), (True, "f4")], 60, [None, "f0", "f1", "f2", "f3", "f4"]),
        
        # Different FPS values
        ("fps_2_seconds", [(True, "f0"), (True, "f1"), (True, "f2"), (True, "f3")], 2.0, [None, "f0", "f1", "f2", "f3"]),
        ("fps_0_5_seconds", [(True, "f0"), (True, "f1"), (True, "f2"), (True, "f3")], 0.5, [None, "f0"]),
        
        # Multiple runs
        ("two_separate_runs", [
           (True, "f0"), (True, "f1"), (True, "f2"), (True, "f3"),  # First run
            (False, "f4"), (False, "f5"),  # Gap
            (True, "f6"), (True, "f7"), (True, "f8"), (True, "f9")   # Second run
            ], 1, [None, "f0", "f1", None, "f6", "f7"]),
        
        # Edge cases
        ("empty_input", [], 60, []),
        ("single_frame", [(True, "f0")], 60, []),
        ("two_frames", [(True, "f0"), (True, "f1")], 60, []),
        ("three_frames", [(True, "f0"), (True, "f1"), (True, "f2")], 60, []),
        
        # Mixed patterns
        ("action_then_non_action", [(True, "f0"), (True, "f1"), (True, "f2"), (True, "f3"), (False, "f4")], 60, [None, "f0", "f1", "f2", "f3", "f4"]),
        ("non_action_then_action", [(False, "f0"), (True, "f1"), (True, "f2"), (True, "f3"), (True, "f4")], 60, [None, "f1", "f2", "f3", "f4"]),
        
        # Complex patterns
        ("long_sequence", [
            (False, "f0"), (False, "f1"), (False, "f2"),  # Non-action start
            (True, "f3"), (True, "f4"), (True, "f5"), (True, "f6"),  # Action run
            (False, "f7"), (False, "f8"),  # Non-action gap
            (True, "f9"), (True, "f10"), (True, "f11"), (True, "f12")  # Second action run
        ], 2, [None, "f3", "f4", "f5", "f6", None, "f9", "f10", "f11", "f12"]),
        ])
    def test_runs_2sec(self, name: str, labelled_frames: List[tuple], fps: float, want: List[List]):
        """Test runs_2sec function with various inputs."""
        got =  list(runs_2sec(iter(labelled_frames), fps))
        assert got == want, f"Test '{name}' failed: input fps {fps} expected {want}, got {got}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
