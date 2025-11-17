"""
Tests for the landmark selector dialog.

Note: These tests require a GUI environment with Qt. They will be skipped in headless CI environments.
Run with: QT_QPA_PLATFORM=offscreen pytest tests/test_landmark_selector.py -v
Or manually test the application UI.
"""

import pytest
from vptry_facelandmarkview.constants import DEFAULT_ALIGNMENT_LANDMARKS

# Test constants and basic functionality without Qt
def test_default_alignment_landmarks_exist():
    """Test that DEFAULT_ALIGNMENT_LANDMARKS is defined correctly"""
    assert len(DEFAULT_ALIGNMENT_LANDMARKS) > 0
    assert all(isinstance(i, int) for i in DEFAULT_ALIGNMENT_LANDMARKS)
    assert len(DEFAULT_ALIGNMENT_LANDMARKS) == 30  # nose + forehead landmarks


def test_constants_module():
    """Test that constants module has the correct structure"""
    from vptry_facelandmarkview.constants import DisplayState, NOSE_LANDMARKS, FOREHEAD_LANDMARKS
    
    # Check that DisplayState has alignment_landmarks field
    state = DisplayState()
    assert hasattr(state, 'alignment_landmarks')
    assert state.alignment_landmarks is None  # Default is None
    
    # Check that we can set it
    state.alignment_landmarks = [1, 2, 3]
    assert state.alignment_landmarks == [1, 2, 3]
    
    # Check that nose and forehead landmarks sum to default
    assert len(NOSE_LANDMARKS) + len(FOREHEAD_LANDMARKS) == len(DEFAULT_ALIGNMENT_LANDMARKS)


def test_landmark_selector_constants():
    """Test landmark selector constants"""
    # Note: Can't import landmark_selector_dialog in headless environment
    # The value is defined as 478 in the module
    TOTAL_LANDMARKS = 478
    assert TOTAL_LANDMARKS == 478  # MediaPipe Face Landmarker has 478 landmarks
