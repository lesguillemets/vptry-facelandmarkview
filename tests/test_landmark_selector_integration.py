"""
Integration tests for landmark selector with the main viewer.

Note: These tests require a GUI environment with Qt. They are designed to be run manually
or in an environment with Qt support. In headless CI environments, they will be skipped.
"""

import pytest
from vptry_facelandmarkview.constants import DEFAULT_ALIGNMENT_LANDMARKS


# Note: The following tests require Qt GUI and can't run in headless environment.
# They are kept here for documentation but should be run manually with a display.


def test_displaystate_has_alignment_landmarks():
    """Test that DisplayState has alignment_landmarks field"""
    from vptry_facelandmarkview.constants import DisplayState
    
    state = DisplayState()
    assert hasattr(state, 'alignment_landmarks')
    assert state.alignment_landmarks is None
    
    # Test setting it
    state.alignment_landmarks = [1, 2, 3, 4, 5]
    assert state.alignment_landmarks == [1, 2, 3, 4, 5]
