# Implementation Summary: Multiple Alignment Methods

## Issue Description
The original issue requested the ability to use different alignment methods in the Face Landmark Viewer application, specifically:
1. Restructure the code to make it easy to try different alignment functions
2. Add scipy's procrustes function as an alternative to the existing implementation
3. Add a dropdown menu in the UI to choose between methods
4. Label the methods as "default" and "scipy procrustes"

## Solution Implemented

### 1. Code Restructuring
Created a dedicated module `src/vptry_facelandmarkview/alignments/` with:
- `__init__.py`: Registry system and exports
- `default.py`: Original Kabsch/Procrustes implementation
- `scipy_procrustes.py`: New scipy-based implementation

This modular structure makes it trivial to add new alignment methods in the future.

### 2. Alignment Methods

#### Default Method (`default.py`)
- Implementation: Kabsch algorithm (Procrustes alignment)
- Transformation: Rigid (rotation + translation only)
- Use case: When you want to preserve relative scale of facial features
- Performance: Aligns perfectly when only rigid transformations are present

#### Scipy Procrustes Method (`scipy_procrustes.py`)
- Implementation: `scipy.spatial.procrustes`
- Transformation: Similarity (rotation + translation + uniform scaling)
- Use case: When comparing faces of different sizes or normalizing scale variations
- Performance: More flexible but may introduce small errors due to scaling

### 3. UI Integration

Added a dropdown menu labeled "Alignment Method:" to the top control panel with:
- Location: After "Limit to Static Points" checkbox
- Options: "default" and "scipy procrustes"
- Behavior: Real-time updates across all views when changed

### 4. Widget Updates

All visualization widgets were updated to support dynamic alignment method selection:
- `LandmarkGLWidget` (main 3D view)
- `ProjectionWidget` (X-Z and Y-Z projections)
- `HistogramWidget` (distance histogram)

Each widget:
- Added `set_alignment_method(method: str)` method
- Dynamically retrieves the alignment function from the registry
- Updates immediately when method changes

### 5. State Management

Updated `DisplayState` dataclass to include `alignment_method: str` field, maintaining consistency across all widgets.

## Files Changed

### New Files
- `src/vptry_facelandmarkview/alignments/__init__.py`
- `src/vptry_facelandmarkview/alignments/default.py`
- `src/vptry_facelandmarkview/alignments/scipy_procrustes.py`
- `tests/test_alignment_methods.py`
- `tests/test_ui_integration.py`
- `tests/demo_alignment_methods.py`
- `UI_CHANGES.md`

### Modified Files
- `src/vptry_facelandmarkview/viewer.py` (added dropdown, handler)
- `src/vptry_facelandmarkview/gl_widget.py` (dynamic alignment method selection)
- `src/vptry_facelandmarkview/projection_widget.py` (dynamic alignment method selection)
- `src/vptry_facelandmarkview/histogram_widget.py` (dynamic alignment method selection)
- `src/vptry_facelandmarkview/constants.py` (added alignment_method to DisplayState)
- `src/vptry_facelandmarkview/utils.py` (backward compatibility wrapper)
- `pyproject.toml` (added scipy dependency)
- `README.md` (documentation updates)

## Testing

### Test Coverage
1. **Backward Compatibility Tests**: All existing tests pass
   - `test_alignment.py`: Original alignment tests
   - `test_imports.py`: Package structure tests

2. **New Functionality Tests**
   - `test_alignment_methods.py`: Tests both alignment methods
   - Tests with and without alignment indices
   - Validates alignment quality metrics

3. **Code Quality**
   - All code passes `ruff` linting
   - No security vulnerabilities found by CodeQL

### Test Results
```
✓ test_alignment.py - 7/7 tests passed
✓ test_alignment_methods.py - 4/4 tests passed  
✓ test_imports.py - 5/5 tests passed
✓ ruff check - All checks passed
✓ codeql - 0 alerts found
```

## Documentation

### Updated Documentation
1. **README.md**:
   - Added "Alignment Method" to UI Controls section
   - New "Alignment Methods" subsection explaining both methods
   - Updated programmatic usage examples
   - Updated project structure diagram

2. **UI_CHANGES.md**:
   - Detailed description of UI changes
   - Layout diagrams
   - Behavior specifications
   - Future enhancement ideas

3. **Demo Script**:
   - `demo_alignment_methods.py` demonstrates the feature
   - Shows practical comparison between methods
   - Explains real-world use cases

## Extensibility

Adding a new alignment method requires only:

1. Create new file: `src/vptry_facelandmarkview/alignments/your_method.py`
```python
def align_landmarks_your_method(
    landmarks: npt.NDArray[np.float64],
    base_landmarks: npt.NDArray[np.float64],
    alignment_indices: Optional[set[int] | list[int]] = None,
) -> npt.NDArray[np.float64]:
    # Your implementation
    pass
```

2. Register in `alignments/__init__.py`:
```python
from vptry_facelandmarkview.alignments.your_method import align_landmarks_your_method

ALIGNMENT_METHODS = {
    "default": align_landmarks_default,
    "scipy procrustes": align_landmarks_scipy_procrustes,
    "your method": align_landmarks_your_method,  # Add this
}
```

That's it! The UI will automatically include the new method in the dropdown.

## Performance Considerations

- Alignment functions are retrieved from registry on-demand
- No performance impact when alignment is disabled
- Minimal overhead from dynamic function lookup (single dictionary access)
- Both methods are efficient for typical face landmark datasets

## Backward Compatibility

- `align_landmarks_to_base()` in `utils.py` maintained as wrapper
- Existing code using this function continues to work
- Default behavior unchanged (uses "default" method)
- All existing tests pass without modification

## Dependencies

Added `scipy>=1.16.0` to `pyproject.toml` dependencies.

## Security

- CodeQL analysis: 0 vulnerabilities found
- No user input directly affects alignment computation
- Registry is immutable after initialization
- Type hints and validation throughout

## Conclusion

The implementation successfully addresses all requirements from the original issue:
- ✓ Code structure is easy for trying different functions
- ✓ Scipy's procrustes function added
- ✓ Dropdown menu added to UI
- ✓ Methods labeled as requested
- ✓ Real-time updates across all views
- ✓ Extensible architecture for future methods
- ✓ Comprehensive testing and documentation
- ✓ No security vulnerabilities

The solution is production-ready and follows best practices for modularity, testing, and documentation.
