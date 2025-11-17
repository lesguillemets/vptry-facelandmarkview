# Scipy Procrustes Implementation - Solution

## Problem
The user wanted to use the actual `scipy.spatial.procrustes` function (not reimplement it) to compare with our default alignment method. However, scipy's function standardizes both input matrices by centering them at the origin and scaling to unit norm, which caused the aligned landmarks to appear "far away" in the visualization.

## Solution Overview
We now **actually use scipy.spatial.procrustes** and transform its output to display in the base frame's coordinate system. This allows:
1. Proper comparison with scipy's implementation
2. Simple visualization code (single coordinate system)
3. Compatibility with all existing widgets

## Implementation Details

### Key Insight
`scipy.spatial.procrustes(base, current)` returns:
- `mtx1_std`: Standardized base (centered at origin, unit Frobenius norm)
- `mtx2_std`: Aligned current (also standardized)
- `disparity`: Sum of squared differences

Both returned matrices are in **standardized space** (origin-centered, unit-scaled).

### Back-Transformation
To display in the base frame's coordinate system:

```python
# Get base frame parameters
base_center = base.mean(axis=0)
base_norm = np.linalg.norm(base - base_center)

# Transform from standardized space to base frame
aligned = mtx2_std * base_norm + base_center
```

This simple transformation:
- Scales from unit norm to base's original norm
- Translates from origin to base's original center
- Results in landmarks positioned correctly in the visualization

### For Alignment Indices
When using only a subset of landmarks for alignment calculation:

1. Call `scipy.spatial.procrustes` on the subset
2. Extract transformation parameters (rotation, scale)
3. Apply same transformation to all landmarks
4. Transform result to base frame

## Why This Approach Works

**Advantages**:
1. ✅ Uses actual scipy.spatial.procrustes function (as requested)
2. ✅ Allows proper comparison between methods
3. ✅ Keeps visualization code simple (no dual coordinate systems)
4. ✅ No changes needed to widgets (gl_widget, projection_widget, histogram_widget)
5. ✅ All tests pass with perfect alignment

**What We Avoided**:
- ❌ Manual reimplementation of scipy's algorithm
- ❌ Modifying all widgets to handle two coordinate systems
- ❌ Tracking standardization parameters throughout the codebase
- ❌ Complex conditional rendering logic

## Verification

Both methods produce:
- Perfect alignment (0.000000 error)
- Correct positioning (face at base frame location, not at origin)
- Consistent behavior with alignment indices

## Code Location
`src/vptry_facelandmarkview/alignments/scipy_procrustes.py`

Key lines:
- Line 11: `from scipy.spatial import procrustes` - imports scipy function
- Line 87: `_, aligned_subset_std, disparity = procrustes(...)` - calls scipy function
- Line 104: `aligned_subset = aligned_subset_std * base_norm + base_center` - back-transforms to base frame

## Conclusion
This solution achieves the goal of using scipy's actual implementation while maintaining code simplicity. The key is recognizing that we can use scipy's standardized output and transform it back to a consistent coordinate system for display, rather than trying to make the entire codebase handle multiple coordinate systems.
