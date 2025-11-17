# Bug Fix: Scipy Procrustes Alignment Position Issue

## Issue
When selecting "scipy procrustes" alignment method, the face landmarks appeared "far, far away" in the visualization, effectively disappearing from view.

## Root Cause
The `scipy.spatial.procrustes` function standardizes **both** input matrices by:
1. Centering them at the origin (subtracting the mean)
2. Scaling them to unit Frobenius norm

The original implementation used the output `mtx2` from `scipy.spatial.procrustes(base, current)` directly as the aligned result. However, `mtx2` is centered at the origin, not at the base landmarks' original position.

### Example:
```python
import numpy as np
from scipy.spatial import procrustes

base = np.array([[100, 200, 300], [101, 200, 300], ...])  # Center at (100.25, 200.25, 300.25)
current = np.array([...])  # Different position

mtx1, mtx2, disparity = procrustes(base, current)

print(mtx1.mean(axis=0))  # ~[0, 0, 0]  <- Centered at origin!
print(mtx2.mean(axis=0))  # ~[0, 0, 0]  <- Centered at origin!
```

When landmarks with center at (100, 200, 300) were aligned and returned at (0, 0, 0), they appeared "far away" from the expected position in the viewer.

## Solution
Instead of using scipy's standardized output directly, we now:

1. **Manually compute the Procrustes transformation** following scipy's algorithm:
   - Center both point sets
   - Normalize to unit norm
   - Compute optimal rotation using SVD (Kabsch algorithm)
   - Compute optimal scale factor

2. **Apply transformation while preserving base position**:
   ```python
   # Center the landmarks
   base_centered = base - base_center
   landmarks_centered = landmarks - landmarks_center
   
   # Compute rotation and scale
   R = compute_rotation(...)
   scale = compute_scale(...)
   
   # Apply transformation and translate back to base position
   aligned = scale * (landmarks_centered @ R.T) * (base_norm / landmarks_norm) + base_center
   ```

This ensures the aligned landmarks are positioned at the base landmarks' location, not at the origin.

## Verification

### Before Fix:
- Aligned landmarks center: ~(0, 0, 0) - at origin
- Face appears far away or invisible in viewer

### After Fix:
- Aligned landmarks center: matches base center exactly
- Face stays in correct position in viewer
- Both "default" and "scipy procrustes" methods work correctly

### Test Results:
```
Base landmarks center: [100.25, 200.25, 300.25]
Scipy procrustes aligned center: [100.25, 200.25, 300.25]  ✓
Distance from base center: 0.000000  ✓
Alignment error: 0.000000  ✓
```

## Changes Made

**File**: `src/vptry_facelandmarkview/alignments/scipy_procrustes.py`

- Removed direct use of `scipy.spatial.procrustes` output
- Implemented manual Procrustes transformation computation
- Preserved base landmarks' position and scale in output
- Updated documentation to explain the approach
- Removed unused import of `scipy.spatial.procrustes`

## Impact

- **UI**: Face landmarks now stay at correct position when using scipy procrustes
- **Alignment quality**: Perfect alignment (0.000000 error) for both methods
- **Tests**: All tests pass, including new verification tests
- **Performance**: No significant change (manual computation is fast)

## Related

- Commit: 62243c3
- Issue comment: https://github.com/lesguillemets/vptry-facelandmarkview/pull/.../3540229205
