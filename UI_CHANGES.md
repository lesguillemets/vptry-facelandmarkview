# UI Changes: Alignment Method Selection

## Overview
This document describes the UI changes made to support multiple alignment methods in the Face Landmark Viewer application.

## New UI Component

### Alignment Method Dropdown

**Location:** Top control panel, positioned after the "Limit to Static Points" checkbox

**Label:** "Alignment Method:"

**Options:**
- `default` - Custom Procrustes alignment using Kabsch algorithm (rigid transformation)
- `scipy procrustes` - SciPy's implementation including scaling

## Visual Layout

The control panel now has the following layout from left to right:

```
[Load .npy File] | Base Frame: [0▼] | [☐ Show Vectors] | [☐ Align Faces] | [☐ Limit to Static Points] | Alignment Method: [default▼]
```

## Behavior

1. **Default State:**
   - When the application starts, "default" is selected
   - The dropdown is always visible and active

2. **Selection Change:**
   - User clicks the dropdown to see available methods
   - Selecting a method immediately updates the alignment across all views
   - The change is applied to:
     - Main 3D OpenGL view
     - X-Z projection (top view)
     - Y-Z projection (side view)
     - Distance histogram

3. **Interaction with Other Controls:**
   - Works in conjunction with "Align Faces" checkbox
   - When "Align Faces" is unchecked, alignment method has no effect
   - When "Align Faces" is checked, the selected method is used
   - The "Limit to Static Points" option applies to whichever method is selected

## Implementation Details

### Widget Type
- `QComboBox` (PySide6)
- Populated dynamically from the alignment registry
- Easy to extend with additional methods in the future

### Integration Points
The alignment method selection updates the following widgets:
1. `LandmarkGLWidget` (main 3D view)
2. `ProjectionWidget` (X-Z and Y-Z projections)
3. `HistogramWidget` (distance histogram)

Each widget receives updates through the `set_alignment_method(method: str)` method.

## User Benefits

1. **Easy Experimentation:** Users can quickly try different alignment methods to see which works best for their data
2. **Real-time Updates:** No need to reload data or restart the application
3. **Consistent Interface:** All views update simultaneously to maintain consistency
4. **Educational:** Users can compare methods side-by-side by toggling between them

## Technical Notes

- The dropdown is populated automatically from the alignment registry
- Adding new alignment methods only requires:
  1. Creating a new alignment function in `src/vptry_facelandmarkview/alignments/`
  2. Registering it in `alignments/__init__.py`
  3. No UI code changes needed!

## Accessibility

- The dropdown has a clear label ("Alignment Method:")
- Method names are descriptive and user-friendly
- The control follows standard Qt/PySide6 patterns for keyboard navigation

## Future Enhancements

Possible future improvements:
- Add tooltips explaining each method's characteristics
- Show method-specific parameters in a secondary dialog
- Add a "Compare Methods" view showing side-by-side results
- Save the selected method in user preferences
