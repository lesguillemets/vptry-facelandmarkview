# Coding Guidelines for AI Agents

This document provides instructions for AI coding agents working on the vptry-facelandmarkview codebase.

## Code Formatting

**ALWAYS format code using Ruff before committing.**

```bash
# Format all Python files
ruff format .

# Format specific files
ruff format path/to/file.py
```

- All Python code must be formatted with Ruff
- Run `ruff format .` after making any code changes
- Verify formatting is applied before committing

## Code Linting

**ALWAYS lint and fix code using Ruff before committing.**

```bash
# Lint all Python files
ruff check .

# Lint and auto-fix issues
ruff check --fix .

# Lint specific files
ruff check path/to/file.py
```

- All Python code must pass Ruff linting
- Run `ruff check --fix .` to automatically fix issues
- Address any remaining lint warnings/errors manually
- Verify all lint issues are resolved before committing

## Type Annotations

**All Python code must be well type-annotated.**

### Required Type Annotations

1. **Function signatures** - Always annotate:
   - All function parameters
   - Function return types (use `-> None` for functions without return value)
   
   ```python
   def process_landmarks(
       landmarks: npt.NDArray[np.float64],
       scale: float,
       center: npt.NDArray[np.float64]
   ) -> npt.NDArray[np.float64]:
       ...
   ```

2. **Class attributes** - Annotate instance variables:
   ```python
   class MyClass:
       data: npt.NDArray[np.float64]
       count: int
       name: str
   ```

3. **Module-level variables** - Annotate when type is not obvious:
   ```python
   DEFAULT_SCALE: float = 1.0
   LANDMARKS: list[int] = [0, 1, 2, 3]
   ```

### Type Annotation Best Practices

- Use `typing` module types: `Optional`, `Union`, `Callable`, etc.
- Use `numpy.typing` for NumPy arrays: `npt.NDArray[np.float64]`
- Use modern type syntax where supported (Python 3.12+):
  - `list[int]` instead of `List[int]`
  - `dict[str, int]` instead of `Dict[str, int]`
  - `set[int]` instead of `Set[int]`
  - `tuple[int, str]` instead of `Tuple[int, str]`
- Use `| None` syntax instead of `Optional` for Python 3.10+
- Use specific collection types over generic ones

### Type Checking

While not strictly required for every change, consider running:
```bash
pyright
```

This project uses Pyright for static type checking. Address type errors when practical.

## Path Handling

**Prefer `pathlib.Path` over string paths.**

### Do This ✓

```python
from pathlib import Path

def load_file(filepath: Path) -> None:
    """Load data from file."""
    if filepath.exists():
        data = filepath.read_text()
    parent_dir = filepath.parent
    filename = filepath.name
```

```python
# When receiving paths from user input or arguments
import argparse
parser.add_argument("file", type=Path, help="Path to file")
```

```python
# Building paths
config_dir = Path("~/.config/myapp").expanduser()
data_file = config_dir / "data.json"
```

### Don't Do This ✗

```python
import os

def load_file(filepath: str) -> None:
    """Load data from file."""
    if os.path.exists(filepath):
        with open(filepath) as f:
            data = f.read()
    parent_dir = os.path.dirname(filepath)
    filename = os.path.basename(filepath)
```

### Path Type Annotations

- Function parameters accepting paths: use `Path`
- When paths can be strings or Path objects: use `Path | str` or `Union[Path, str]`
- Convert string inputs to Path early: `path = Path(path_input)`

### Exceptions

It's acceptable to use strings for paths when:
- Interfacing with libraries that require strings (convert at the boundary)
- Working with environment variables or configuration
- Using in subprocess calls (convert Path to str with `str(path)`)

## Testing

Before committing code changes:

1. **Format the code:**
   ```bash
   ruff format .
   ```

2. **Lint and fix issues:**
   ```bash
   ruff check --fix .
   ```

3. **Run existing tests** (if applicable):
   ```bash
   pytest
   # or appropriate test command for the project
   ```

4. **Verify type annotations:**
   ```bash
   pyright
   ```

## Summary Checklist

Before committing any Python code changes:

- [ ] Code formatted with `ruff format .`
- [ ] Code linted with `ruff check --fix .`
- [ ] All functions have type annotations
- [ ] Using `pathlib.Path` instead of string paths where appropriate
- [ ] Existing tests pass (if applicable)
- [ ] No new lint warnings or errors

## Configuration

The project includes Ruff and Pyright in the development dependencies:

```toml
[dependency-groups]
dev = [
    "pyright>=1.1.407",
    "ruff>=0.14.4",
]
```

Install development dependencies with:
```bash
pip install -e ".[dev]"
# or with uv:
uv pip install -e ".[dev]"
```

## Additional Notes

- This project targets Python 3.12+ (see `requires-python` in pyproject.toml)
- The codebase follows a modular src-layout structure
- Maintain consistency with existing code style and patterns
- Keep changes minimal and focused
- Document complex logic with clear comments when necessary
