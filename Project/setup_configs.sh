#!/bin/bash

# ============================================================================
# Setup Configuration Files for Testing and Quality Checks
# Run this once to create all necessary config files
# ============================================================================

echo "========================================================================"
echo "Creating Configuration Files"
echo "========================================================================"
echo ""

# Create pytest.ini
cat > pytest.ini << 'EOF'
[pytest]
testpaths = tests
python_files = test_*.py
python_classes = Test*
python_functions = test_*
addopts = 
    --verbose
    --tb=short
    --disable-warnings

[coverage:run]
source = .
omit = 
    */tests/*
    */venv/*
    */.venv/*
    */materials_env/*

[coverage:report]
precision = 2
show_missing = True
exclude_lines =
    pragma: no cover
    def __repr__
    if __name__ == .__main__.:
EOF
echo "✅ Created pytest.ini"

# Create .flake8
cat > .flake8 << 'EOF'
[flake8]
max-line-length = 100
extend-ignore = E203,W503,E501
exclude =
    .git,
    __pycache__,
    .venv,
    venv,
    materials_env,
    .pytest_cache,
    htmlcov,
    *.egg-info
show-source = True
statistics = True
count = True
EOF
echo "✅ Created .flake8"

# Create pyproject.toml
cat > pyproject.toml << 'EOF'
[tool.black]
line-length = 100
target-version = ['py39', 'py310', 'py311']
extend-exclude = '''
/(
  \.venv
  | venv
  | materials_env
  | \.pytest_cache
  | htmlcov
)/
'''

[tool.isort]
profile = "black"
line_length = 100
skip = [".venv", "venv", "materials_env", ".pytest_cache"]
EOF
echo "✅ Created pyproject.toml"

# Create .gitignore
cat > .gitignore << 'EOF'
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python

# Virtual environments
materials_env/
.venv/
venv/
ENV/

# Testing
.pytest_cache/
.coverage
htmlcov/
coverage.xml
*.cover

# IDEs
.vscode/
.idea/
*.swp
*.swo

# OS
.DS_Store
Thumbs.db

# Project specific
*.csv
*.png
*.jpg
EOF
echo "✅ Created .gitignore"

# Create tests directory structure
mkdir -p tests
touch tests/__init__.py
echo "✅ Created tests directory"

# Create basic test file if it doesn't exist
if [ ! -f "tests/test_basic.py" ]; then
    cat > tests/test_basic.py << 'EOF'
"""Basic tests to verify environment setup"""
import numpy as np
import pandas as pd
import pytest


def test_numpy_available():
    """Test that numpy is installed and working"""
    arr = np.array([1, 2, 3, 4, 5])
    assert arr.mean() == 3.0
    assert arr.sum() == 15


def test_pandas_available():
    """Test that pandas is installed and working"""
    df = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})
    assert len(df) == 3
    assert list(df.columns) == ['a', 'b']


def test_scipy_available():
    """Test that scipy is available"""
    from scipy import optimize
    assert optimize is not None


def test_matplotlib_available():
    """Test that matplotlib is available"""
    import matplotlib.pyplot as plt
    assert plt is not None


class TestNumpyOperations:
    """Test numpy mathematical operations"""
    
    def test_array_creation(self):
        arr = np.linspace(0, 1, 11)
        assert len(arr) == 11
        assert arr[0] == 0.0
        assert arr[-1] == 1.0
    
    def test_array_math(self):
        arr = np.array([1, 2, 3, 4, 5])
        assert np.sum(arr) == 15
        assert np.mean(arr) == 3.0
        assert np.std(arr) > 0


class TestPandasOperations:
    """Test pandas data operations"""
    
    def test_dataframe_creation(self):
        df = pd.DataFrame({
            'x': [1, 2, 3],
            'y': [4, 5, 6]
        })
        assert len(df) == 3
        assert 'x' in df.columns
        assert 'y' in df.columns
    
    def test_dataframe_operations(self):
        df = pd.DataFrame({'a': [1, 2, 3, 4, 5]})
        assert df['a'].mean() == 3.0
        assert df['a'].sum() == 15
EOF
    echo "✅ Created tests/test_basic.py"
fi

# Create README for testing
cat > TESTING.md << 'EOF'
# Testing Guide

## Quick Start

### Run Tests
```bash
./run.sh
# Select option 4 for tests, 5 for quality checks, or 6 for full CI/CD
```

### Individual Commands

#### Run Tests with Coverage
```bash
pytest --verbose --cov=. --cov-report=term-missing --cov-report=html
```

#### Check Code Formatting
```bash
black --check --line-length=100 *.py tests/*.py
```

#### Run Linter
```bash
flake8 *.py tests/*.py --max-line-length=100 --statistics
```

#### Check Import Sorting
```bash
isort --check-only --profile black *.py tests/*.py
```

## Configuration Files

- `pytest.ini` - Test discovery and coverage settings
- `.flake8` - Linting rules and ignore patterns
- `pyproject.toml` - Black and isort configuration

## Expected Output

### Tests (Option 4)
```
tests/test_basic.py::test_numpy_available PASSED          [ 16%]
tests/test_basic.py::test_pandas_available PASSED         [ 33%]
...
---------- coverage: platform win32, python 3.13.4 -----------
Name                  Stmts   Miss  Cover   Missing
---------------------------------------------------
tests/test_basic.py      25      0   100%
---------------------------------------------------
TOTAL                    25      0   100%
```

### Quality Checks (Option 5)
```
1️⃣  BLACK FORMATTING CHECK
==========================
✅ Black: All files properly formatted

2️⃣  FLAKE8 LINTING
==================
✅ Flake8: No issues found

3️⃣  ISORT IMPORT ORDER CHECK
============================
✅ isort: Imports properly sorted
```

### Full CI/CD (Option 6)
Runs all tests and quality checks in sequence, same as GitHub Actions would run.
EOF
echo "✅ Created TESTING.md"

echo ""
echo "========================================================================"
echo "✅ Configuration Complete!"
echo "========================================================================"
echo ""
echo "Created files:"
echo "  - pytest.ini          (Test configuration)"
echo "  - .flake8             (Linting rules)"
echo "  - pyproject.toml      (Black & isort settings)"
echo "  - .gitignore          (Git ignore patterns)"
echo "  - tests/test_basic.py (Basic test suite)"
echo "  - TESTING.md          (Testing documentation)"
echo ""
echo "Next steps:"
echo "  1. Run: ./run.sh"
echo "  2. Select option 6 (Full CI/CD pipeline)"
echo "  3. Take screenshots of the output"
echo ""