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
