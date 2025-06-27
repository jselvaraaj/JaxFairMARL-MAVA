# Tests

This directory contains tests for the Mava package.

## Structure

```
tests/
├── conftest.py              # Shared pytest configuration and fixtures
├── test_basic.py           # Basic tests to verify setup
└── unit/                   # Tests mirroring the mava package structure
    └── environments/
        └── jaxfairspread_test.py
```

## Running Tests

### Run all tests

```bash
pytest
```

### Run with coverage

```bash
pytest --cov=mava
```

### Run specific test categories

```bash
# Run only unit tests
pytest -m unit

# Run only integration tests
pytest -m integration

# Skip slow tests
pytest -m "not slow"
```

### Run specific test files

```bash
pytest tests/test_basic.py
pytest tests/unit/environments/jaxfairspread_test.py
```

## Test Conventions

- Test files should be named `test_*.py` or `*_test.py`
- Test functions should be named `test_*`
- Use pytest markers to categorize tests:
  - `@pytest.mark.unit` for unit tests
  - `@pytest.mark.integration` for integration tests
  - `@pytest.mark.slow` for slow-running tests
- Use fixtures for shared test setup
- Tests should be deterministic and reproducible

## Configuration

Pytest is configured in `pyproject.toml` with:

- Coverage reporting enabled
- Strict markers and configuration
- Verbose output
- Short traceback format
