# Contributing to Clay Foundation Model

Thank you for your interest in contributing to Clay! This guide covers how to set up a development environment, run tests, and submit changes.

## Development Setup

### Prerequisites

- Python 3.11 or later
- Git
- (Optional) CUDA-capable GPU for model training/inference

### Installation

```bash
# Clone the repository
git clone https://github.com/Clay-foundation/model.git
cd model

# Install in development mode with all extras
uv pip install -e ".[dev]"
```

### Verify Installation

```bash
# Run the test suite
uv run pytest tests/ -v

# Check linting
uv run ruff check claymodel/ tests/

# Check formatting
uv run ruff format --check claymodel/ tests/
```

## Project Structure

```
claymodel/
    __init__.py          # Package exports (lazy imports)
    api.py               # High-level API: embed(), load_model(), normalize()
    cli.py               # `clay` commands: embed, info, benchmark
    model.py             # Core model: Encoder, Decoder, ClayMAE, factory functions
    module.py            # Lightning module: ClayMAEModule
    utils.py             # Utilities: position embeddings, weight loading
    datamodule.py        # Training data loading
    configs/
        metadata.yaml    # Bundled sensor metadata (wavelengths, normalization stats)
    inference/
        deterministic.py # DeterministicInference context manager
        elle.py          # ELLE quality scoring probe
        masking.py       # PatchAnalyzer for chip quality filtering
    finetune/            # Downstream task examples
tests/                   # Test suite
docs/                    # Documentation (Jupyter Book)
configs/                 # Training configs
```

## Running Tests

```bash
# Run all tests
pytest tests/ -v

# Run a specific test file
pytest tests/test_model.py -v

# Run with coverage
pytest tests/ --cov=claymodel --cov-report=term-missing
```

Tests use a tiny encoder (dim=192, random weights) so they run in seconds without a GPU or checkpoint.

## Code Style

We use [ruff](https://docs.astral.sh/ruff/) for linting and formatting.

```bash
# Check for lint errors
uv run ruff check claymodel/ tests/

# Auto-fix fixable errors
uv run ruff check claymodel/ tests/ --fix

# Check formatting
uv run ruff format --check claymodel/ tests/

# Auto-format
uv run ruff format claymodel/ tests/
```

Configuration is in `ruff.toml`. Key rules:
- Max line length: 88 (ruff default)
- Max function arguments: 6 (with exceptions for model constructors)
- Import sorting enforced (isort-compatible)

## Making Changes

### Before You Start

1. Check [existing issues](https://github.com/Clay-foundation/model/issues) for related work
2. For significant changes, open an issue first to discuss the approach

### Workflow

1. Create a branch from `main`
2. Make your changes
3. Run `ruff check` and `ruff format`
4. Run `pytest tests/`
5. Submit a pull request

### Key Principles

- **No changes to model computation**: Clay v1.5 must produce identical embeddings. Any refactoring must be verified with before/after numerical comparison.
- **Test new functionality**: Add tests for new features. Tests should be fast (<30s total).
- **Follow existing patterns**: Look at how similar features are implemented before adding new ones.
- **Sensor metadata in metadata.yaml**: When adding new sensor support, add entries to both `configs/metadata.yaml` and `claymodel/configs/metadata.yaml`.

## Adding New Sensors

To add support for a new satellite sensor:

1. Compute normalization statistics (mean/std per band) from a representative sample
2. Find the central wavelength of each band in micrometers
3. Add an entry to `configs/metadata.yaml` and `claymodel/configs/metadata.yaml`
4. Test with `clay info --sensor your-sensor` and `normalize(pixels, "your-sensor")`
5. Submit a PR with the metadata and a brief description of the instrument

## Questions?

- Open an [issue](https://github.com/Clay-foundation/model/issues)
- Start a [discussion](https://github.com/Clay-foundation/model/discussions)
- Email: hello@madewithclay.org
