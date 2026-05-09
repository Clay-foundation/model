"""Test that package exports work correctly."""

import pytest

import claymodel
from claymodel import (
    ClayMAEModule,
    __version__,
    clay_mae_base,
    clay_mae_large,
    clay_mae_small,
    clay_mae_tiny,
    load_metadata,
)
from claymodel.inference import DeterministicInference, ELLEProbe, PatchAnalyzer


def test_version_is_string():
    assert isinstance(__version__, str)
    assert len(__version__) > 0


def test_import_clay_mae_module():
    assert ClayMAEModule is not None


def test_import_clay_data_module():
    pytest.importorskip("sklearn", reason="requires [train] extra")
    from training.datamodule import ClayDataModule

    assert ClayDataModule is not None


def test_import_factory_functions():
    assert callable(clay_mae_tiny)
    assert callable(clay_mae_small)
    assert callable(clay_mae_base)
    assert callable(clay_mae_large)


def test_import_load_metadata():
    assert callable(load_metadata)


def test_all_exports_listed():
    for name in claymodel.__all__:
        assert hasattr(claymodel, name), f"{name} listed in __all__ but not accessible"


def test_import_unknown_raises_attribute_error():
    with pytest.raises(AttributeError):
        _ = claymodel.nonexistent_thing_xyz  # ty: ignore[unresolved-attribute]


def test_inference_package_exports():
    assert callable(DeterministicInference)
    assert ELLEProbe is not None
    assert PatchAnalyzer is not None
