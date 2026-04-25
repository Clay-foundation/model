"""Test that package exports work correctly (A1 verification)."""


def test_version_is_string():
    from claymodel import __version__

    assert isinstance(__version__, str)
    assert len(__version__) > 0


def test_import_clay_mae_module():
    from claymodel import ClayMAEModule

    assert ClayMAEModule is not None


def test_import_clay_data_module():
    from claymodel import ClayDataModule

    assert ClayDataModule is not None


def test_import_factory_functions():
    from claymodel import clay_mae_base, clay_mae_large, clay_mae_small, clay_mae_tiny

    assert callable(clay_mae_tiny)
    assert callable(clay_mae_small)
    assert callable(clay_mae_base)
    assert callable(clay_mae_large)


def test_import_load_metadata():
    from claymodel import load_metadata

    assert callable(load_metadata)


def test_all_exports_listed():
    import claymodel

    for name in claymodel.__all__:
        assert hasattr(claymodel, name), f"{name} listed in __all__ but not accessible"


def test_import_unknown_raises_attribute_error():
    import claymodel

    try:
        _ = claymodel.nonexistent_thing_xyz
        assert False, "Should have raised AttributeError"
    except AttributeError:
        pass


def test_inference_package_exports():
    from claymodel.inference import DeterministicInference, ELLEProbe, PatchAnalyzer

    assert callable(DeterministicInference)
    assert ELLEProbe is not None
    assert PatchAnalyzer is not None
