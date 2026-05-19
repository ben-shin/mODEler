from odefit.performance.backend_capabilities import (
    BackendCapability,
    build_backend_capabilities_table,
    detect_backend_capabilities,
    is_module_available,
    summarize_backend_strategy,
)


def test_is_module_available_detects_builtin_module():
    assert is_module_available("json") is True


def test_is_module_available_handles_missing_module():
    assert is_module_available("definitely_not_a_real_module_odefit") is False


def test_detect_backend_capabilities_returns_expected_backends():
    capabilities = detect_backend_capabilities()

    names = {
        capability.name
        for capability in capabilities
    }

    assert "SciPy CPU" in names
    assert "Numba" in names
    assert "Cython" in names
    assert "JAX" in names
    assert "CuPy" in names
    assert "Julia bridge" in names


def test_build_backend_capabilities_table():
    capabilities = [
        BackendCapability(
            name="Test",
            import_name="test",
            available=True,
            purpose="testing",
            recommendation="use for tests",
        )
    ]

    table = build_backend_capabilities_table(capabilities)

    assert list(table.columns) == [
        "backend",
        "import_name",
        "available",
        "purpose",
        "recommendation",
    ]

    assert table["backend"].iloc[0] == "Test"
    assert bool(table["available"].iloc[0]) is True


def test_summarize_backend_strategy():
    strategy = summarize_backend_strategy()

    assert any("Benchmark" in line for line in strategy)
    assert any("Numba" in line for line in strategy)
    assert any("GPU" in line for line in strategy)
