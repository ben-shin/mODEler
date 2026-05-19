from __future__ import annotations

import importlib.util
from dataclasses import dataclass

import pandas as pd


@dataclass
class BackendCapability:
    """
    Availability and purpose of an optional performance backend.
    """

    name: str
    import_name: str
    available: bool
    purpose: str
    recommendation: str


def is_module_available(
    import_name: str,
) -> bool:
    """
    Return True if a module can be imported.
    """

    return importlib.util.find_spec(import_name) is not None


def detect_backend_capabilities() -> list[BackendCapability]:
    """
    Detect optional performance-related backends.

    This does not activate any backend. It only reports availability.
    """

    capabilities = [
        BackendCapability(
            name="SciPy CPU",
            import_name="scipy",
            available=is_module_available("scipy"),
            purpose="Current default ODE solving and least-squares fitting backend.",
            recommendation="Keep as the stable default backend.",
        ),
        BackendCapability(
            name="Numba",
            import_name="numba",
            available=is_module_available("numba"),
            purpose="JIT compilation for array-based RHS functions.",
            recommendation=(
                "Best first acceleration experiment after benchmarking. "
                "Requires array-based model representation."
            ),
        ),
        BackendCapability(
            name="Cython",
            import_name="Cython",
            available=is_module_available("Cython"),
            purpose="Compiled Python extension modules for hot loops.",
            recommendation=(
                "Consider only if profiling shows Python RHS overhead remains "
                "large after simpler optimizations."
            ),
        ),
        BackendCapability(
            name="JAX",
            import_name="jax",
            available=is_module_available("jax"),
            purpose="JIT, autodiff, batched simulation, possible GPU support.",
            recommendation=(
                "Consider for batched workloads, many parameter sets, bootstrap, "
                "or GPU experiments. Not a drop-in SciPy solve_ivp replacement."
            ),
        ),
        BackendCapability(
            name="CuPy",
            import_name="cupy",
            available=is_module_available("cupy"),
            purpose="GPU-backed NumPy-like arrays.",
            recommendation=(
                "Useful only after the solver/fitting code is redesigned around "
                "GPU array operations."
            ),
        ),
        BackendCapability(
            name="Julia bridge",
            import_name="juliacall",
            available=is_module_available("juliacall"),
            purpose="Calling Julia/DifferentialEquations.jl from Python.",
            recommendation=(
                "Potentially strong for stiff/large ODE systems, but adds "
                "distribution and environment complexity."
            ),
        ),
    ]

    return capabilities


def build_backend_capabilities_table(
    capabilities: list[BackendCapability] | None = None,
) -> pd.DataFrame:
    """
    Build a dataframe summarizing backend capabilities.
    """

    if capabilities is None:
        capabilities = detect_backend_capabilities()

    return pd.DataFrame(
        [
            {
                "backend": capability.name,
                "import_name": capability.import_name,
                "available": capability.available,
                "purpose": capability.purpose,
                "recommendation": capability.recommendation,
            }
            for capability in capabilities
        ]
    )


def summarize_backend_strategy() -> list[str]:
    """
    Return the recommended backend acceleration roadmap.
    """

    return [
        "1. Benchmark real workflows first.",
        "2. Optimize the current SciPy CPU path.",
        "3. Add array-based RHS representation.",
        "4. Try Numba before Cython.",
        "5. Consider Julia for stiff or large ODE systems.",
        "6. Consider JAX/GPU only for batched workloads.",
    ]
