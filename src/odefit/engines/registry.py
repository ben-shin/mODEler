from __future__ import annotations

from odefit.engines.base import BackendEngineBundle
from odefit.engines.reference import (
    ReferenceNumpyProjectionEngine,
    ReferenceScipyLeastSquaresEngine,
    ReferenceScipySolverEngine,
)


def create_reference_engine_bundle() -> BackendEngineBundle:
    return BackendEngineBundle(
        name="reference",
        solver=ReferenceScipySolverEngine(),
        projection=ReferenceNumpyProjectionEngine(),
        least_squares=ReferenceScipyLeastSquaresEngine(),
    )


_ENGINE_FACTORIES = {
    "reference": create_reference_engine_bundle,
    "numpy_scipy": create_reference_engine_bundle,
}


def available_engine_names() -> list[str]:
    return sorted(_ENGINE_FACTORIES)


def get_engine_bundle(name: str = "reference") -> BackendEngineBundle:
    key = name.lower()

    if key not in _ENGINE_FACTORIES:
        raise ValueError(
            f"Unknown engine bundle: {name}. "
            f"Available engines: {', '.join(available_engine_names())}"
        )

    return _ENGINE_FACTORIES[key]()


def describe_available_engines() -> list[dict]:
    descriptions = []

    for name in available_engine_names():
        bundle = get_engine_bundle(name)

        descriptions.append(
            {
                "name": name,
                "capabilities": {
                    key: capability.__dict__
                    for key, capability in bundle.capabilities().items()
                },
            }
        )

    return descriptions
