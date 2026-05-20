from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.smoke_test_backend_workflows import build_smoke_commands


def test_smoke_command_registry_fast_workflows():
    commands = build_smoke_commands(
        python_executable="python",
        include_slow=False,
    )

    names = {command.name for command in commands}

    assert "single_species_variable_projection_fit" in names
    assert "single_species_variable_projection_multistart" in names
    assert "single_species_variable_projection_model_comparison" in names
    assert "single_species_variable_projection_multistart_model_comparison" in names
    assert "multispecies_variable_projection_fit" in names
    assert "multispecies_variable_projection_multistart" in names
    assert "multispecies_variable_projection_model_comparison" in names
    assert "multispecies_variable_projection_multistart_model_comparison" in names


def test_smoke_command_registry_slow_workflows():
    commands = build_smoke_commands(
        python_executable="python",
        include_slow=True,
    )

    names = {command.name for command in commands}

    assert "single_species_variable_projection_bootstrap" in names
    assert "single_species_variable_projection_profile_likelihood" in names
    assert "multispecies_variable_projection_bootstrap" in names
    assert "multispecies_variable_projection_profile_likelihood" in names


def test_smoke_commands_use_cli_module():
    commands = build_smoke_commands(
        python_executable="python",
        include_slow=True,
    )

    for command in commands:
        assert command.command[0] == "python"
        assert command.command[1] == "-m"
        assert command.command[2] == "odefit.cli"
