from odefit.api.project_state_bridge import (
    infer_observed_species_from_project_state,
    project_state_to_backend_config,
    project_state_to_gui_metadata,
    project_state_to_gui_project_payload,
)
from odefit.fitting.fit_settings import FitSettings
from odefit.fitting.initial_condition_spec import InitialConditionSpec
from odefit.fitting.observable_spec import ObservableSpec
from odefit.fitting.parameter_spec import ParameterSpec
from odefit.project.project_state import create_project_state


def make_project_state():
    return create_project_state(
        project_name="test project",
        project_notes="project notes",
        model_text="A -> B",
        data_path="data.csv",
        time_column="time",
        signal_columns=["A23_HN", "A24_HN"],
        normalization_method="none",
        species_mapping={
            "A23_HN": "A",
            "A24_HN": "A",
        },
        parameter_specs=[
            ParameterSpec(
                name="k1f",
                initial_guess=0.1,
                lower_bound=0.001,
                upper_bound=10.0,
            )
        ],
        initial_condition_specs=[
            InitialConditionSpec(
                species="A",
                initial_guess=1.0,
                lower_bound=0.0,
                upper_bound=2.0,
                fixed=True,
                fixed_value=1.0,
            ),
            InitialConditionSpec(
                species="B",
                initial_guess=0.0,
                lower_bound=0.0,
                upper_bound=2.0,
                fixed=True,
                fixed_value=0.0,
            ),
        ],
        observable_specs=[
            ObservableSpec(
                data_column="A23_HN",
                species="A",
                scale_initial_guess=1.0,
                scale_lower_bound=0.0,
                scale_upper_bound=10.0,
                offset_initial_guess=0.0,
                offset_lower_bound=-1.0,
                offset_upper_bound=1.0,
            ),
            ObservableSpec(
                data_column="A24_HN",
                species="A",
                scale_initial_guess=1.0,
                scale_lower_bound=0.0,
                scale_upper_bound=10.0,
                offset_initial_guess=0.0,
                offset_lower_bound=-1.0,
                offset_upper_bound=1.0,
            ),
        ],
        fit_settings=FitSettings(
            species_mapping={
                "A23_HN": "A",
                "A24_HN": "A",
            },
            method="trf",
            loss="linear",
            rtol=1e-8,
            atol=1e-10,
        ),
        output_dir="outputs",
        last_fit_output_dir="outputs/latest",
        metadata={
            "experiment": "test",
        },
    )


def test_infer_observed_species_from_project_state():
    project_state = make_project_state()

    observed_species = infer_observed_species_from_project_state(
        project_state
    )

    assert observed_species == "A"


def test_project_state_to_backend_config_single_species_vp():
    project_state = make_project_state()

    config = project_state_to_backend_config(
        project_state,
        workflow="fit",
        use_variable_projection=True,
    )

    assert config["model_text"] == "A -> B"
    assert config["data"] == "data.csv"
    assert config["time_column"] == "time"
    assert config["signal_columns"] == ["A23_HN", "A24_HN"]
    assert config["observed_species"] == "A"

    assert config["use_variable_projection"] is True
    assert config["use_multispecies_variable_projection"] is False

    assert config["parameters"]["k1f"]["initial_guess"] == 0.1
    assert config["parameters"]["k1f"]["lower_bound"] == 0.001
    assert config["parameters"]["k1f"]["upper_bound"] == 10.0

    assert config["initial_conditions"]["A"]["mode"] == "fixed"
    assert config["initial_conditions"]["A"]["value"] == 1.0

    assert config["method"] == "trf"
    assert config["loss"] == "linear"
    assert config["rtol"] == 1e-8
    assert config["atol"] == 1e-10


def test_project_state_to_backend_config_multispecies_vp():
    project_state = make_project_state()

    config = project_state_to_backend_config(
        project_state,
        workflow="fit",
        use_variable_projection=False,
        use_multispecies_variable_projection=True,
        observed_species=["A", "B"],
    )

    assert config["use_variable_projection"] is False
    assert config["use_multispecies_variable_projection"] is True
    assert config["observed_species"] == ["A", "B"]


def test_project_state_to_backend_config_bootstrap_adds_defaults():
    project_state = make_project_state()

    config = project_state_to_backend_config(
        project_state,
        workflow="bootstrap",
        use_variable_projection=True,
    )

    assert config["n_bootstrap"] == 100
    assert config["n_workers"] == 1
    assert config["confidence_level"] == 0.95


def test_project_state_to_backend_config_profile_adds_defaults():
    project_state = make_project_state()

    config = project_state_to_backend_config(
        project_state,
        workflow="profile_likelihood",
        use_variable_projection=True,
    )

    assert config["profile_parameters"] == ["k1f"]
    assert config["profile_n_points"] == 15
    assert config["profile_span_factor"] == 10.0
    assert config["profile_log_space"] is True


def test_project_state_to_gui_metadata():
    project_state = make_project_state()

    metadata = project_state_to_gui_metadata(project_state)

    assert metadata["project_name"] == "test project"
    assert metadata["project_notes"] == "project notes"
    assert metadata["normalization_method"] == "none"
    assert metadata["last_fit_output_dir"] == "outputs/latest"
    assert metadata["metadata"]["experiment"] == "test"


def test_project_state_to_gui_project_payload():
    project_state = make_project_state()

    payload = project_state_to_gui_project_payload(
        project_state,
        workflow="fit",
    )

    assert payload["workflow"] == "fit"
    assert payload["metadata"]["project_name"] == "test project"
    assert payload["config"]["model_text"] == "A -> B"
    assert payload["config"]["use_variable_projection"] is True
