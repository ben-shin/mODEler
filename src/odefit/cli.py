import sys
from .app import run_gui


def main():
    # If the user passes a file like `python -m odefit.cli my_model.fit`
    project_file = sys.argv[1] if len(sys.argv) > 1 else None

    # Launch the GUI!
    sys.exit(run_gui(project_file))


if __name__ == "__main__":
    main()