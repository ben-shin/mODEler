from __future__ import annotations

import re
from pathlib import Path


TARGET = Path("src/odefit/fitting/variable_projection.py")


def main() -> None:
    text = TARGET.read_text()

    original = text

    text, init_count = re.subn(
        pattern=(
            r'(?m)^(\s*)predicted_dataframe = pd\.DataFrame'
            r'\(\{"time": timepoints\}\)\n'
            r'\1residuals_dataframe = pd\.DataFrame'
            r'\(\{"time": timepoints\}\)'
        ),
        repl=(
            r'\1predicted_columns = {"time": np.asarray(timepoints, dtype=float)}\n'
            r'\1residual_columns = {"time": np.asarray(timepoints, dtype=float)}'
        ),
        string=text,
    )

    text, assign_count = re.subn(
        pattern=(
            r'(?m)^(\s*)predicted_dataframe\[signal_column\] = predicted\n'
            r'\1residuals_dataframe\[signal_column\] = residual'
        ),
        repl=(
            r'\1predicted_columns[signal_column] = predicted\n'
            r'\1residual_columns[signal_column] = residual'
        ),
        string=text,
    )

    text, return_count = re.subn(
        pattern=(
            r'(?m)^(\s*)predicted_dataframe=predicted_dataframe,\n'
            r'\1residuals_dataframe=residuals_dataframe,'
        ),
        repl=(
            r'\1predicted_dataframe=pd.DataFrame(predicted_columns, copy=False),\n'
            r'\1residuals_dataframe=pd.DataFrame(residual_columns, copy=False),'
        ),
        string=text,
    )

    print("Variable projection DataFrame assembly patch")
    print("===========================================")
    print(f"DataFrame initializers replaced: {init_count}")
    print(f"Column insert pairs replaced: {assign_count}")
    print(f"Return blocks replaced: {return_count}")

    expected_init = 2
    expected_assign = 2
    expected_return = 2

    if init_count != expected_init:
        raise SystemExit(
            f"Expected {expected_init} initializer replacements, got {init_count}."
        )

    if assign_count != expected_assign:
        raise SystemExit(
            f"Expected {expected_assign} assignment replacements, got {assign_count}."
        )

    if return_count != expected_return:
        raise SystemExit(
            f"Expected {expected_return} return replacements, got {return_count}."
        )

    if text == original:
        raise SystemExit("No changes made.")

    TARGET.write_text(text)
    print(f"Patched {TARGET}")


if __name__ == "__main__":
    main()
