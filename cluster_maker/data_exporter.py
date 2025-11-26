###
## cluster_maker
## Data export utilities
## James Foadi - University of Bath
## November 2025
###

from __future__ import annotations

import os
import pandas as pd


def export_summary(summary_df: pd.DataFrame, csv_path: str, txt_path: str) -> None:
    """
    Export a numeric summary DataFrame to a CSV file and to a human-readable
    text summary file.

    Parameters
    ----------
    summary_df : pandas.DataFrame
        Summary DataFrame returned by `summarise_numeric`, with numeric column names
        as the index and columns ['mean', 'std', 'min', 'max', 'n_missing'].
    csv_path : str
        Path where the CSV file will be written.
    txt_path : str
        Path where the text summary will be written.

    Raises
    ------
    TypeError
        If summary_df is not a pandas DataFrame.
    ValueError
        If summary_df is None.
    FileNotFoundError
        If the directory for either output path does not exist.
    """
    # Validate summary_df
    if summary_df is None:
        raise ValueError("summary_df must not be None.")
    if not isinstance(summary_df, pd.DataFrame):
        raise TypeError("summary_df must be a pandas DataFrame.")

    # Check directories exist; do NOT silently create them
    csv_dir = os.path.dirname(csv_path) or "."
    txt_dir = os.path.dirname(txt_path) or "."

    for directory in (csv_dir, txt_dir):
        if not os.path.isdir(directory):
            raise FileNotFoundError(f"Output directory does not exist: {directory}")

    # Write CSV output
    summary_df.to_csv(csv_path, index=True)

    # Build human-readable summary lines
    lines = []
    for col, row in summary_df.iterrows():
        line = (
            f"Column '{col}': "
            f"mean={row['mean']:.4g}, "
            f"std={row['std']:.4g}, "
            f"min={row['min']:.4g}, "
            f"max={row['max']:.4g}, "
            f"n_missing={int(row['n_missing'])}"
        )
        lines.append(line)

    # Write text output
    with open(txt_path, "w", encoding="utf-8") as f:
        for line in lines:
            f.write(line + "\n")
