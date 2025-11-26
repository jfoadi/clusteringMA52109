from __future__ import annotations

import sys
import os
import pandas as pd

from cluster_maker.data_analyser import summarise_numeric_columns
from cluster_maker.data_exporter import export_summary_reports

OUTPUT_DIR = "demo_output"

def main(argv: list[str] | None = None) -> None:
    if argv is None:
        argv = sys.argv

    # require exactly 2 args (script and input path)
    if len(argv) != 2:
        print("ERROR: Incorrect number of arguments provided.")
        print("Usage: python demo/analyse_from_csv.py path/to/input.csv")
        # exit cleanly without traceback
        sys.exit(1)

    input_path = argv[1]
    if not os.path.exists(input_path):
        print(f"ERROR: Input file not found: {input_path}")
        print("Usage: python demo/analyse_from_csv.py path/to/input.csv")
        sys.exit(1)

    print(f"Reading CSV: {input_path}")
    df = pd.read_csv(input_path)
    print(f"Loaded {len(df)} rows; computing numeric summary...")

    summary = summarise_numeric_columns(df)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    csv_out = os.path.join(OUTPUT_DIR, "summary.csv")
    txt_out = os.path.join(OUTPUT_DIR, "summary.txt")

    print(f"Saving CSV summary to: {csv_out}")
    print(f"Saving human-readable summary to: {txt_out}")
    export_summary_reports(summary, csv_out, txt_out, include_index=True)
    print("Analysis complete.")

if __name__ == "__main__":
    main()