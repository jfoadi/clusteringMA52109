from __future__ import annotations
import sys, os, pandas as pd
from cluster_maker import summarize_numeric, export_summary

OUTPUT_DIR = "demo_output"

def main(argv):
    if len(argv) != 2:
        print("ERROR: Incorrect number of arguments.")
        print("Usage: python demo/analyse_from_csv.py path/to/input.csv")
        sys.exit(0)

    input_path = argv[1]
    if not os.path.exists(input_path):
        print(f"ERROR: File '{input_path}' not found.")
        sys.exit(0)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print(f"Reading CSV: {input_path}")
    df = pd.read_csv(input_path)

    print("Computing numeric summary...")
    summary = summarize_numeric(df)

    csv_out = os.path.join(OUTPUT_DIR, "numeric_summary.csv")
    txt_out = os.path.join(OUTPUT_DIR, "numeric_summary.txt")

    print(f"Saving summary to:\n  - {csv_out}\n  - {txt_out}")
    export_summary(summary, csv_out, txt_out)

    print("Done.")

if __name__ == "__main__":
    main(sys.argv)