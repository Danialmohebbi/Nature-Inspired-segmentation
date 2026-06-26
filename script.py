"""
Standalone diagnostic script: loads one real convergence value from
the combined results CSV and tests parsing it step by step, printing
exactly what works and what fails at each stage.

Run from the project root:
    python test_convergence_parse.py
"""

import re
import ast
import pandas as pd

DATA_PATH = "results/all_results_combined.csv"


def clean_numpy_wrappers(s):
    """
    Strip np.float64(...) / np.float32(...) / np.int64(...) wrappers
    down to the bare number inside the parentheses.
    """
    return re.sub(r'np\.(float|int)\d*\(([^)]+)\)', r'\2', s)


def main():
    print(f"Loading {DATA_PATH} ...")
    df = pd.read_csv(DATA_PATH)
    print(f"Loaded {len(df)} rows total.\n")

    if "convergence" not in df.columns:
        print("ERROR: no 'convergence' column found in this file.")
        print(f"Available columns: {list(df.columns)}")
        return

    non_null = df["convergence"].dropna()
    print(f"Non-null convergence values: {len(non_null)}\n")

    if len(non_null) == 0:
        print("ERROR: every value in 'convergence' is null/NaN.")
        return

    sample = non_null.iloc[0]

    print("===== STEP 1: raw value =====")
    print(f"type: {type(sample)}")
    print(f"repr (first 300 chars): {repr(str(sample))[:300]}")
    print(f"repr (last 300 chars):  {repr(str(sample))[-300:]}")
    print()

    s = str(sample)

    print("===== STEP 2: attempt raw ast.literal_eval (expected to fail if np.float64 present) =====")
    try:
        parsed = ast.literal_eval(s)
        print(f"SUCCESS (unexpected): type={type(parsed)}, length={len(parsed)}")
    except Exception as e:
        print(f"FAILED as expected: {type(e).__name__}: {e}")
    print()

    print("===== STEP 3: clean np.float64(...) wrappers =====")
    cleaned = clean_numpy_wrappers(s)
    print(f"cleaned (first 300 chars): {cleaned[:300]}")
    print(f"cleaned (last 300 chars):  {cleaned[-300:]}")
    print()

    print("===== STEP 4: attempt ast.literal_eval on cleaned string =====")
    try:
        parsed = ast.literal_eval(cleaned)
        print(f"SUCCESS: type={type(parsed)}, length={len(parsed)}")
        print(f"first 5 values: {parsed[:5]}")
        print(f"last 5 values:  {parsed[-5:]}")
    except Exception as e:
        print(f"FAILED: {type(e).__name__}: {e}")
        print("\nThis is the actual blocking error -- the cleaning step")
        print("did not fully fix the string. Investigate the cleaned")
        print("string above for any remaining non-literal syntax.")
        return
    print()

    print("===== STEP 5: convert to plain floats =====")
    try:
        floats = [float(v) for v in parsed]
        print(f"SUCCESS: {len(floats)} plain floats, e.g. {floats[:3]}")
    except Exception as e:
        print(f"FAILED: {type(e).__name__}: {e}")
        return

    print("\n===== ALL STEPS PASSED -- parsing works correctly on this sample. =====")
    print("If your main script still fails, the issue is likely that the")
    print("fix wasn't saved/applied in plot_summary.py itself, not the logic.")


if __name__ == "__main__":
    main()