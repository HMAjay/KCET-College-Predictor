import argparse
import joblib

from config import MODEL_PATH


def main():
    parser = argparse.ArgumentParser(description="Inspect the saved KCET model bundle.")
    parser.add_argument("--rows", type=int, default=10, help="Number of trend rows to show.")
    args = parser.parse_args()

    bundle = joblib.load(MODEL_PATH)
    metadata = bundle.get("metadata", {})
    trend_df = bundle.get("trend_df")

    print("Model bundle keys:")
    print(sorted(bundle.keys()))
    print("\nMetadata:")
    for key, value in metadata.items():
        print(f"  {key}: {value}")

    if trend_df is None:
        print("\nNo trend dataframe found in the saved bundle.")
        return

    print(f"\nTrend table shape: {trend_df.shape}")
    print("Trend table columns:")
    print(list(trend_df.columns))

    rows = max(1, args.rows)
    print(f"\nFirst {rows} projected rows:")
    print(trend_df.head(rows).to_string(index=False))


if __name__ == "__main__":
    main()
