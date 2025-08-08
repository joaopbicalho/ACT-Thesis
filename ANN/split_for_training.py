import pandas as pd
import ast
from sklearn.preprocessing import StandardScaler
import pickle
import os

def load_and_split(
    csv_path,
    output_prefix="sub_data",
    test_size=0.15,
    val_size=0.15,
    use_residue=False,
    sequence_length=1
):
    """
    Loads the joined CSV (containing columns: Epoch, Params, Coeffs, Signal, Error, Residue, label),
    normalizes features, and splits into Train / Validation / Test sets, sequence-aware.

    Parameters
    ----------
    csv_path : str
        Path to the CSV file with columns [Epoch, Params, Coeffs, Signal, Error, Residue, label].
    output_prefix : str
        Prefix for output files.
    test_size : float
        Fraction of total data to reserve for the test split.
    val_size : float
        Fraction for validation.
    use_residue : bool
        Whether or not to include Residue column.
    sequence_length : int
        Length of sequence input (used to align split indices).
    """

    # --- 1) Load CSV ---
    df = pd.read_csv(csv_path)

    # --- 2) Parse stringified lists in Params, Coeffs (Residue optional) ---
    def parse_stringified_list(item):
        return ast.literal_eval(item)

    df["Params"] = df["Params"].apply(parse_stringified_list)
    df["Coeffs"] = df["Coeffs"].apply(parse_stringified_list)

    if use_residue:
        df["Residue"] = df["Residue"].apply(parse_stringified_list)

    # --- 3) Build features (X) and labels (y) ---
    features = []
    for _, row in df.iterrows():
        param_list = row["Params"]
        coeff_list = row["Coeffs"]
        if use_residue:
            residue_list = row["Residue"]
            row_features = param_list + coeff_list + residue_list
        else:
            row_features = param_list + coeff_list
        features.append(row_features)

    X = pd.DataFrame(features)
    y = df["label"]

    total_samples = len(X)

    # --- 4) Sequence-aware Split Indices ---
    def adjust_split_idx(idx, sequence_length):
        return (idx // sequence_length) * sequence_length

    test_split_idx = int(total_samples * (1 - test_size))
    val_split_idx = int(test_split_idx * (1 - val_size / (1 - test_size)))

    # Align indices
    test_split_idx = adjust_split_idx(test_split_idx, sequence_length)
    val_split_idx = adjust_split_idx(val_split_idx, sequence_length)

    print(f"Adjusted split indices:")
    print(f"  Validation split index: {val_split_idx}")
    print(f"  Test split index:       {test_split_idx}")

    # --- 5) Split ---
    X_train = X.iloc[:val_split_idx]
    y_train = y.iloc[:val_split_idx]

    X_val = X.iloc[val_split_idx:test_split_idx]
    y_val = y.iloc[val_split_idx:test_split_idx]

    X_test = X.iloc[test_split_idx:]
    y_test = y.iloc[test_split_idx:]

    # --- 6) Normalization (Train set only) ---
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    # Save scaler for reuse
    scaler_filename = f"{output_prefix}_scaler.pkl"
    with open(scaler_filename, "wb") as f:
        pickle.dump(scaler, f)

    print(f"Scaler saved to {scaler_filename}")

    # --- 7) Save splits ---
    pd.DataFrame(X_train_scaled).to_csv(f"{output_prefix}_X_train.csv", index=False)
    y_train.to_csv(f"{output_prefix}_y_train.csv", index=False)

    pd.DataFrame(X_val_scaled).to_csv(f"{output_prefix}_X_val.csv", index=False)
    y_val.to_csv(f"{output_prefix}_y_val.csv", index=False)

    pd.DataFrame(X_test_scaled).to_csv(f"{output_prefix}_X_test.csv", index=False)
    y_test.to_csv(f"{output_prefix}_y_test.csv", index=False)

    print("Splits saved:")
    print(f"  Train: {len(X_train)} examples")
    print(f"  Val:   {len(X_val)} examples")
    print(f"  Test:  {len(X_test)} examples")


if __name__ == "__main__":
    sequence_length = 10  # Example sequence length for your LSTM model

    for i in range(1, 4):
        csv_file = f"act_results_with_labels_sub{i}.csv"

        output_prefix = f"subject{i}_data"

        load_and_split(
            csv_path=csv_file,
            output_prefix=output_prefix,
            test_size=0.15,
            val_size=0.15,
            use_residue=False,
            sequence_length=sequence_length
        )
