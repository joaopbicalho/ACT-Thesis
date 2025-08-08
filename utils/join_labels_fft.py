'''This script merges label data from a TSV file with EEG feature data from a CSV file for multiple subjects.
It repeats each label value for a specified number of rows to match the feature data, handles possible row mismatches,
and saves the merged results to new CSV files.
'''

import pandas as pd


def process_table_pair(table1_path, table2_path, repeat_count=4):
    # Load the tables
    table1 = pd.read_csv(table1_path, sep="\t")  # Assuming TSV format for table1
    table2 = pd.read_csv(table2_path)  # Assuming CSV format for table2

    # Check if repeat count matches the data length
    required_rows = len(table1) * repeat_count
    if required_rows > len(table2):
        raise ValueError(
            f"{table2_path} needs at least {required_rows} rows to accommodate the repeated labels."
        )

    # Add the label column to table2 by repeating the ai_hb values for each group of repeat_count rows
    expanded_labels = table1["ai_hb"].repeat(repeat_count).reset_index(drop=True)

    # Adjust table2 if it has extra rows
    if len(expanded_labels) != len(table2):
        # Check if table2 has exactly 2 extra rows
        if len(table2) == len(expanded_labels) + 2:
            # Ignore the last two rows of table2
            table2 = table2.iloc[:-2].reset_index(drop=True)
        else:
            raise ValueError(
                f"Size mismatch after label expansion: expected {len(expanded_labels)} rows in {table2_path}, but found {len(table2)}."
            )

    # Add the expanded labels to table2
    table2["label"] = expanded_labels

    # Keep only the specified columns
    table2 = table2[
        ["Epoch", "Electrode", "Delta", "Theta", "Alpha", "Beta", "Gamma", "label"]
    ]

    # Return the updated table2
    return table2


if __name__ == "__main__":
    subject_ids = [20]  # Replace with the list of subject IDs you want to process
    for sub_id in range(1, 129):
        table1_path = (
            f"sub-{sub_id}/eeg/sub-{sub_id}_task-Sleep_acq-headband_events.tsv"
        )
        table2_path = f"act_results_sub-{sub_id}.csv"

        try:
            # Process the tables
            updated_table = process_table_pair(table1_path, table2_path)

            # Save the updated table to a new CSV file
            output_path = f"act_results_with_labels_sub-{sub_id}.csv"
            updated_table.to_csv(output_path, index=False)
            print(f"Processed and saved for sub-{sub_id} to {output_path}")

        except FileNotFoundError:
            print(f"File for sub-{sub_id} not found.")
        except ValueError as e:
            print(f"Error processing sub-{sub_id}: {e}")
