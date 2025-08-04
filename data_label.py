import pandas as pd
import numpy as np

def join_labels(
    csv_path,
    tsv_path,
    output_path,
    fs=256,
    epoch_time=15.0,
    overlap=0.75,
    label_col="ai_hb"
):
    """
    Matches per-epoch data in the CSV file to 30-second labels in the TSV file,
    discarding epochs that cross differing labels.

    Parameters
    ----------
    csv_path : str
        Path to the CSV file containing the ACT results (or other epoch-based data).
    tsv_path : str
        Path to the TSV file containing 30-second labels.
    output_path : str
        Where to write the CSV results with assigned labels.
    fs : int, optional
        Sampling rate in Hz. Default 256.
    epoch_time : float, optional
        Epoch length in seconds. Default 15.
    overlap : float, optional
        Fractional overlap between epochs (e.g., 0.20 => 20% overlap).
    label_col : str, optional
        Name of the column in the TSV that contains labels. Default 'ai_hb'.
    """

    # --- 1) Read input files ---
    df_csv = pd.read_csv(csv_path)
    df_tsv = pd.read_csv(tsv_path, sep="\t")

    # The TSV has columns like [onset, duration, ai_hb (the label)]
    if not set(["onset", "duration", label_col]).issubset(df_tsv.columns):
        raise ValueError(
            f"TSV file must have columns onset, duration, and {label_col}. "
            f"Columns found: {df_tsv.columns}"
        )

    # Build a list of label intervals from the TSV
    # Each row is e.g. onset=0, duration=30 => [0,30) has label "ai_hb"...
    label_intervals = []
    for _, row in df_tsv.iterrows():
        start_sec = row["onset"]
        end_sec = start_sec + row["duration"]
        label_value = row[label_col]
        label_intervals.append((start_sec, end_sec, label_value))

    # Compute epoch start/end times in seconds for each CSV row ---
    # For epoch i (1-based):
    #   start_time = (i - 1) * stride_samples / fs
    #   end_time   = start_time + epoch_time
    # where stride_samples = epoch_time * fs * (1 - overlap)
    stride_samples = int(epoch_time * fs * (1.0 - overlap))

    def get_label_for_epoch(epoch_index):
        # 1-based index of epoch
        start_sample = (epoch_index - 1) * stride_samples
        start_time = start_sample / fs
        end_time = start_time + epoch_time
        return find_label_in_interval(start_time, end_time)

    #  Function to find the correct label for a time interval 
    def find_label_in_interval(t0, t1):
        """
        Returns the label if [t0, t1) overlaps only intervals with the same label.
        Otherwise returns None if it spans different labels or no label at all.
        """
        # Find intervals from label_intervals that overlap with [t0, t1).
        # Overlap condition: an interval [s,e) overlaps if s < t1 and e > t0.
        overlapping = []
        for (start_sec, end_sec, lab) in label_intervals:
            if (start_sec < t1) and (end_sec > t0):
                overlapping.append(lab)

        if len(overlapping) == 0:
            # No intervals found => no label
            return None
        # If more than one label is present in 'overlapping' and they differ, discard
        unique_labels = set(overlapping)
        if len(unique_labels) == 1:
            # All the same label => that's our label
            return unique_labels.pop()
        # Otherwise multiple distinct => discard
        return None

    # --- 5) Apply label assignment to each row in df_csv ---
    labels_assigned = []
    for i, row in df_csv.iterrows():
        epoch_idx = row["Epoch"]
        lab = get_label_for_epoch(epoch_idx)
        labels_assigned.append(lab)

    # Add new column "label" 
    df_csv["label"] = labels_assigned

    # If you want to discard rows with no label, drop them:
    df_csv = df_csv.dropna(subset=["label"]).reset_index(drop=True)

    #  Save the output with new label column 
    df_csv.to_csv(output_path, index=False)
    print(f"Saved labeled epochs to {output_path}")


if __name__ == "__main__":
    # Process each subject's data
    for i in range(1, 4):
        csv_file = f"act_results_sub-{i}.csv"
        tsv_file = f"sub-{i}_task-Sleep_acq-headband_events.tsv"
        out_file = f"act_results_with_labels_sub{i}.csv"

        print(f"Processing subject {i}...")
        
        join_labels(
            csv_path=csv_file,
            tsv_path=tsv_file,
            output_path=out_file,
            fs=256,
            epoch_time=15,
            overlap=0.75,
            label_col="ai_hb"
        )
    
    print("Finished processing all subjects.")

