import os
import argparse
import numpy as np

def extract_unique_labels(dataset_folder):
    """
    Scans the specified dataset folder for .label files and extracts all unique label IDs.

    Args:
        dataset_folder (str): The root path of the dataset.

    Returns:
        list: A sorted list containing all unique label IDs encountered.
    """
    all_unique_labels = set()
    label_files_processed = 0
    total_label_files_found = 0

    print(f"Scanning dataset folder: {dataset_folder}")

    # Walk through the dataset folder
    for root, _, files in os.walk(dataset_folder):
        for file_name in files:
            if file_name.endswith(".label"):
                total_label_files_found += 1
                label_file_path = os.path.join(root, file_name)

                try:
                    # Read the label file as int32
                    # SemanticKitti labels are typically stored as int32,
                    # where semantic ID is in the lower 16 bits.
                    with open(label_file_path, "rb") as f:
                        labels = np.fromfile(f, dtype=np.int32)
                        # Apply mask to get semantic labels
                        semantic_labels = labels & 0xFFFF
                        all_unique_labels.update(np.unique(semantic_labels).tolist())
                        label_files_processed += 1
                except Exception as e:
                    print(f"Error processing label file {label_file_path}: {e}")

    print(f"\n--- Scan Summary ---")
    print(f"Total .label files found: {total_label_files_found}")
    print(f"Total .label files successfully processed: {label_files_processed}")
    print(f"--- End Summary ---\n")

    return sorted(list(all_unique_labels)) # Return sorted list for consistent output

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Collects all unique label IDs from .label files in a dataset."
    )
    parser.add_argument(
        "--dataset_folder",
        type=str,
        required=True,
        help="Path to the root of your dataset folder (e.g., /mnt/T/mnt/trainingdata/lidar/Dataset2_Aragon_semanticKitty_onlyxyzintensity)"
    )
    args = parser.parse_args()

    if not os.path.isdir(args.dataset_folder):
        print(f"Error: Dataset folder '{args.dataset_folder}' not found or is not a directory.")
    else:
        unique_labels = extract_unique_labels(args.dataset_folder)
        print("List of all unique label IDs encountered:")
        print(unique_labels)

