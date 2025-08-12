import argparse
import numpy as np

def inspect_np(np_file):
    """
    Reads a NumPy .npy file and prints information about its contents.

    Args:
        np_file (str): The path to the input .npy file.
    """
    try:
        # Load the NumPy array
        data = np.load(np_file)
    except Exception as e:
        print(f"Error reading NumPy file: {e}")
        return

    print(f"--- NumPy File Inspection: {np_file} ---")

    # --- General Information ---
    print(f"Shape: {data.shape}")
    print(f"Data Type (dtype): {data.dtype}")
    point_count = data.size
    print(f"Total number of elements: {point_count:,}")

    if point_count == 0:
        print("\nArray is empty. Exiting.")
        return

    # --- Value Range ---
    print("\n--- Value Range ---")
    print(f"Min value: {data.min()}")
    print(f"Max value: {data.max()}")

    # --- Unique Values and Counts (useful for classification arrays) ---
    print("\n--- Unique Values (Classes) ---")
    unique_values, counts = np.unique(data, return_counts=True)
    print(f"Found {len(unique_values)} unique values.")
    for value, count in zip(unique_values, counts):
        print(f"  Value {value}: {count:,} occurrences")

    print("\n--- End of Inspection ---")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inspect a NumPy .npy file and print summary statistics.")
    parser.add_argument("--np_file", required=True, help="Path to the input .npy file.")
    args = parser.parse_args()

    inspect_np(args.np_file)
