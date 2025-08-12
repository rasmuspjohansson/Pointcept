
import argparse
import os
import numpy as np
from pathlib import Path

def split_bin_for_inference(input_file, output_dir, chunk_size=50.0, overlap=10.0):
    """
    Splits a large .bin point cloud file into smaller, overlapping chunks suitable for inference.

    Creates a directory structure that mimics the SemanticKITTI format, which is expected
    by the Pointcept testing tools.

    Args:
        input_file (str): Path to the large input .bin file.
        output_dir (str): The root directory where the split data will be saved.
        chunk_size (float): The side length of each square chunk in meters.
        overlap (float): The overlap between adjacent chunks in meters.
    """
    print(f"Loading large .bin file: {input_file}...")
    try:
        scan = np.fromfile(input_file, dtype=np.float32).reshape(-1, 4)
        print(f"Loaded {len(scan):,} points.")
    except Exception as e:
        print(f"Error reading .bin file: {e}")
        return

    # Determine the spatial extent of the entire point cloud
    x_min, y_min = scan[:, 0].min(), scan[:, 1].min()
    x_max, y_max = scan[:, 0].max(), scan[:, 1].max()

    # --- Create the output directory structure ---
    # The script will create a new "sequences" for the chunks.
    # Example: <output_dir>/sequences/00_split/velodyne/
    sequence_dir = Path(output_dir) / "sequences" / "00_split"
    velodyne_dir = sequence_dir / "velodyne"
    velodyne_dir.mkdir(parents=True, exist_ok=True)
    print(f"Created output directory: {velodyne_dir}")

    # --- Iterate and create chunks ---
    stride = chunk_size - overlap
    x_steps = np.arange(x_min, x_max, stride)
    y_steps = np.arange(y_min, y_max, stride)
    total_chunks = len(x_steps) * len(y_steps)
    processed_chunks = 0

    for i, x in enumerate(x_steps):
        for j, y in enumerate(y_steps):
            # Define the boundaries for the current chunk
            chunk_x_min = x
            chunk_x_max = x + chunk_size
            chunk_y_min = y
            chunk_y_max = y + chunk_size

            # Select points that fall within the chunk boundaries
            chunk_mask = (
                (scan[:, 0] >= chunk_x_min) & (scan[:, 0] < chunk_x_max) &
                (scan[:, 1] >= chunk_y_min) & (scan[:, 1] < chunk_y_max)
            )

            chunk_points = scan[chunk_mask]

            # Skip empty chunks
            if len(chunk_points) == 0:
                continue

            # Save the chunk to a new .bin file using grid-based naming
            output_filename = velodyne_dir / f"{i:02d}_{j:06d}.bin"
            chunk_points.astype(np.float32).tofile(output_filename)
            processed_chunks += 1

    print(f"\nSuccessfully split the file into {processed_chunks} chunks out of {total_chunks} possible grid locations.")
    print(f"Next Step: Run inference on the directory: {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Split a large .bin file into smaller, overlapping chunks for inference."
    )
    parser.add_argument("--input_file", required=True, help="Path to the large input .bin file.")
    parser.add_argument(
        "--output_dir",
        required=True,
        help="The root directory to save the new 'sequences' folder."
    )
    parser.add_argument(
        "--chunk_size",
        type=float,
        default=50.0,
        help="The side length of each square chunk in meters."
    )
    parser.add_argument(
        "--overlap",
        type=float,
        default=10.0,
        help="The overlap between adjacent chunks in meters."
    )
    args = parser.parse_args()

    split_bin_for_inference(args.input_file, args.output_dir, args.chunk_size, args.overlap)
