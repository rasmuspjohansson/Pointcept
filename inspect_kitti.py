import numpy as np
import argparse
import os

def inspect_kitti_bin(file_path):
    """
    Inspects a Semantic KITTI .bin file to show coordinate ranges.

    Args:
        file_path (str): The path to the velodyne .bin file.
    """
    if not os.path.exists(file_path):
        print(f"Error: File not found at {file_path}")
        return
    if not file_path.endswith('.bin'):
        print(f"Warning: The file '{file_path}' does not have a .bin extension. Attempting to process anyway.")

    try:
        with open(file_path, "rb") as f:
            scan = np.fromfile(f, dtype=np.float32).reshape(-1, 4)


        print("num points in .bin file : " +str(len(scan)))

        coord = scan[:, :3] # x, y, z coordinates
        strength = scan[:, -1] # intensity value

        print(f"Successfully loaded {scan.shape[0]} points from {file_path}")
        print("\nFirst 5 points (x, y, z):")
        print(coord[:5])
        print("\nCoordinate Ranges:")
        print(f"  Min X: {np.min(coord[:, 0]):.3f}, Max X: {np.max(coord[:, 0]):.3f}")
        print(f"  Min Y: {np.min(coord[:, 1]):.3f}, Max Y: {np.max(coord[:, 1]):.3f}")
        print(f"  Min Z: {np.min(coord[:, 2]):.3f}, Max Z: {np.max(coord[:, 2]):.3f}")
        print(f"\nIntensity Ranges:")
        print(f"  Min Intensity: {np.min(strength):.3f}, Max Intensity: {np.max(strength):.3f}")

    except Exception as e:
        print(f"An error occurred while processing the file: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Inspects Semantic KITTI velodyne .bin files to display coordinate and intensity ranges."
    )
    parser.add_argument(
        "file_path",
        type=str,
        help="The path to the Semantic KITTI velodyne .bin file (e.g., /path/to/dataset/sequences/00/velodyne/000000.bin)"
    )
    args = parser.parse_args()

    inspect_kitti_bin(args.file_path)
