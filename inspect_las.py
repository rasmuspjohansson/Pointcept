
import argparse
import laspy
import numpy as np

def inspect_las(las_file):
    """
    Reads a LAS file and prints information about its dimensions and classification.

    Args:
        las_file (str): The path to the input LAS file.
    """
    try:
        # Read the LAS file
        las = laspy.read(las_file)
    except Exception as e:
        print(f"Error reading LAS file: {e}")
        return

    print("--- LAS File Inspection: {las_file} ---")

    # --- General Information ---
    point_count = len(las.points)
    print(f"Total number of points: {point_count:,}")
    print(f"Point Format ID: {las.header.point_format.id}")
    print(f"Point Record Length: {las.header.point_format.size}")

    if point_count == 0:
        print("\nFile contains no points. Exiting.")
        return

    # --- Dimensional Information ---
    print("\n--- Dimensions (Min, Max, Mean) ---")
    x_dim = np.array(las.x)
    y_dim = np.array(las.y)
    z_dim = np.array(las.z)

    print(f"X: {x_dim.min():.3f}, {x_dim.max():.3f}, {x_dim.mean():.3f}")
    print(f"Y: {y_dim.min():.3f}, {y_dim.max():.3f}, {y_dim.mean():.3f}")
    print(f"Z: {z_dim.min():.3f}, {z_dim.max():.3f}, {z_dim.mean():.3f}")

    # --- Intensity Information (if available) ---
    if hasattr(las, "intensity"):
        print("\n--- Intensity ---")
        intensity_dim = las.intensity
        print(f"Intensity: {intensity_dim.min()}, {intensity_dim.max()}")

    # --- Classification Information ---
    if hasattr(las, "classification"):
        print("\n--- Classification ---")
        class_dim = las.classification
        unique_classes, counts = np.unique(class_dim, return_counts=True)
        print(f"Found {len(unique_classes)} unique classes.")
        for cls, count in zip(unique_classes, counts):
            print(f"  Class {cls}: {count:,} points")

    print("\n--- End of Inspection ---")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inspect a LAS file and print summary statistics.")
    parser.add_argument("--las_file", required=True, help="Path to the input LAS file.")
    args = parser.parse_args()

    inspect_las(args.las_file)
