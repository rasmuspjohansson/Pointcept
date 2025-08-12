
import argparse
import os
import numpy as np
import laspy
from pathlib import Path
from scipy.spatial import cKDTree

def stitch_to_las(pred_dir, original_bin_file, split_dir, output_las):
    """
    Stitches predictions from smaller chunks back into a single, large, classified .las file.

    Args:
        pred_dir (str): Directory containing the .npy prediction files for each chunk.
        original_bin_file (str): Path to the original, large .bin file that was split.
        split_dir (str): The root directory where the split chunks were saved (e.g., the one
                         containing the 'sequences/00_split' folder).
        output_las (str): Path for the final, combined .las file.
    """
    try:
        # --- 1. Load Original Full Point Cloud ---
        print(f"Loading original full point cloud: {original_bin_file}")
        original_scan = np.fromfile(original_bin_file, dtype=np.float32).reshape(-1, 4)
        original_coords = original_scan[:, :3]
        print(f"Loaded {len(original_coords):,} points from original file.")

        # --- 2. Build a KD-Tree for fast point lookup ---
        print("Building KD-Tree for spatial indexing... (This may take a moment)")
        kdtree = cKDTree(original_coords)

        # --- 3. Initialize Final Classification Array ---
        # Start with an "unclassified" value. We'll use 0, but it could be any default.
        final_classification = np.zeros(len(original_coords), dtype=np.uint8)

        # --- 4. Process Each Prediction Chunk ---
        pred_files = sorted(Path(pred_dir).glob("*_pred.npy"))
        if not pred_files:
            print(f"Error: No prediction files found in {pred_dir}")
            return

        print(f"Found {len(pred_files)} prediction chunks to stitch.")

        for i, pred_file in enumerate(pred_files):
            print(f"  Processing chunk {i+1}/{len(pred_files)}: {pred_file.name}")

            # Load the prediction data for the chunk
            chunk_pred_labels = np.load(pred_file)

            # Find the corresponding original chunk .bin file to get coordinates
            base_name = pred_file.name.replace('_pred.npy', '').replace("00_split_","")
            # The name is now the grid index, e.g., "00_000000"
            chunk_bin_file = Path(split_dir) / "sequences" / "00_split"/"velodyne" / f"{base_name}.bin"

            if not chunk_bin_file.exists():
                print(f"    Warning: Could not find matching chunk .bin file: {chunk_bin_file}")
                continue

            chunk_scan = np.fromfile(chunk_bin_file, dtype=np.float32).reshape(-1, 4)
            chunk_coords = chunk_scan[:, :3]

            # Ensure consistency
            if len(chunk_pred_labels) != len(chunk_coords):
                print(f"    Warning: Mismatch in chunk {base_name}. Skipping.")
                continue

            # Find the indices of the chunk points in the original full point cloud
            _, indices = kdtree.query(chunk_coords, k=1)

            # Place the predictions into the final classification array
            # This will overwrite predictions in overlapping regions. The last one wins.
            final_classification[indices] = chunk_pred_labels.astype(np.uint8)

        # --- 5. Create and Save the Final LAS File ---
        print("\nStitching complete. Creating final .las file...")
        # Use point format 3 which includes RGB colors, often better supported.
        header = laspy.LasHeader(point_format=3, version="1.2")
        header.wkt = 'PROJCS["ETRS89 / UTM zone 32N",GEOGCS["ETRS89",DATUM["European_Terrestrial_Reference_System_1989",SPHEROID["GRS 1980",6378137,298.257222101,AUTHORITY["EPSG","7019"]],TOWGS84[0,0,0,0,0,0,0],AUTHORITY["EPSG","6258"]],PRIMEM["Greenwich",0,AUTHORITY["EPSG","8901"]],UNIT["degree",0.0174532925199433,AUTHORITY["EPSG","9122"]],AUTHORITY["EPSG","4258"]],PROJECTION["Transverse_Mercator"],PARAMETER["latitude_of_origin",0],PARAMETER["central_meridian",9],PARAMETER["scale_factor",0.9996],PARAMETER["false_easting",500000],PARAMETER["false_northing",0],UNIT["metre",1,AUTHORITY["EPSG","9001"]],AXIS["Easting",EAST],AXIS["Northing",NORTH],AUTHORITY["EPSG","25832"]]' # noqa

        las = laspy.LasData(header)
        las.x = original_scan[:, 0]
        las.y = original_scan[:, 1]
        las.z = original_scan[:, 2]
        las.intensity = (original_scan[:, 3] * 255).astype(np.uint8)
        las.classification = final_classification

        # Add dummy color information, as it's required by point format 3
        las.red = np.zeros_like(final_classification, dtype=np.uint16)
        las.green = np.zeros_like(final_classification, dtype=np.uint16)
        las.blue = np.zeros_like(final_classification, dtype=np.uint16)

        las.write(output_las)
        print(f"Successfully created final classified LAS file: {output_las}")

    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Stitch multiple chunk predictions into a single classified .las file."
    )
    parser.add_argument(
        "--pred_dir",
        required=True,
        help="Directory containing the .npy prediction files (e.g., .../exp/default/result/)."
    )
    parser.add_argument(
        "--original_bin_file",
        required=True,
        help="Path to the original, large .bin file that was split."
    )
    parser.add_argument(
        "--split_dir",
        required=True,
        help="The root directory where the split chunks were saved."
    )
    parser.add_argument("--output_las", required=True, help="Path for the final output .las file.")
    args = parser.parse_args()

    stitch_to_las(args.pred_dir, args.original_bin_file, args.split_dir, args.output_las)
