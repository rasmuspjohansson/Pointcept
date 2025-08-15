
import argparse
import os
import numpy as np
import laspy

def combine_to_las(pred_file, data_root, output_las):
    """
    Combines an original .bin point cloud with a .npy prediction file to create
     a classified .las file.

    Args:
        pred_file (str): Path to the prediction file (e.g., '.../00_000135_pred.npy').
        data_root (str): Path to the root of the dataset to be classified.
        output_las (str): Path for the output .las file.
    """
    try:
        # --- 1. Load Predictions ---
        predictions = np.load(pred_file)

        # --- 2. Find and Load Original Point Cloud ---
        # Extract the data name (e.g., '00_000135') from the prediction filename
        base_name = os.path.basename(pred_file).replace('_pred.npy', '')
        sequence_name, frame_name = base_name.split('_')

        # Reconstruct the path to the original .bin file
        bin_path = os.path.join(data_root, 'sequences', sequence_name, 'velodyne', f"{frame_name}.bin")

        if not os.path.exists(bin_path):
            print(f"Error: Could not find original .bin file at: {bin_path}")
            return

        # Load the .bin file (X, Y, Z, Intensity)
        scan = np.fromfile(bin_path, dtype=np.float32).reshape(-1, 4)

        # --- 3. Check for consistency ---
        if len(predictions) != len(scan):
            print(f"Error: Point count mismatch between prediction ({len(predictions)}) and scan ({len(scan)}).")
            print("This may be due to SphereCrop during testing. Re-run test without SphereCrop for full classification.")
            # As a fallback, we will truncate the longer array to match the shorter one.
            min_points = min(len(predictions), len(scan))
            predictions = predictions[:min_points]
            scan = scan[:min_points]
            print(f"Continuing with {min_points} points.")

        # --- 4. Create and Populate LAS file ---
        # Use a point format that supports intensity and classification
        header = laspy.LasHeader(point_format=2, version="1.2")
        las = laspy.LasData(header)

        # Set the CRS using the appropriate method for your laspy version
        try:
            # For newer laspy versions
            from laspy.crs import CRS
            las.header.add_crs(CRS.from_epsg(25832))
        except ImportError:
            # For older laspy versions
            las.header.wkt = 'PROJCS["ETRS89 / UTM zone 32N",GEOGCS["ETRS89",DATUM["European_Terrestrial_Reference_System_1989",SPHEROID["GRS 1980",6378137,298.257222101,AUTHORITY["EPSG","7019"]],TOWGS84[0,0,0,0,0,0,0],AUTHORITY["EPSG","6258"]],PRIMEM["Greenwich",0,AUTHORITY["EPSG","8901"]],UNIT["degree",0.0174532925199433,AUTHORITY["EPSG","9122"]],AUTHORITY["EPSG","4258"]],PROJECTION["Transverse_Mercator"],PARAMETER["latitude_of_origin",0],PARAMETER["central_meridian",9],PARAMETER["scale_factor",0.9996],PARAMETER["false_easting",500000],PARAMETER["false_northing",0],UNIT["metre",1,AUTHORITY["EPSG","9001"]],AXIS["Easting",EAST],AXIS["Northing",NORTH],AUTHORITY["EPSG","25832"]]'

        las.x = scan[:, 0]
        las.y = scan[:, 1]
        las.z = scan[:, 2]
        las.intensity = (scan[:, 3] * 255).astype(np.uint8) # Scale intensity to 0-255 for LAS
        las.classification = predictions.astype(np.uint8)

        # --- 5. Write to file ---
        las.write(output_las)
        print(f"Successfully created classified LAS file: {output_las}")

    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":

    print("start")
    parser = argparse.ArgumentParser(
        description="Combine a .bin point cloud and a .npy prediction into a classified .las file."
    )
    parser.add_argument(
        "--pred_file",
        required=True,
        help="Path to the input NumPy prediction file (e.g., .../result/00_000135_pred.npy)."
    )
    parser.add_argument(
        "--data_root",
        required=True,
        help="Path to the root of the original dataset (e.g., .../dataset_to_be_classified/)."
    )
    parser.add_argument("--output_las", required=True, help="Path for the output .las file.")
    args = parser.parse_args()

    combine_to_las(args.pred_file, args.data_root, args.output_las)
