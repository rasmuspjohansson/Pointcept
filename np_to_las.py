
import numpy as np
import laspy
import argparse


def np_to_las(np_file, output_las):
    # Load the numpy array
    predictions = np.load(np_file)

    # Create a new LAS file
    header = laspy.LasHeader(point_format=2, version="1.2")
    las = laspy.LasData(header)

    # Set the point coordinates (assuming the original point cloud data is not available,
    # we will create a dummy point cloud with the same number of points as the predictions)
    num_points = len(predictions)
    las.x = np.zeros(num_points)
    las.y = np.zeros(num_points)
    las.z = np.zeros(num_points)

    # Set the classification for each point
    las.classification = predictions.astype(np.uint8)

    # Write the LAS file
    las.write(output_las)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert a NumPy prediction file to a LAS file.")
    parser.add_argument("--np_file", required=True, help="Path to the input NumPy file.")
    parser.add_argument("--output_las", required=True, help="Path to the output LAS file.")
    args = parser.parse_args()

    np_to_las(args.np_file, args.output_las)
