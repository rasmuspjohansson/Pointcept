import argparse
import laspy


def print_bounds(path_to_laz_file):
    print(f"Loading: {path_to_laz_file}")
    las = laspy.read(path_to_laz_file)

    min_x, max_x = las.x.min(), las.x.max()
    min_y, max_y = las.y.min(), las.y.max()
    min_z, max_z = las.z.min(), las.z.max()

    print(f"X: min = {min_x}, max = {max_x}, range = {max_x - min_x}")
    print(f"Y: min = {min_y}, max = {max_y}, range = {max_y - min_y}")
    print(f"Z: min = {min_z}, max = {max_z}, range = {max_z - min_z}")


def main():
    parser = argparse.ArgumentParser(description="Print min/max/range of X, Y, Z from a .laz file")
    parser.add_argument("--path_to_laz_file", required=True, help="Path to input .laz file")
    args = parser.parse_args()

    print_bounds(args.path_to_laz_file)


if __name__ == "__main__":
    main()
