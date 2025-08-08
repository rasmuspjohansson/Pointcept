import argparse
import os
import laspy
import numpy as np


def split_laz_file(path_to_laz_file, width, height, overlap, output_folder):
    # Load the .laz file
    print(f"Loading: {path_to_laz_file}")
    las = laspy.read(path_to_laz_file)
    points = las.points

    x = las.x
    y = las.y

    min_x, max_x = x.min(), x.max()
    min_y, max_y = y.min(), y.max()

    step_x = width - overlap
    step_y = height - overlap

    base_name = os.path.splitext(os.path.basename(path_to_laz_file))[0]
    os.makedirs(output_folder, exist_ok=True)

    count = 0
    for x0 in np.arange(min_x, max_x, step_x):
        for y0 in np.arange(min_y, max_y, step_y):
            x1 = x0 + width
            y1 = y0 + height

            mask = (x >= x0) & (x < x1) & (y >= y0) & (y < y1)
            if not np.any(mask):
                continue

            sub_las = laspy.LasData(las.header)
            sub_las.points = points[mask]

            tile_name = f"{base_name}_{int(x0)}_{int(y0)}.laz"
            tile_path = os.path.join(output_folder, tile_name)

            sub_las.write(tile_path)
            count += 1
            print(f"Saved tile: {tile_path}")

    print(f"Total tiles saved: {count}")


def main():
    parser = argparse.ArgumentParser(description="Split .laz file into tiles")
    parser.add_argument("--path_to_laz_file", required=True, help="Path to input .laz file")
    parser.add_argument("--width", type=float, default=75, help="Tile width (default: 75)")
    parser.add_argument("--height", type=float, default=75, help="Tile height (default: 75)")
    parser.add_argument("--overlap", type=float, default=2, help="Tile overlap (default: 2)")
    parser.add_argument("--output_folder", required=True, help="Folder to save the tiles")

    args = parser.parse_args()
    split_laz_file(args.path_to_laz_file, args.width, args.height, args.overlap, args.output_folder)


if __name__ == "__main__":
    main()
