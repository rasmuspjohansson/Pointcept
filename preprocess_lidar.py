import os
import argparse
import numpy as np
import laspy
from tqdm import tqdm
from pathlib import Path
from sklearn.neighbors import NearestNeighbors

def read_laz_file(path):
    las = laspy.read(path)
    points = np.vstack([las.x, las.y, las.z]).T
    colors = np.vstack([las.red, las.green, las.blue]).T
    try:
        labels = las.classification
    except:
        labels = np.zeros(points.shape[0], dtype=np.int32)
    return np.hstack([points, colors]), labels

def tile_pointcloud(points, labels, block_size=10.0, stride=10.0, min_points=1024):
    xyz = points[:, :3]
    min_coords = xyz.min(axis=0)
    max_coords = xyz.max(axis=0)
    blocks = []

    x_range = np.arange(min_coords[0], max_coords[0], stride)
    y_range = np.arange(min_coords[1], max_coords[1], stride)

    for x in x_range:
        for y in y_range:
            mask = (
                (xyz[:, 0] >= x) & (xyz[:, 0] < x + block_size) &
                (xyz[:, 1] >= y) & (xyz[:, 1] < y + block_size)
            )
            if np.sum(mask) < min_points:
                continue
            block_points = points[mask]
            block_labels = labels[mask]
            # Normalize to block center
            center = np.array([x + block_size / 2, y + block_size / 2, 0])
            block_points[:, :3] -= center
            blocks.append((block_points, block_labels))
    return blocks

def estimate_normals(points, k=16):
    xyz = points[:, :3]
    nbrs = NearestNeighbors(n_neighbors=k + 1).fit(xyz)
    _, indices = nbrs.kneighbors(xyz)
    normals = []
    for i in range(len(points)):
        neighbors = xyz[indices[i][1:]]  # skip the point itself
        cov = np.cov(neighbors - neighbors.mean(axis=0), rowvar=False)
        _, _, vh = np.linalg.svd(cov)
        normal = vh[-1]
        normals.append(normal)
    return np.array(normals)

def process_file(filepath, output_dir, block_size, stride, min_points, add_normals):
    points, labels = read_laz_file(filepath)
    blocks = tile_pointcloud(points, labels, block_size, stride, min_points)
    stem = Path(filepath).stem
    for i, (block_points, block_labels) in enumerate(blocks):
        if add_normals:
            normals = estimate_normals(block_points)
            block_points = np.hstack([block_points, normals])
        out_path = os.path.join(output_dir, f"{stem}_block{i:03d}.npz")
        np.savez_compressed(out_path, points=block_points.astype(np.float32), labels=block_labels.astype(np.int64))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", required=True, help="Directory with .laz/.las/.txt files")
    parser.add_argument("--output_dir", required=True, help="Where to save processed blocks")
    parser.add_argument("--block_size", type=float, default=10.0, help="Block size in meters")
    parser.add_argument("--stride", type=float, default=10.0, help="Stride for sliding blocks")
    parser.add_argument("--min_points", type=int, default=1024, help="Minimum points per block to keep")
    parser.add_argument("--normals", action="store_true", help="Whether to compute normals")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    files = list(Path(args.input_dir).glob("*.laz")) + list(Path(args.input_dir).glob("*.las"))
    for file in tqdm(files, desc="Processing files"):
        process_file(file, args.output_dir, args.block_size, args.stride, args.min_points, args.normals)

if __name__ == "__main__":
    main()
