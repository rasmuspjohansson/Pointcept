import os
import argparse
import numpy as np
import laspy
from tqdm import tqdm
def convert_laz_to_bin_and_label(laz_file, bin_path, label_path):
    las = laspy.read(laz_file)

    xyz = np.vstack((las.x, las.y, las.z)).T

    if hasattr(las, 'red') and hasattr(las, 'green') and hasattr(las, 'blue'):
        rgb = np.vstack((las.red, las.green, las.blue)).T / 65535.0
    else:
        rgb = np.zeros_like(xyz)

    if hasattr(las, 'intensity'):
        intensity_raw = las.intensity
        intensity = (
            intensity_raw.array if hasattr(intensity_raw, "array") else intensity_raw
        ).reshape(-1, 1).astype(np.float32)
    else:
        raise ValueError(f"No intensity field found in {laz_file}")

    points = np.hstack((xyz, intensity, rgb)).astype(np.float32)
    points.tofile(bin_path)

    if hasattr(las, 'classification'):
        label_raw = las.classification
        labels = (
            label_raw.array if hasattr(label_raw, "array") else label_raw
        ).astype(np.uint32)
    else:
        raise ValueError(f"No classification labels found in {laz_file}")

    labels.tofile(label_path)

def main():
    parser = argparse.ArgumentParser(description="Convert LAZ files to SemanticKITTI format")
    parser.add_argument('--laz_folder', type=str, required=True, help='Folder containing .laz files with RGBXYZ+intensity+label')
    parser.add_argument('--output_folder', type=str, required=True, help='Folder to write SemanticKITTI-style output')
    args = parser.parse_args()

    laz_folder = args.laz_folder
    output_folder = args.output_folder

    # SemanticKITTI expects this structure
    velodyne_dir = os.path.join(output_folder, 'sequences', '00', 'velodyne')
    label_dir = os.path.join(output_folder, 'sequences', '00', 'labels')
    os.makedirs(velodyne_dir, exist_ok=True)
    os.makedirs(label_dir, exist_ok=True)

    laz_files = sorted([f for f in os.listdir(laz_folder) if f.lower().endswith('.laz')])

    for i, laz_file in enumerate(tqdm(laz_files, desc="Converting")):
        input_path = os.path.join(laz_folder, laz_file)
        base_name = f"{i:06d}"  # Name like 000000.bin
        bin_path = os.path.join(velodyne_dir, base_name + '.bin')
        label_path = os.path.join(label_dir, base_name + '.label')

        convert_laz_to_bin_and_label(input_path, bin_path, label_path)

    print("✅ Conversion complete.")

if __name__ == '__main__':
    main()
