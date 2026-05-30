import os
import sys
import csv
import argparse

import numpy as np
import pydicom
import torch
import tqdm


def parse_args():
    parser = argparse.ArgumentParser(description='Preprocess DICOM videos to resized .pt cache')
    parser.add_argument('--csv_path', type=str, required=True, help='Path to dataset CSV')
    parser.add_argument('--dicom_root', type=str, required=True, help='Root directory for DICOM files')
    parser.add_argument('--cache_dir', type=str, required=True, help='Output directory for .pt cache')
    parser.add_argument('--resize', type=int, default=112, help='Resize spatial dimensions')
    return parser.parse_args()


def load_csv(csv_path, dicom_root):
    samples = []
    with open(csv_path, "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            pid = row["patient_id"].strip()
            tp = row["timepoint"].strip()

            for view, key in [("A4C", "video_a4c"), ("A2C", "video_a2c")]:
                fname = row.get(key, "").strip()
                if not fname:
                    continue
                dcm_path = os.path.join(dicom_root, pid, tp, fname)
                if not os.path.exists(dcm_path):
                    continue
                samples.append((pid, tp, view, dcm_path))

    print(f"Found {len(samples)} files to process")
    return samples


def process_dicom(dcm_path, resize_size):
    ds = pydicom.dcmread(dcm_path)
    video = ds.pixel_array
    video = np.ascontiguousarray(video, dtype=np.float32)
    video = video.transpose(3, 0, 1, 2)

    video_tensor = torch.from_numpy(video)
    if resize_size:
        from torchvision import transforms
        video_tensor = transforms.Resize((resize_size, resize_size), antialias=True)(video_tensor)

    return video_tensor


def main():
    args = parse_args()

    samples = load_csv(args.csv_path, args.dicom_root)

    processed = 0
    skipped = 0

    for pid, tp, view, dcm_path in tqdm.tqdm(samples):
        cache_dir = os.path.join(args.cache_dir, pid)
        os.makedirs(cache_dir, exist_ok=True)
        cache_path = os.path.join(cache_dir, f"{tp}_{view}.pt")

        if os.path.exists(cache_path):
            skipped += 1
            continue

        try:
            video_tensor = process_dicom(dcm_path, args.resize)
            torch.save(video_tensor, cache_path)
            processed += 1
        except Exception as e:
            print(f"\nError processing {dcm_path}: {e}")

    print(f"\nDone: {processed} processed, {skipped} skipped, {processed + skipped}/{len(samples)} total")


if __name__ == "__main__":
    main()
