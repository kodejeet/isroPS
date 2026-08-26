"""CLI utility script for inspecting image dimensions, metadata, and channel stats."""

import argparse
import os
import sys

sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
)

from lunar_correspondence.io.image_loader import load_image
from lunar_correspondence.planetary.instruments import get_instrument_info


def main():
    parser = argparse.ArgumentParser(
        description="Inspect lunar image characteristics and metadata."
    )
    parser.add_argument("image_path", type=str, help="Path to image file")
    parser.add_argument(
        "--instrument",
        type=str,
        default="UNKNOWN",
        help="Sensor name (OHRC, TMC-2, IIRS, LRO_NAC, SELENE)",
    )
    args = parser.parse_args()

    print(f"[*] Inspecting image: {args.image_path}")
    img_data = load_image(args.image_path, instrument=args.instrument)
    inst_info = get_instrument_info(img_data.metadata.instrument)

    print("\n--- Image Properties ---")
    print(f" Array Shape:     {img_data.array.shape} (Height, Width, Channels)")
    print(f" Data Type:       {img_data.array.dtype}")
    print(f" Min / Max Value: {img_data.array.min()} / {img_data.array.max()}")
    print("\n--- Metadata ---")
    print(f" Instrument Key:  {img_data.metadata.instrument}")
    print(f" Full Name:       {inst_info['name']}")
    print(f" Sensor Type:     {inst_info['type']}")
    print(f" Plain English:   {inst_info['plain_english']}")
    print(f" Resolution GSD:  {img_data.metadata.resolution_m_per_px} m/px")
    print(f" Processing Hint: {inst_info['processing_hints']}")


if __name__ == "__main__":
    main()
