import os
import argparse
import json
from natsort import natsorted

def rename_spot_track_files(input_dir, masks_dir, spot_suffix, track_suffix, mask_suffix, dry_run=True):
    spot_files = natsorted([f for f in os.listdir(input_dir) if f.endswith(spot_suffix)])
    track_files = natsorted([f for f in os.listdir(input_dir) if f.endswith(track_suffix)])
    mask_files = natsorted([f for f in os.listdir(masks_dir) if f.endswith(mask_suffix)])

    if not (len(spot_files) == len(track_files) == len(mask_files)):
        raise ValueError(f"Mismatch in number of files:\nSpots: {len(spot_files)}, Tracks: {len(track_files)}, Masks: {len(mask_files)}")

    print(f"{'Dry run' if dry_run else 'Renaming'}: {len(spot_files)} sets of files will be processed.\n")

    for spot_file, track_file, mask_file in zip(spot_files, track_files, mask_files):
        new_base = mask_file.replace(mask_suffix, "")
        new_spot_file = f"{new_base}{spot_suffix}"
        new_track_file = f"{new_base}{track_suffix}"

        old_spot_path = os.path.join(input_dir, spot_file)
        new_spot_path = os.path.join(input_dir, new_spot_file)

        old_track_path = os.path.join(input_dir, track_file)
        new_track_path = os.path.join(input_dir, new_track_file)

        if dry_run:
            print(f"[DRY RUN] Would rename:\n  {spot_file} -> {new_spot_file}\n  {track_file} -> {new_track_file}\n")
        else:
            os.rename(old_spot_path, new_spot_path)
            os.rename(old_track_path, new_track_path)
            print(f"Renamed:\n  {spot_file} -> {new_spot_file}\n  {track_file} -> {new_track_file}\n")

    print("Done.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Rename spots and tracks files based on mask file basenames.")
    parser.add_argument("--config", required=True, help="Path to the pipeline config JSON file.")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = json.load(f)

    path_config = config["path"]

    rename_spot_track_files(
        input_dir=path_config["input_dir"],
        masks_dir=path_config["masks_dir"],
        spot_suffix=path_config.get("spot_suffix", "_spots.csv"),
        track_suffix=path_config.get("track_suffix", "_tracks.csv"),
        mask_suffix=path_config.get("mask_suffix", "_w2T2 GFP_cp_masks.png"),
        dry_run=path_config.get("dry_run",True)
    )
