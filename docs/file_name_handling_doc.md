# file_name_handling Script


To ensure consistent mapping between spots, tracks, and masks, the pipeline includes a file renaming utility that aligns the base names of spot and track files with the corresponding mask files.
This script standardizes the filenames of **TrackMate spots and tracks** by renaming them to match the base names of their corresponding **segmentation mask** files. This ensures consistent file pairing throughout the pipeline.

This is especially important when:
The segmentation mask files have a consistent naming convention (e.g., sampleX_w2T2 GFP_cp_masks.png)

But the spot/track files exported from TrackMate do not follow that same naming convention

###### How it works
The script uses suffix-based matching to associate each set of files (spot, track, mask). It:

- Loads all files from the input_dir and masks_dir.
- Matches files based on their suffixes (spot_suffix, track_suffix, mask_suffix) as defined in the config file.
- Renames each spot and track file to match the basename of its corresponding mask file.


###### Script Behavior
- If dry_run is enabled, the script prints what it would rename but does not change the files.
- If dry_run is false, it permanently renames the files in place.
- The script will raise an error if the number of spot, track, and mask files do not match.

**Note**
This step is optional, but highly recommended if your filenames are inconsistent. Ensuring consistent basenames across spot, track, and mask files allows the pipeline to correctly pair and analyze data without ambiguity.
---

## Inputs


`--config`(str) (path) : Path to the pipeline configuration `config.json` file. This file must include the `path` section with the keys described below. 
| Input Key           | Type        | Description |
|---------------------|-------------|-------------|
| `input_dir`         | Directory   | Directory containing original TrackMate **spots** and **tracks** files. |
| `masks_dir`         | Directory   | Directory containing **segmentation mask** files (e.g., from Cellpose). |
| `spot_suffix`       | String      | File suffix used to identify spot files (e.g., `_spots.csv`). |
| `track_suffix`      | String      | File suffix used to identify track files (e.g., `_tracks.csv`). |
| `mask_suffix`       | String      | File suffix used to identify mask files (e.g., `_w2T2 GFP_cp_masks.png`). |
| `dry_run`           | Boolean     | If `true`, performs a simulation and prints renaming actions in CLI without changing files. If `false`, files are renamed in-place. |

All suffix values are defined under the `path` section of the config JSON.

---

## Outputs

There are **no new output files** or directories created. The script **renames files in place** within the `input_dir`.

For each matched set (spot, track, mask), the script will:
- Rename the **spot file**:  
  `original_name.csv → mask_basename + spot_suffix`
- Rename the **track file**:  
  `original_name.csv → mask_basename + track_suffix`

### Example

Suppose:
- `mask file`: `sample001_w2T2 GFP_cp_masks.png`
- `spot file`: `trackmate_export_01.csv`
- `track file`: `trackmate_export_01_track.csv`

And the config defines:
```json
{
  "spot_suffix": "_spots.csv",
  "track_suffix": "_tracks.csv",
  "mask_suffix": "_w2T2 GFP_cp_masks.png"
}
```
Then the renamed files will be:
- `sample001_spots.csv`
- `sample001_tracks.csv`

#### Usage Example
``` bash
python file_name_handling.py --config config.json
```
If `dry_run = true` in the `config`, the output will be something like:
``` bash
[DRY RUN] Would rename:
  trackmate_spot_01.csv -> sample001_spots.csv
  trackmate_track_01.csv -> sample001_tracks.csv
```
