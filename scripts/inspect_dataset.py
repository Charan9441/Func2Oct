#!/usr/bin/env python3
"""
scripts/inspect_dataset.py

Automates:
1. Downloading and unzipping the Kaggle dataset:
   'saketlad/armd-combined-dataset-fundus-and-oct'
2. Organizing extracted assets into:
   - data/fundus/
   - data/oct/
3. Inspecting:
   - Dataset image dimensions (resolutions, min/max/median)
   - Color channels and modes (RGB, Grayscale, bit depth)
   - Class / category distributions across Fundus and OCT
   - Paired sample alignment, matching rates, and manifests
"""

import argparse
import os
import shutil
import sys
import tempfile
import zipfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

try:
    from PIL import Image
except ImportError:
    print("Error: Pillow is required. Please install it using: pip install pillow")
    sys.exit(1)

try:
    import pandas as pd
except ImportError:
    pd = None

DEFAULT_DATASET = "saketlad/armd-combined-dataset-fundus-and-oct"
DEFAULT_KAGGLE_TOKEN = "KGAT_24bf2aeaeb521cf604edbb1eb093b158"
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def ensure_kaggle_credentials(token: Optional[str] = None):
    """Configures Kaggle credentials using environment variable or config file."""
    api_token = token or os.environ.get("KAGGLE_API_TOKEN") or DEFAULT_KAGGLE_TOKEN
    os.environ["KAGGLE_API_TOKEN"] = api_token

    # Also write to ~/.kaggle/access_token as fallback
    kaggle_home = Path.home() / ".kaggle"
    try:
        kaggle_home.mkdir(parents=True, exist_ok=True)
        token_file = kaggle_home / "access_token"
        token_file.write_text(api_token.strip(), encoding="utf-8")
    except Exception as e:
        print(f"[Warning] Could not write access token file to {kaggle_home}: {e}")


def find_modality_dir(search_root: Path, keyword: str) -> Optional[Path]:
    """
    Recursively searches for a directory corresponding to 'keyword' (e.g., 'fundus' or 'oct')
    and returns the innermost directory containing image files or class subfolders.
    """
    keyword_lower = keyword.lower()
    matches = []

    for path in search_root.rglob("*"):
        if path.is_dir() and keyword_lower in path.name.lower():
            # Check if this folder or its direct children have images
            img_count = sum(1 for p in path.rglob("*") if p.suffix.lower() in IMAGE_EXTENSIONS)
            if img_count > 0:
                matches.append((path, img_count))

    if not matches:
        return None

    # Sort to pick directory with the highest image count and most specific depth
    matches.sort(key=lambda x: (x[1], len(x[0].parts)), reverse=True)
    best_candidate = matches[0][0]

    # If the candidate has a nested folder with the exact same name (e.g., Fundus/Fundus), unnest
    sub_nested = [d for d in best_candidate.iterdir() if d.is_dir() and d.name.lower() == keyword_lower]
    if sub_nested:
        nested_img_count = sum(1 for p in sub_nested[0].rglob("*") if p.suffix.lower() in IMAGE_EXTENSIONS)
        if nested_img_count > 0:
            return sub_nested[0]

    return best_candidate


def move_modality_contents(src_dir: Path, dest_dir: Path):
    """Moves contents of src_dir into dest_dir, preserving class subfolders."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    for item in src_dir.iterdir():
        target = dest_dir / item.name
        if target.exists():
            if target.is_dir():
                # Merge directory contents
                for sub in item.rglob("*"):
                    if sub.is_file():
                        rel = sub.relative_to(item)
                        sub_target = target / rel
                        sub_target.parent.mkdir(parents=True, exist_ok=True)
                        shutil.move(str(sub), str(sub_target))
            else:
                shutil.move(str(item), str(target))
        else:
            shutil.move(str(item), str(dest_dir))


def download_and_extract_dataset(
    dataset_name: str,
    base_data_dir: Path,
    kaggle_token: Optional[str] = None,
    force_download: bool = False,
):
    """Downloads dataset from Kaggle and arranges into base_data_dir/fundus and base_data_dir/oct."""
    fundus_dest = base_data_dir / "fundus"
    oct_dest = base_data_dir / "oct"

    fundus_existing = sum(1 for _ in fundus_dest.rglob("*") if _.suffix.lower() in IMAGE_EXTENSIONS)
    oct_existing = sum(1 for _ in oct_dest.rglob("*") if _.suffix.lower() in IMAGE_EXTENSIONS)

    if (fundus_existing > 0 or oct_existing > 0) and not force_download:
        print(f"[Info] Found existing images: {fundus_existing} in {fundus_dest}, {oct_existing} in {oct_dest}.")
        print("[Info] Skipping download. Use --force-download to redownload.")
        return

    ensure_kaggle_credentials(kaggle_token)

    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
        api = KaggleApi()
        api.authenticate()
    except Exception as e:
        print(f"[Error] Failed to authenticate with Kaggle: {e}")
        print("Please verify your KAGGLE_API_TOKEN or internet connection.")
        sys.exit(1)

    temp_staging = Path(tempfile.mkdtemp(prefix="kaggle_armd_"))
    print(f"\n[1/3] Downloading '{dataset_name}' from Kaggle to staging area...")
    print(f"      Staging directory: {temp_staging}")

    try:
        api.dataset_download_files(dataset_name, path=str(temp_staging), unzip=True, quiet=False)
        print("[2/3] Download and extraction complete. Scanning archive contents...")

        # If a zip remained without being auto-unzipped
        for zip_file in temp_staging.glob("*.zip"):
            print(f"      Extracting zip: {zip_file.name}...")
            with zipfile.ZipFile(zip_file, "r") as zf:
                zf.extractall(temp_staging)
            zip_file.unlink()

        # Locate fundus and oct folders
        fundus_src = find_modality_dir(temp_staging, "fundus")
        oct_src = find_modality_dir(temp_staging, "oct")

        if not fundus_src:
            raise FileNotFoundError(f"Could not locate 'fundus' directory in extracted dataset at {temp_staging}")
        if not oct_src:
            raise FileNotFoundError(f"Could not locate 'oct' directory in extracted dataset at {temp_staging}")

        print(f"      Detected Fundus source: {fundus_src}")
        print(f"      Detected OCT source:    {oct_src}")

        print(f"[3/3] Organizing files into:\n      -> {fundus_dest}\n      -> {oct_dest}")
        move_modality_contents(fundus_src, fundus_dest)
        move_modality_contents(oct_src, oct_dest)

        print("[Success] Dataset files organized successfully.")

    finally:
        if temp_staging.exists():
            shutil.rmtree(temp_staging, ignore_errors=True)


def scan_modality_files(root_dir: Path) -> List[Tuple[str, str, Path]]:
    """
    Scans image files under root_dir.
    Returns list of tuples: (relative_sample_id, class_name, file_path)
    """
    records = []
    if not root_dir.exists():
        return records

    for file_path in root_dir.rglob("*"):
        if file_path.is_file() and file_path.suffix.lower() in IMAGE_EXTENSIONS:
            rel_path = file_path.relative_to(root_dir)
            # Class name is the immediate subfolder under root, or 'unclassified'
            parts = rel_path.parts
            if len(parts) > 1:
                class_name = parts[0]
                sample_id = "/".join(parts[1:])  # e.g., 001.png or subfolder/001.png
            else:
                class_name = "default"
                sample_id = parts[0]

            records.append((sample_id, class_name, file_path))

    return records


def inspect_modality(
    name: str,
    records: List[Tuple[str, str, Path]],
    sample_limit: Optional[int] = None,
) -> Dict:
    """Inspects dimensions, channels, formats, and class distribution for a modality."""
    print(f"\n{'='*70}")
    print(f"  INSPECTION: {name.upper()} MODALITY")
    print(f"{'='*70}")

    total_images = len(records)
    print(f"Total Images Found: {total_images}")

    if total_images == 0:
        print("  [Warning] No images found to inspect.")
        return {
            "total": 0,
            "classes": {},
            "resolutions": {},
            "modes": {},
            "channels": {},
        }

    # Class distribution
    class_counts = Counter(r[1] for r in records)
    print("\nClass Distribution:")
    for cls, count in class_counts.most_common():
        pct = (count / total_images) * 100
        print(f"  - {cls:<25} : {count:>5} images ({pct:>5.1f}%)")

    # Image properties inspection
    to_inspect = records if sample_limit is None else records[:sample_limit]
    if sample_limit and sample_limit < total_images:
        print(f"\nInspecting sample of {len(to_inspect)} images for dimensions & channels...")
    else:
        print(f"\nInspecting all {len(to_inspect)} images for dimensions & channels...")

    resolutions = Counter()
    modes = Counter()
    channels = Counter()
    aspect_ratios = []
    widths = []
    heights = []

    for _, _, path in to_inspect:
        try:
            with Image.open(path) as img:
                w, h = img.size
                widths.append(w)
                heights.append(h)
                resolutions[(w, h)] += 1
                modes[img.mode] += 1
                aspect_ratios.append(round(w / h, 2))

                # Channel count
                if img.mode in ("RGB", "YCbCr"):
                    c = 3
                elif img.mode in ("RGBA", "CMYK"):
                    c = 4
                elif img.mode in ("L", "P", "1", "I", "F"):
                    c = 1
                else:
                    c = len(img.getbands())
                channels[c] += 1
        except Exception as e:
            print(f"  [Warning] Failed to read {path.name}: {e}")

    print("\nColor Modes & Channels:")
    for mode, count in modes.most_common():
        pct = (count / len(to_inspect)) * 100
        print(f"  - Mode '{mode}'               : {count:>5} ({pct:>5.1f}%)")
    for ch, count in channels.most_common():
        pct = (count / len(to_inspect)) * 100
        print(f"  - Channels: {ch}              : {count:>5} ({pct:>5.1f}%)")

    print("\nResolutions (Width x Height):")
    for (w, h), count in resolutions.most_common(10):
        pct = (count / len(to_inspect)) * 100
        print(f"  - {w} x {h:<16} : {count:>5} images ({pct:>5.1f}%)")
    if len(resolutions) > 10:
        print(f"  ... and {len(resolutions) - 10} other unique resolution(s)")

    if widths and heights:
        min_w, max_w = min(widths), max(widths)
        min_h, max_h = min(heights), max(heights)
        avg_w = sum(widths) / len(widths)
        avg_h = sum(heights) / len(heights)
        print("\nDimension Summary:")
        print(f"  - Width  : min={min_w}, max={max_w}, avg={avg_w:.1f}")
        print(f"  - Height : min={min_h}, max={max_h}, avg={avg_h:.1f}")
        print(f"  - Common Aspect Ratio : {Counter(aspect_ratios).most_common(1)[0][0]}")

    return {
        "total": total_images,
        "classes": dict(class_counts),
        "resolutions": dict(resolutions),
        "modes": dict(modes),
        "channels": dict(channels),
    }


def inspect_pairings_and_manifest(
    fundus_records: List[Tuple[str, str, Path]],
    oct_records: List[Tuple[str, str, Path]],
    manifest_path: Optional[Path] = None,
) -> Dict:
    """Analyzes pairing between Fundus and OCT images and writes manifest if requested."""
    print(f"\n{'='*70}")
    print("  PAIRED SAMPLE DISTRIBUTION ANALYSIS")
    print(f"{'='*70}")

    # Build lookup dictionaries: key by normalized file stem / relative path
    # Often pairs share the same stem (e.g. '001' or '001.png') and class folder
    fundus_by_key = {}
    for sample_id, class_name, path in fundus_records:
        stem = Path(sample_id).stem
        fundus_by_key[(class_name, stem)] = (sample_id, path)

    oct_by_key = {}
    for sample_id, class_name, path in oct_records:
        stem = Path(sample_id).stem
        oct_by_key[(class_name, stem)] = (sample_id, path)

    all_keys = set(fundus_by_key.keys()).union(set(oct_by_key.keys()))

    paired_count = 0
    fundus_only_count = 0
    oct_only_count = 0
    paired_by_class = Counter()

    rows = []

    for cls, stem in sorted(all_keys):
        has_fundus = (cls, stem) in fundus_by_key
        has_oct = (cls, stem) in oct_by_key
        is_paired = has_fundus and has_oct

        fundus_path_str = str(fundus_by_key[(cls, stem)][1]) if has_fundus else ""
        oct_path_str = str(oct_by_key[(cls, stem)][1]) if has_oct else ""

        if is_paired:
            paired_count += 1
            paired_by_class[cls] += 1
        elif has_fundus:
            fundus_only_count += 1
        else:
            oct_only_count += 1

        rows.append({
            "sample_id": stem,
            "class_name": cls,
            "is_paired": is_paired,
            "has_fundus": has_fundus,
            "has_oct": has_oct,
            "fundus_path": fundus_path_str,
            "oct_path": oct_path_str,
        })

    total_unique = len(all_keys)
    print(f"Total Unique Keys (Class + Stem): {total_unique}")
    if total_unique > 0:
        paired_pct = (paired_count / total_unique) * 100
        print(f"  - Perfectly Paired Samples   : {paired_count:>5} ({paired_pct:>5.1f}%)")
        print(f"  - Fundus-Only Samples        : {fundus_only_count:>5}")
        print(f"  - OCT-Only Samples           : {oct_only_count:>5}")

        print("\nPaired Samples Breakdown by Class:")
        for cls, count in paired_by_class.most_common():
            print(f"  - {cls:<25} : {count:>5} pairs")

    if manifest_path and pd is not None and rows:
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        df = pd.DataFrame(rows)
        df.to_csv(manifest_path, index=False)
        print(f"\n[Manifest] Saved dataset manifest to: {manifest_path}")

    return {
        "total_unique": total_unique,
        "paired_count": paired_count,
        "fundus_only": fundus_only_count,
        "oct_only": oct_only_count,
        "paired_by_class": dict(paired_by_class),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Download, organize, and inspect the Fundus & OCT dataset."
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=DEFAULT_DATASET,
        help=f"Kaggle dataset slug (default: '{DEFAULT_DATASET}')",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default=None,
        help="Path to data directory (default: project_root/data)",
    )
    parser.add_argument(
        "--token",
        type=str,
        default=None,
        help="Kaggle API token (KGAT_... or credentials)",
    )
    parser.add_argument(
        "--skip-download",
        action="store_true",
        help="Skip downloading and inspect existing files directly.",
    )
    parser.add_argument(
        "--force-download",
        action="store_true",
        help="Force re-downloading even if target folders are non-empty.",
    )
    parser.add_argument(
        "--sample-limit",
        type=int,
        default=None,
        help="Limit number of images to read for detailed dimension checks (default: all).",
    )
    parser.add_argument(
        "--no-manifest",
        action="store_true",
        help="Do not generate data/manifests/dataset.csv.",
    )

    args = parser.parse_args()

    # Determine project root and data directory
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent

    if args.data_dir:
        base_data_dir = Path(args.data_dir).resolve()
    else:
        base_data_dir = project_root / "data"

    fundus_dir = base_data_dir / "fundus"
    oct_dir = base_data_dir / "oct"
    manifest_path = None if args.no_manifest else (base_data_dir / "manifests" / "dataset.csv")

    print(f"{'='*70}")
    print("  FUNDUS2OCT DATASET MANAGER & INSPECTION TOOL")
    print(f"{'='*70}")
    print(f"Project Root : {project_root}")
    print(f"Data Dir     : {base_data_dir}")
    print(f"Dataset Slug : {args.dataset}")

    # 1. Download & extraction
    if not args.skip_download:
        download_and_extract_dataset(
            dataset_name=args.dataset,
            base_data_dir=base_data_dir,
            kaggle_token=args.token,
            force_download=args.force_download,
        )

    # 2. Scan modality files
    fundus_records = scan_modality_files(fundus_dir)
    oct_records = scan_modality_files(oct_dir)

    # 3. Inspect individual modalities
    inspect_modality("Fundus", fundus_records, sample_limit=args.sample_limit)
    inspect_modality("OCT", oct_records, sample_limit=args.sample_limit)

    # 4. Inspect paired distribution and generate manifest
    inspect_pairings_and_manifest(fundus_records, oct_records, manifest_path=manifest_path)

    print(f"\n{'='*70}")
    print("  INSPECTION COMPLETE")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
