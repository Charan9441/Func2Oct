#!/usr/bin/env python3
"""
scripts/inspect_dataset.py

Automates:
1. Ingesting / downloading dataset:
   - Local directory (e.g. 'FundusOCT' containing FundusMD, octMD, Fundusother)
   - Local zip archive (e.g. from Downloads/ or --zip-path)
   - Kaggle API ('saketlad/armd-combined-dataset-fundus-and-oct')
2. Organizing extracted assets into:
   - data/fundus/<class>/
   - data/oct/<class>/
3. Inspecting:
   - Dataset image dimensions (resolutions, min/max/average, aspect ratio)
   - Color channels and modes (RGB, Grayscale, etc.)
   - Class / category distributions across Fundus and OCT
   - Paired sample alignment (with leading-zero normalization like 001 <-> 0001)
   - Exporting data/manifests/dataset.csv
"""

import argparse
import os
import re
import shutil
import sys
import tempfile
import zipfile
from collections import Counter
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

    kaggle_home = Path.home() / ".kaggle"
    try:
        kaggle_home.mkdir(parents=True, exist_ok=True)
        token_file = kaggle_home / "access_token"
        token_file.write_text(api_token.strip(), encoding="utf-8")
    except Exception as e:
        print(f"[Warning] Could not write access token file to {kaggle_home}: {e}")


def normalize_class_name(class_name: str) -> str:
    """Normalizes class names by stripping modality prefixes (FundusMD -> MD, octMD -> MD)."""
    c = class_name.lower().replace("fundus", "").replace("oct", "").strip(" _-")
    return c.upper() if c else class_name


def normalize_stem(stem: str) -> str:
    """Normalizes sample stems (e.g. '001' and '0001' both become '1')."""
    digits = "".join(ch for ch in stem if ch.isdigit())
    if digits:
        return str(int(digits))
    return stem.lower()


def find_modality_dir(search_root: Path, keyword: str) -> Optional[Path]:
    """Recursively searches for a directory corresponding to keyword ('fundus' or 'oct')."""
    keyword_lower = keyword.lower()
    matches = []

    for path in search_root.rglob("*"):
        if path.is_dir() and keyword_lower in path.name.lower():
            img_count = sum(1 for p in path.rglob("*") if p.suffix.lower() in IMAGE_EXTENSIONS)
            if img_count > 0:
                matches.append((path, img_count))

    if not matches:
        return None

    matches.sort(key=lambda x: (x[1], len(x[0].parts)), reverse=True)
    best_candidate = matches[0][0]

    sub_nested = [d for d in best_candidate.iterdir() if d.is_dir() and d.name.lower() == keyword_lower]
    if sub_nested:
        nested_img_count = sum(1 for p in sub_nested[0].rglob("*") if p.suffix.lower() in IMAGE_EXTENSIONS)
        if nested_img_count > 0:
            return sub_nested[0]

    return best_candidate


def copy_or_move_contents(src_dir: Path, dest_dir: Path, clean_class: Optional[str] = None):
    """Copies/moves contents into dest_dir, preserving or normalizing class subfolder."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    target_folder = dest_dir / (clean_class or src_dir.name)
    target_folder.mkdir(parents=True, exist_ok=True)

    for item in src_dir.rglob("*"):
        if item.is_file() and item.suffix.lower() in IMAGE_EXTENSIONS:
            dest_file = target_folder / item.name
            if not dest_file.exists():
                shutil.copy2(str(item), str(dest_file))


def ingest_local_directory(source_dir: Path, base_data_dir: Path):
    """
    Ingests a local dataset folder (like 'FundusOCT' with FundusMD, Fundusother, octMD)
    and places normalized classes into data/fundus and data/oct.
    """
    fundus_dest = base_data_dir / "fundus"
    oct_dest = base_data_dir / "oct"

    print(f"\n[Ingest] Detected local dataset directory: {source_dir}")

    # Check immediate subdirectories
    subdirs = [d for d in source_dir.iterdir() if d.is_dir()]

    for d in subdirs:
        d_lower = d.name.lower()
        clean_cls = normalize_class_name(d.name)
        img_count = sum(1 for _ in d.rglob("*") if _.suffix.lower() in IMAGE_EXTENSIONS)

        if "fundus" in d_lower:
            print(f"  -> Ingesting Fundus subfolder: {d.name} ({img_count} images) as class '{clean_cls}'")
            copy_or_move_contents(d, fundus_dest, clean_class=clean_cls)
        elif "oct" in d_lower:
            print(f"  -> Ingesting OCT subfolder:    {d.name} ({img_count} images) as class '{clean_cls}'")
            copy_or_move_contents(d, oct_dest, clean_class=clean_cls)
        else:
            # Check files inside
            print(f"  -> Ambiguous subfolder {d.name}, checking images...")
            copy_or_move_contents(d, fundus_dest, clean_class=d.name)

    print("[Success] Local directory files mapped into data/fundus/ and data/oct/.")


def extract_local_archive(archive_path: Path, base_data_dir: Path):
    """Extracts a local zip file and arranges contents into data/fundus and data/oct."""
    fundus_dest = base_data_dir / "fundus"
    oct_dest = base_data_dir / "oct"

    if not archive_path.exists():
        raise FileNotFoundError(f"Local archive not found: {archive_path}")

    temp_staging = Path(tempfile.mkdtemp(prefix="local_armd_"))
    print(f"\n[1/2] Extracting local archive '{archive_path.name}' to staging area...")
    print(f"      Staging directory: {temp_staging}")

    try:
        with zipfile.ZipFile(archive_path, "r") as zf:
            zf.extractall(temp_staging)

        # First check if unzipped root contains a FundusOCT-style folder
        fundus_oct_dirs = [d for d in temp_staging.rglob("*") if d.is_dir() and "fundusoct" in d.name.lower()]
        if fundus_oct_dirs:
            ingest_local_directory(fundus_oct_dirs[0], base_data_dir)
            return

        fundus_src = find_modality_dir(temp_staging, "fundus")
        oct_src = find_modality_dir(temp_staging, "oct")

        if not fundus_src:
            raise FileNotFoundError(f"Could not locate 'fundus' directory in archive {archive_path}")
        if not oct_src:
            raise FileNotFoundError(f"Could not locate 'oct' directory in archive {archive_path}")

        print(f"      Detected Fundus source: {fundus_src}")
        print(f"      Detected OCT source:    {oct_src}")

        print(f"[2/2] Organizing files into:\n      -> {fundus_dest}\n      -> {oct_dest}")
        for item in fundus_src.iterdir():
            if item.is_dir():
                clean_cls = normalize_class_name(item.name)
                copy_or_move_contents(item, fundus_dest, clean_class=clean_cls)
            else:
                shutil.copy2(str(item), str(fundus_dest / item.name))

        for item in oct_src.iterdir():
            if item.is_dir():
                clean_cls = normalize_class_name(item.name)
                copy_or_move_contents(item, oct_dest, clean_class=clean_cls)
            else:
                shutil.copy2(str(item), str(oct_dest / item.name))

        print("[Success] Local archive extracted and organized successfully.")
    finally:
        if temp_staging.exists():
            shutil.rmtree(temp_staging, ignore_errors=True)


def auto_find_local_dataset() -> Tuple[Optional[Path], Optional[Path]]:
    """Checks for existing FundusOCT directories or downloaded zip files."""
    search_dirs = [
        Path.cwd().parent / "FundusOCT",
        Path.cwd() / "FundusOCT",
        Path.cwd().parent,
        Path.home() / "Downloads" / "FundusOCT",
        Path.home() / "Downloads",
    ]

    # 1. Check for directory named FundusOCT
    for d in search_dirs:
        if d.exists() and d.is_dir() and "fundusoct" in d.name.lower():
            return d, None

    # 2. Check for zip files
    for d in search_dirs:
        if d.exists() and d.is_dir():
            for p in d.glob("*.zip"):
                name_l = p.name.lower()
                if "fundus" in name_l or "armd" in name_l or "archive" in name_l:
                    return None, p

    return None, None


def setup_data_sources(
    dataset_name: str,
    base_data_dir: Path,
    source_dir: Optional[Path] = None,
    local_zip: Optional[Path] = None,
    kaggle_token: Optional[str] = None,
    force_download: bool = False,
):
    """Sets up data from local directory, zip, or Kaggle API."""
    fundus_dest = base_data_dir / "fundus"
    oct_dest = base_data_dir / "oct"

    fundus_existing = sum(1 for _ in fundus_dest.rglob("*") if _.suffix.lower() in IMAGE_EXTENSIONS)
    oct_existing = sum(1 for _ in oct_dest.rglob("*") if _.suffix.lower() in IMAGE_EXTENSIONS)

    if (fundus_existing > 0 or oct_existing > 0) and not force_download:
        print(f"[Info] Found existing images: {fundus_existing} in {fundus_dest}, {oct_existing} in {oct_dest}.")
        print("[Info] Proceeding directly to inspection (use --force-download to refresh).")
        return

    # Check explicit source directory
    if source_dir and source_dir.exists():
        ingest_local_directory(source_dir, base_data_dir)
        return

    # Check explicit zip file
    if local_zip and local_zip.exists():
        extract_local_archive(local_zip, base_data_dir)
        return

    # Check auto-detected local sources
    auto_dir, auto_zip = auto_find_local_dataset()
    if auto_dir:
        print(f"[Info] Auto-detected local dataset folder at: {auto_dir}")
        ingest_local_directory(auto_dir, base_data_dir)
        return

    if auto_zip:
        print(f"[Info] Auto-detected local archive at: {auto_zip}")
        extract_local_archive(auto_zip, base_data_dir)
        return

    # Fallback to Kaggle API
    ensure_kaggle_credentials(kaggle_token)

    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
        api = KaggleApi()
        api.authenticate()
    except Exception as e:
        print(f"[Error] Failed to authenticate with Kaggle: {e}")
        print("Please provide a local directory using --source-dir or a zip via --zip-path.")
        sys.exit(1)

    temp_staging = Path(tempfile.mkdtemp(prefix="kaggle_armd_"))
    print(f"\n[1/3] Downloading '{dataset_name}' from Kaggle to staging area...")
    print(f"      Staging directory: {temp_staging}")

    try:
        api.dataset_download_files(dataset_name, path=str(temp_staging), unzip=True, quiet=False)
        print("[2/3] Download and extraction complete. Scanning archive contents...")

        for zip_file in temp_staging.glob("*.zip"):
            print(f"      Extracting zip: {zip_file.name}...")
            with zipfile.ZipFile(zip_file, "r") as zf:
                zf.extractall(temp_staging)
            zip_file.unlink()

        fundus_src = find_modality_dir(temp_staging, "fundus")
        oct_src = find_modality_dir(temp_staging, "oct")

        if not fundus_src or not oct_src:
            raise FileNotFoundError(f"Could not locate 'fundus' and 'oct' in {temp_staging}")

        print(f"[3/3] Organizing files into:\n      -> {fundus_dest}\n      -> {oct_dest}")
        for item in fundus_src.iterdir():
            if item.is_dir():
                clean_cls = normalize_class_name(item.name)
                copy_or_move_contents(item, fundus_dest, clean_class=clean_cls)

        for item in oct_src.iterdir():
            if item.is_dir():
                clean_cls = normalize_class_name(item.name)
                copy_or_move_contents(item, oct_dest, clean_class=clean_cls)

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
            parts = rel_path.parts
            if len(parts) > 1:
                class_name = parts[0]
                sample_id = "/".join(parts[1:])
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
    """
    Analyzes pairing between Fundus and OCT images with numeric ID & class normalization.
    (e.g., FundusMD/001.png matches octMD/0001.png via normalized ID '1' and class 'MD')
    """
    print(f"\n{'='*70}")
    print("  PAIRED SAMPLE DISTRIBUTION ANALYSIS")
    print(f"{'='*70}")

    fundus_by_key = {}
    for sample_id, class_name, path in fundus_records:
        stem = Path(sample_id).stem
        norm_id = normalize_stem(stem)
        clean_cls = normalize_class_name(class_name)
        fundus_by_key[(clean_cls, norm_id)] = (stem, path)

    oct_by_key = {}
    for sample_id, class_name, path in oct_records:
        stem = Path(sample_id).stem
        norm_id = normalize_stem(stem)
        clean_cls = normalize_class_name(class_name)
        oct_by_key[(clean_cls, norm_id)] = (stem, path)

    all_keys = set(fundus_by_key.keys()).union(set(oct_by_key.keys()))

    paired_count = 0
    fundus_only_count = 0
    oct_only_count = 0
    paired_by_class = Counter()

    rows = []

    for cls, norm_id in sorted(all_keys, key=lambda x: (x[0], int(x[1]) if x[1].isdigit() else x[1])):
        has_fundus = (cls, norm_id) in fundus_by_key
        has_oct = (cls, norm_id) in oct_by_key
        is_paired = has_fundus and has_oct

        fundus_stem, fundus_path = fundus_by_key[(cls, norm_id)] if has_fundus else ("", "")
        oct_stem, oct_path = oct_by_key[(cls, norm_id)] if has_oct else ("", "")

        if is_paired:
            paired_count += 1
            paired_by_class[cls] += 1
        elif has_fundus:
            fundus_only_count += 1
        else:
            oct_only_count += 1

        rows.append({
            "normalized_id": norm_id,
            "class_name": cls,
            "is_paired": is_paired,
            "has_fundus": has_fundus,
            "has_oct": has_oct,
            "fundus_filename": Path(str(fundus_path)).name if fundus_path else "",
            "oct_filename": Path(str(oct_path)).name if oct_path else "",
            "fundus_path": str(fundus_path),
            "oct_path": str(oct_path),
        })

    total_unique = len(all_keys)
    print(f"Total Unique Keys (Class + Normalized ID): {total_unique}")
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
        "--source-dir",
        type=str,
        default=None,
        help="Path to local unzipped dataset directory (e.g., FundusOCT).",
    )
    parser.add_argument(
        "--zip-path",
        "--archive",
        type=str,
        default=None,
        help="Path to local dataset .zip file (e.g. from Downloads).",
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
        help="Skip downloading/ingesting and inspect existing files directly.",
    )
    parser.add_argument(
        "--force-download",
        action="store_true",
        help="Force re-downloading/re-ingesting even if target folders are non-empty.",
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

    # 1. Setup Data Sources (local directory / local zip / Kaggle API)
    if not args.skip_download:
        src_path = Path(args.source_dir).resolve() if args.source_dir else None
        local_zip_path = Path(args.zip_path).resolve() if args.zip_path else None
        setup_data_sources(
            dataset_name=args.dataset,
            base_data_dir=base_data_dir,
            source_dir=src_path,
            local_zip=local_zip_path,
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
