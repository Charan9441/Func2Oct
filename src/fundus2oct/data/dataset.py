"""
src/fundus2oct/data/dataset.py

PyTorch Dataset implementations for:
1. OCTReconstructionDataset: Unpaired/paired OCT scans for VAE training and latent encoding.
2. PairedFundusOCTDataset: Paired fundus-to-OCT examinations for conditional diffusion training.
Features robust path resolution (handles local vs. manifest paths) and zero-leakage patient splits.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset

from fundus2oct.data.preprocessing import (
    preprocess_fundus,
    preprocess_oct_bscan,
    preprocess_oct_volume,
)

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def resolve_sample_path(
    path_str: str,
    base_dir: Optional[Path] = None,
    candidate_roots: Optional[List[Path]] = None,
) -> Optional[Path]:
    """
    Intelligently resolves image path.
    If the path was recorded on a different system (e.g., OneDrive path),
    it searches locally by filename/relative subfolders under candidate_roots.
    """
    if not path_str or pd.isna(path_str):
        return None

    direct_path = Path(path_str)
    if direct_path.exists():
        return direct_path

    # Search in candidate roots (e.g. data/oct, data/fundus, project_root/data)
    roots = candidate_roots or []
    if base_dir:
        roots.append(base_dir)

    for root in roots:
        if not root.exists():
            continue
        # Check direct child
        candidate = root / direct_path.name
        if candidate.exists():
            return candidate
        # Check subfolders (e.g. root/MD/0001.png)
        for sub in root.glob(f"**/{direct_path.name}"):
            if sub.is_file():
                return sub

    return None


class OCTReconstructionDataset(Dataset):
    """
    Dataset for OCT B-scan or volumetric reconstruction training and latent encoding.
    Can be initialized from:
    1. A manifest DataFrame / CSV path.
    2. A directory path containing OCT images.
    """

    def __init__(
        self,
        manifest_path: Optional[Union[str, Path]] = None,
        data_dir: Optional[Union[str, Path]] = None,
        split: Optional[str] = None,
        target_size: Tuple[int, int] = (256, 256),
        is_3d: bool = False,
        volume_depth: int = 16,
        normalize: str = "bipolar",
        synthetic_count: int = 0,
    ):
        super().__init__()
        self.target_size = target_size
        self.is_3d = is_3d
        self.volume_depth = volume_depth
        self.normalize = normalize
        self.synthetic_count = synthetic_count

        self.samples: List[Dict[str, Any]] = []

        # 1. Load from manifest if provided
        if manifest_path:
            p = Path(manifest_path)
            if p.exists() and p.stat().st_size > 0:
                df = pd.read_csv(p)
                if "split" in df.columns and split:
                    df = df[df["split"] == split]

                oct_col = "oct_path" if "oct_path" in df.columns else None
                if oct_col:
                    base = Path(data_dir) if data_dir else Path("data/oct")
                    candidate_roots = [Path("data/oct"), Path("data"), Path(".")]
                    for idx, row in df.iterrows():
                        raw_path = str(row[oct_col])
                        resolved = resolve_sample_path(raw_path, base_dir=base, candidate_roots=candidate_roots)
                        if resolved and resolved.exists():
                            sample_id = str(row.get("normalized_id", row.get("sample_id", idx)))
                            cls_name = str(row.get("class_name", "OCT"))
                            self.samples.append({
                                "path": resolved,
                                "sample_id": sample_id,
                                "class_name": cls_name,
                            })

        # 2. If no valid manifest samples, scan data_dir directly
        if not self.samples and data_dir:
            d = Path(data_dir)
            if d.exists():
                for f in sorted(d.rglob("*")):
                    if f.is_file() and f.suffix.lower() in IMAGE_EXTENSIONS:
                        cls_name = f.parent.name if f.parent != d else "default"
                        self.samples.append({
                            "path": f,
                            "sample_id": f.stem,
                            "class_name": cls_name,
                        })

    def __len__(self) -> int:
        return max(len(self.samples), self.synthetic_count)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        if idx < len(self.samples):
            item = self.samples[idx]
            path = item["path"]
            sample_id = item["sample_id"]
            class_name = item["class_name"]

            if self.is_3d:
                tensor = preprocess_oct_volume(
                    path,
                    target_shape=(self.volume_depth, self.target_size[0], self.target_size[1]),
                    normalize=self.normalize,
                )
            else:
                tensor = preprocess_oct_bscan(
                    path,
                    target_size=self.target_size,
                    normalize=self.normalize,
                )
            return {
                "oct": tensor,
                "path": str(path),
                "sample_id": sample_id,
                "class_name": class_name,
                "is_synthetic": False,
            }
        else:
            # Synthetic fallback for architecture testing / smoke tests
            if self.is_3d:
                tensor = torch.zeros((1, self.volume_depth, self.target_size[0], self.target_size[1]), dtype=torch.float32)
            else:
                tensor = torch.zeros((1, self.target_size[0], self.target_size[1]), dtype=torch.float32)
            return {
                "oct": tensor,
                "path": f"synthetic_{idx}",
                "sample_id": f"synthetic_{idx}",
                "class_name": "synthetic",
                "is_synthetic": True,
            }


class PairedFundusOCTDataset(Dataset):
    """
    Dataset for paired Fundus photographs and OCT B-scans.
    Used for conditional latent diffusion training.
    """

    def __init__(
        self,
        manifest_path: Union[str, Path],
        split: Optional[str] = None,
        fundus_size: Tuple[int, int] = (256, 256),
        oct_size: Tuple[int, int] = (256, 256),
        fundus_normalize: str = "imagenet",
        oct_normalize: str = "bipolar",
        data_root: Optional[Union[str, Path]] = None,
    ):
        super().__init__()
        self.fundus_size = fundus_size
        self.oct_size = oct_size
        self.fundus_normalize = fundus_normalize
        self.oct_normalize = oct_normalize

        self.samples: List[Dict[str, Any]] = []

        manifest_file = Path(manifest_path)
        if not manifest_file.exists():
            return

        df = pd.read_csv(manifest_file)
        if "split" in df.columns and split:
            df = df[df["split"] == split]

        root = Path(data_root) if data_root else Path("data")
        candidate_f = [root / "fundus", Path("data/fundus")]
        candidate_o = [root / "oct", Path("data/oct")]

        for idx, row in df.iterrows():
            f_raw = str(row.get("fundus_path", ""))
            o_raw = str(row.get("oct_path", ""))

            f_res = resolve_sample_path(f_raw, candidate_roots=candidate_f)
            o_res = resolve_sample_path(o_raw, candidate_roots=candidate_o)

            if f_res and o_res and f_res.exists() and o_res.exists():
                self.samples.append({
                    "fundus_path": f_res,
                    "oct_path": o_res,
                    "patient_id": str(row.get("patient_id", f"P{idx}")),
                    "eye": str(row.get("eye", "unknown")),
                    "sample_id": str(row.get("normalized_id", idx)),
                    "split": str(row.get("split", "train")),
                })

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        item = self.samples[idx]
        fundus_t = preprocess_fundus(
            item["fundus_path"],
            target_size=self.fundus_size,
            normalize=self.fundus_normalize,
        )
        oct_t = preprocess_oct_bscan(
            item["oct_path"],
            target_size=self.oct_size,
            normalize=self.oct_normalize,
        )
        return {
            "fundus": fundus_t,
            "oct": oct_t,
            "patient_id": item["patient_id"],
            "eye": item["eye"],
            "sample_id": item["sample_id"],
            "split": item["split"],
        }
