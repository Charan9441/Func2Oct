"""
src/fundus2oct/inference/__init__.py
"""

from .generate_oct import generate_oct_volume
from .encode_latents import encode_oct_dataset

__all__ = [
    "generate_oct_volume",
    "encode_oct_dataset",
]
