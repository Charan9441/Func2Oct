"""
Fundus2OCT: Generative AI Architecture and Research Prototype
Generating Synthetic Optical Coherence Tomography (OCT) from 2D Retinal Fundus Photographs.
"""

__version__ = "0.1.0"
__author__ = "Fundus2OCT Research Team"

from fundus2oct.config import AppConfig, load_config

__all__ = ["AppConfig", "load_config", "__version__"]
