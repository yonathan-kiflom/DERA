"""DERA: edge-guided residual adaptation for X-ray object detection.

The package root stays deliberately lightweight so dataset validation and
command-line help do not initialize PyTorch, MMCV, or MMDetection. Import
model classes from :mod:`dera.models`; MMEngine configurations register them
through their ``custom_imports`` setting.
"""

__version__ = "0.1.0"

__all__ = ["__version__"]
