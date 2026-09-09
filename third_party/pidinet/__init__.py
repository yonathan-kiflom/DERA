"""The PiDiNet feature pyramid used by DERA.

Only the pixel-difference operators and encoder blocks required by DERA are
kept here.  See this directory's provenance and license files before reuse.
"""

from .pdc import PDCConv2d, make_pdc
from .pyramid import PDC_ARCHITECTURES, PDCBlock, PiDiNetPyramid

__all__ = ["PDC_ARCHITECTURES", "PDCBlock", "PDCConv2d", "PiDiNetPyramid", "make_pdc"]
