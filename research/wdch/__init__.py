"""Wavelet-Decoupled Contextual Homogenization research components."""

from .haar_wavelet import FixedHaarDWT2D
from .bcch import BoundaryAwareContext, HFRMBCCH
from .cbcch import HFRMCBCCH, LocalSemanticAffinity, contrastive_affinity_loss
from .wdch import HFRMWDCH, WaveletDecoupledContext

__all__ = [
    "BoundaryAwareContext",
    "FixedHaarDWT2D",
    "HFRMBCCH",
    "HFRMCBCCH",
    "HFRMWDCH",
    "LocalSemanticAffinity",
    "WaveletDecoupledContext",
    "contrastive_affinity_loss",
]
