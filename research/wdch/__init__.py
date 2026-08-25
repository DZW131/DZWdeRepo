"""Wavelet-Decoupled Contextual Homogenization research components."""

from .haar_wavelet import FixedHaarDWT2D
from .bcch import BoundaryAwareContext, HFRMBCCH
from .bcpch import HFRMBCPCH, LowFrequencyPrototypeRecovery, low_frequency_reconstruction
from .cbcch import HFRMCBCCH, LocalSemanticAffinity, contrastive_affinity_loss
from .wdch import HFRMWDCH, WaveletDecoupledContext

__all__ = [
    "BoundaryAwareContext",
    "FixedHaarDWT2D",
    "HFRMBCCH",
    "HFRMBCPCH",
    "HFRMCBCCH",
    "HFRMWDCH",
    "LocalSemanticAffinity",
    "LowFrequencyPrototypeRecovery",
    "WaveletDecoupledContext",
    "contrastive_affinity_loss",
    "low_frequency_reconstruction",
]
