"""Wavelet-Decoupled Contextual Homogenization research components."""

from .haar_wavelet import FixedHaarDWT2D
from .wdch import (
    HFRMSCWDCH,
    HFRMWDCH,
    StrengthCalibratedWaveletContext,
    WaveletDecoupledContext,
)

__all__ = [
    "FixedHaarDWT2D",
    "HFRMSCWDCH",
    "HFRMWDCH",
    "StrengthCalibratedWaveletContext",
    "WaveletDecoupledContext",
]
