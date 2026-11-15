"""HiPPO-SVGP utilities used by experimental continuous-time networks."""

from .hippo import HiPPO_LegS
from .kernel import FreqSpaceKernel, SE_kernel, SMK_kernel

__all__ = ["HiPPO_LegS", "SE_kernel", "SMK_kernel", "FreqSpaceKernel"]
