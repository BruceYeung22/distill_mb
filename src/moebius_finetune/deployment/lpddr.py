"""LPDDR4X bandwidth model (TDD §8.3).

The default platform assumption is LPDDR4X-4266, 64-bit bus, peak
~34.1 GB/s. The effective bandwidth in the field depends on the
board layout and is **not** the peak. We use the 10/15/20 GB/s
triplet as the default scenarios.

The latency lower bound is just ``bytes / effective_bandwidth`` —
no scheduler / kernel overhead is modelled. Anything that includes
those is a measurement, not a lower bound.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Tuple

import math


__all__ = ["BandwidthModel", "latency_lower_bound"]


@dataclass
class BandwidthModel:
    """A simple LPDDR4X bandwidth scenario.

    Parameters
    ----------
    peak_gbps
        Theoretical peak (GB/s). Default 34.1 GB/s for LPDDR4X-4266.
    effective_gbps_options
        The triplet of effective bandwidth scenarios (GB/s). The
        default is ``(10, 15, 20)`` per TDD §8.3.
    """

    peak_gbps: float = 34.1
    effective_gbps_options: Tuple[float, ...] = (10.0, 15.0, 20.0)

    def latency_lower_bound(self, bytes_traffic: int, *, effective_gbps: float) -> float:
        return latency_lower_bound(bytes_traffic, effective_gbps=effective_gbps)

    def latency_scenarios(self, bytes_traffic: int) -> dict:
        out: dict = {}
        for gbps in self.effective_gbps_options:
            out[f"{gbps:.0f}gbps"] = {
                "effective_gbps": float(gbps),
                "bytes": int(bytes_traffic),
                "latency_ms": float(latency_lower_bound(bytes_traffic, effective_gbps=gbps)),
            }
        return out


def latency_lower_bound(bytes_traffic: int, *, effective_gbps: float) -> float:
    """Return the time (ms) to move ``bytes_traffic`` at ``effective_gbps``.

    Uses decimal SI units: ``1 GB = 1e9 bytes`` and ``1 s = 1000 ms``.
    This matches TDD §8.1 ("MB uses decimal 10^6 bytes"). 300 MB at
    10 GB/s therefore yields 30 ms.

    The function is symmetric: doubling the bytes doubles the time.
    Zero bytes gives zero ms; non-positive effective bandwidth
    raises a ``ValueError``.
    """
    if effective_gbps <= 0:
        raise ValueError(f"effective_gbps must be > 0, got {effective_gbps}")
    if bytes_traffic < 0:
        raise ValueError(f"bytes_traffic must be >= 0, got {bytes_traffic}")
    # 1 GB = 1e9 bytes; 1 s = 1000 ms.
    return float(bytes_traffic) / (float(effective_gbps) * 1e9) * 1000.0
