"""Reference computation of Hami's GRT masks to record fixtures."""
import json
import sys
from pathlib import Path

import torch

# Use the Hami repo as the reference.
HAMI_ROOT = Path(r"D:\project\Hami")
sys.path.insert(0, str(HAMI_ROOT / "src"))

from hami.data.grt_mask import grt_mask, normalize_disparity  # noqa: E402


def compute_one(name, d, direction):
    t = torch.as_tensor(d, dtype=torch.float32)
    m = grt_mask(t, direction)
    return m.cpu().numpy().astype(bool).tolist()


def main():
    fixtures = {}

    # Test 1: two-plane (left half deep disparity 3, right half disparity 0)
    d1 = [[3.0] * 4 + [0.0] * 4 for _ in range(8)]
    fixtures["two_plane"] = {
        "shape": [8, 8],
        "disparity": d1,
        "R": compute_one("two_plane_R", d1, "R"),
        "L": compute_one("two_plane_L", d1, "L"),
    }

    # Test 2: flat (all equal disparity 2.0) -> no holes (ties survive)
    d2 = [[2.0] * 8 for _ in range(8)]
    fixtures["flat_ties"] = {
        "shape": [8, 8],
        "disparity": d2,
        "R": compute_one("flat_R", d2, "R"),
        "L": compute_one("flat_L", d2, "L"),
    }

    # Test 3: out-of-bounds (left col has disparity past W)
    d3 = [[0.0] * 8 for _ in range(8)]
    for r in range(8):
        d3[r][0] = 8.0 + 5.0  # 13, beyond W=8
    fixtures["out_of_bounds"] = {
        "shape": [8, 8],
        "disparity": d3,
        "R": compute_one("oob_R", d3, "R"),
        "L": compute_one("oob_L", d3, "L"),
    }

    # Test 4: L/R flip — ramp gradient disparity
    d4 = [
        [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
        [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
        [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
        [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
    ]
    fixtures["gradient_ramp"] = {
        "shape": [4, 8],
        "disparity": d4,
        "R": compute_one("ramp_R", d4, "R"),
        "L": compute_one("ramp_L", d4, "L"),
    }

    # Test 5: sparse single point (verify single-pixel occlusion handling)
    d5 = [[0.0] * 8 for _ in range(8)]
    d5[4][4] = 5.0  # single point in the middle, big disparity
    fixtures["single_point"] = {
        "shape": [8, 8],
        "disparity": d5,
        "R": compute_one("sp_R", d5, "R"),
        "L": compute_one("sp_L", d5, "L"),
    }

    # Reference disparity-to-signed-normalized behaviour
    d6 = [0.05, 0.5, 0.95, 1.0, 1.5, 2.0]
    p99 = torch.quantile(torch.tensor(d6, dtype=torch.float32), 0.99).item()
    print(f"ref p99 = {p99}")
    fixtures["signed_p99"] = {
        "values": d6,
        "p99": p99,
    }

    out_path = Path(__file__).parent / "hami_grt_reference.json"
    out_path.write_text(json.dumps(fixtures, indent=2), encoding="utf-8")
    print(f"wrote reference to {out_path}")

    # Also pretty print some
    print("two_plane R:")
    for row in fixtures["two_plane"]["R"]:
        print("  ", " ".join("1" if v else "0" for v in row))
    print("two_plane L:")
    for row in fixtures["two_plane"]["L"]:
        print("  ", " ".join("1" if v else "0" for v in row))

    print("out_of_bounds R:")
    for row in fixtures["out_of_bounds"]["R"]:
        print("  ", " ".join("1" if v else "0" for v in row))

    print("gradient_ramp R:")
    for row in fixtures["gradient_ramp"]["R"]:
        print("  ", " ".join("1" if v else "0" for v in row))
    print("gradient_ramp L:")
    for row in fixtures["gradient_ramp"]["L"]:
        print("  ", " ".join("1" if v else "0" for v in row))


if __name__ == "__main__":
    main()
