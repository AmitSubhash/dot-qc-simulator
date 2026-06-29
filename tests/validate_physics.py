"""Standalone validation of the simulator's diffuse-optics physics.

It parses the actual constants embedded in ``dot_qc_simulator.py`` (extinction
coefficients, DPF, and the real cap geometry) so the checks cannot drift out of
sync with the app, then verifies the core invariants:

  1. Hemoglobin extinction coefficients match the Prahl/OMLC values and respect
     the ~800 nm isosbestic ordering.
  2. The modified Beer-Lambert forward -> inverse round-trip recovers HbO/HbR to
     machine precision.
  3. The cap is the 26-source / 31-detector pad with NN bands at 10/22/30/36 mm.

Run directly (``python3 tests/validate_physics.py``) or via ``pytest``.
"""
from __future__ import annotations

import ast
import math
from pathlib import Path

import numpy as np

APP = Path(__file__).resolve().parents[1] / "dot_qc_simulator.py"


def _load_constants() -> dict:
    """Pull the literal constants we validate out of the app via the AST."""
    tree = ast.parse(APP.read_text())
    want = {"EXT", "DPF", "_SRC_XY", "_DET_XY"}
    found: dict = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            target = node.targets[0]
            if isinstance(target, ast.Name) and target.id in want and target.id not in found:
                found[target.id] = ast.literal_eval(node.value)
    missing = want - set(found)
    assert not missing, f"could not parse constants from app: {missing}"
    return found


C = _load_constants()
EXT = C["EXT"]
DPF = C["DPF"]
SRC = np.array(C["_SRC_XY"], dtype=float)
DET = np.array(C["_DET_XY"], dtype=float)


def test_extinction_coefficients():
    """Prahl/OMLC values, base-10 cm^-1/M, with the correct isosbestic ordering."""
    assert EXT[750]["HbO"] == 518.0
    assert EXT[750]["HbR"] == 1405.24
    assert EXT[850]["HbO"] == 1058.0
    assert EXT[850]["HbR"] == 691.32
    # below the ~800 nm isosbestic HbR absorbs more; above it HbO absorbs more
    assert EXT[750]["HbR"] > EXT[750]["HbO"], "750 nm: HbR should exceed HbO"
    assert EXT[850]["HbO"] > EXT[850]["HbR"], "850 nm: HbO should exceed HbR"


def test_mbll_round_trip():
    """Forward (HbO/HbR -> OD) then inverse (OD -> HbO/HbR) must be the identity."""
    E = np.array([[EXT[750]["HbO"], EXT[750]["HbR"]],
                  [EXT[850]["HbO"], EXT[850]["HbR"]]])
    e_inv = np.linalg.inv(E)
    d_cm = 3.0
    truth = np.array([0.55e-6, -0.21e-6])   # [HbO, HbR] in M

    # forward: natural-log OD per wavelength
    od = {}
    for wl in (750, 850):
        od[wl] = math.log(10) * DPF[wl] * d_cm * (
            EXT[wl]["HbO"] * truth[0] + EXT[wl]["HbR"] * truth[1]
        )
    # inverse
    y = np.array([od[750] / (math.log(10) * DPF[750] * d_cm),
                  od[850] / (math.log(10) * DPF[850] * d_cm)])
    recovered = e_inv @ y

    err = float(np.max(np.abs(recovered - truth)))
    assert err < 1e-15, f"MBLL round-trip error too large: {err:.2e} M"


def test_cap_geometry():
    """The embedded pad is 26 sources / 31 detectors with NN bands 10/22/30/36 mm."""
    assert SRC.shape == (26, 2), f"expected 26 sources, got {SRC.shape}"
    assert DET.shape == (31, 2), f"expected 31 detectors, got {DET.shape}"
    rsd = np.hypot(SRC[:, None, 0] - DET[None, :, 0], SRC[:, None, 1] - DET[None, :, 1])
    bands = set(np.unique(np.round(rsd[rsd <= 40]).astype(int)).tolist())
    for nn in (10, 22, 30, 36):
        assert nn in bands, f"missing nearest-neighbour band ~{nn} mm (got {sorted(bands)})"
    assert int((rsd <= 40).sum()) > 100, "too few measurement pairs within 40 mm"


def main() -> None:
    checks = [test_extinction_coefficients, test_mbll_round_trip, test_cap_geometry]
    for check in checks:
        check()
        print(f"PASS  {check.__name__}")
    # a couple of headline numbers for the log
    E = np.array([[EXT[750]["HbO"], EXT[750]["HbR"]], [EXT[850]["HbO"], EXT[850]["HbR"]]])
    print(f"\nMBLL matrix condition number: {np.linalg.cond(E):.1f}")
    rsd = np.hypot(SRC[:, None, 0] - DET[None, :, 0], SRC[:, None, 1] - DET[None, :, 1])
    print(f"cap: {SRC.shape[0]} sources, {DET.shape[0]} detectors, "
          f"{int((rsd <= 40).sum())} pairs <= 40 mm")
    print("\nAll physics checks passed.")


if __name__ == "__main__":
    main()
