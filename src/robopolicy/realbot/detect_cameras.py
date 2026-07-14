"""Discover stable camera ids + serial ports for the SO-101 config (§10.5).

Run this ONCE with the arm and both cameras plugged in, then paste the printed
values into configs/smolvla_so101.yaml. Pin cameras by a stable identifier, not a
bare index — macOS AVFoundation indices can shuffle between USB re-plugs (§12).

    python -m robopolicy.realbot.detect_cameras

On macOS this reports:
  * AVFoundation cameras with their Unique IDs (from system_profiler)
  * serial ports likely belonging to the SO-101 leader/follower (/dev/tty.usb*)
  * which OpenCV camera indices actually open + their live resolution (if cv2)

Nothing here writes the config for you — mapping "which physical camera is
overhead vs wrist" needs a human eye. Cover a lens and re-run to disambiguate.
"""

from __future__ import annotations

import glob
import json
import platform
import subprocess
import sys


def _mac_cameras() -> list[dict]:
    """Cameras known to AVFoundation, with their stable Unique IDs."""
    try:
        out = subprocess.run(
            ["system_profiler", "-json", "SPCameraDataType"],
            capture_output=True, text=True, timeout=15, check=True,
        ).stdout
        data = json.loads(out).get("SPCameraDataType", [])
    except (subprocess.SubprocessError, json.JSONDecodeError, OSError) as e:
        print(f"  (could not query system_profiler: {e})")
        return []
    cams = []
    for item in data:
        name = item.get("_name", "?")
        # key name varies by macOS version; grab whatever looks like a unique id
        uid = (item.get("spcamera_unique-id")
               or item.get("spcamera_model-id")
               or next((v for k, v in item.items() if "unique" in k.lower()), "?"))
        cams.append({"name": name, "unique_id": uid})
    return cams


def _serial_ports() -> list[str]:
    """USB serial ports — the SO-101 leader/follower arms show up here.

    Match the common macOS USB-serial bridge names. The SO-101 uses a WCH
    (CH340-family) bridge -> tty.wchusbserial*; others show as usbserial /
    usbmodem / SLAB_USBtoUART (CP210x). We report tty.* (what LeRobot expects on
    mac); the matching cu.* alias exists too.
    """
    patterns = ["usb*", "wchusbserial*", "usbmodem*", "usbserial*", "SLAB_USBtoUART*"]
    ports = set()
    for pat in patterns:
        ports.update(glob.glob(f"/dev/tty.{pat}"))
    return sorted(ports)


def _opencv_probe(max_index: int = 8) -> list[dict]:
    """Which integer indices open under OpenCV, and at what resolution."""
    try:
        import cv2
    except ImportError:
        print("  (opencv-python not installed — skipping live index probe; "
              "install with pip install -e '.[realbot]')")
        return []
    found = []
    for idx in range(max_index):
        cap = cv2.VideoCapture(idx)
        if cap is not None and cap.isOpened():
            ok, frame = cap.read()
            res = f"{frame.shape[1]}x{frame.shape[0]}" if ok and frame is not None else "opened, no frame"
            found.append({"index": idx, "resolution": res})
            cap.release()
    return found


def main() -> int:
    print("=" * 66)
    print("SO-101 device discovery  (paste results into configs/smolvla_so101.yaml)")
    print("=" * 66)

    if platform.system() != "Darwin":
        print(f"\nNOTE: this helper is tuned for macOS; detected {platform.system()}.")
        print("On Linux, cameras live at /dev/video* and arms at /dev/ttyACM* / by-id.")

    print("\n[1] Serial ports (SO-101 leader + follower arms)")
    ports = _serial_ports()
    if ports:
        for p in ports:
            print(f"    {p}")
        print("    -> set robot.port / teleop.port. Unplug one arm and re-run to tell them apart.")
    else:
        print("    none found — is the arm powered and its USB cable connected?")

    print("\n[2] AVFoundation cameras (stable Unique IDs)")
    cams = _mac_cameras()
    if cams:
        for c in cams:
            print(f"    {c['name']:<28} unique_id={c['unique_id']}")
        print("    -> prefer the unique_id for cameras.<name>.index_or_path when supported.")
    else:
        print("    none reported.")

    print("\n[3] OpenCV index probe (which indices actually open right now)")
    probed = _opencv_probe()
    if probed:
        for c in probed:
            print(f"    index {c['index']}  ({c['resolution']})")
        print("    -> if unique_id isn't accepted by your lerobot camera backend, use the")
        print("       index here, but re-verify it each session (indices are not stable).")
    else:
        print("    no indices opened (or cv2 missing).")

    print("\nTip: the built-in FaceTime camera usually claims index 0 — your external")
    print("wrist/overhead cams are the higher indices. Cover a lens and re-run to map them.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
