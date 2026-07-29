from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

DLL_DIRS = [
    r"C:\Program Files\Teledyne DALSA\Sapera\Bin",
    r"C:\Program Files\Teledyne DALSA\Sapera\Components\NET\Bin",
    r"C:\Program Files\Teledyne DALSA\GenICam 3.20\bin\Win64_x64",
    r"C:\Program Files\Teledyne\Common Components\Bin",
    r"C:\Program Files\Teledyne\GigE Vision Interface\Bin",
]
SAPERA_DOTNET_DLL = (
    r"C:\Program Files\Teledyne DALSA\Sapera\Components\NET\Bin"
    r"\DALSA.SaperaLT.SapClassBasic.dll"
)

DEFAULT_EXPECTED_Y = {
    "M0006674": 140.0,
    "M0006994": 990.0,
}

CRITICAL_FEATURES = [
    "UserSetSelector",
    "profilesPerScan",
    "AcquisitionLineRate",
    "profileRate",
    "ExposureTime",
    "profileMedianFilterMode",
    "firSize",
    "laserPower",
    "peakDetectorReflectanceThreshold",
    "noiseReductionLevel",
    "aoiControlMode",
    "aoiZStart",
    "aoiHeight",
    "aoiNFOVStartX",
    "aoiNFOVWidth",
    "uniformXStepSize",
    "displacementY",
    "streamed_aoiZStart",
    "streamed_aoiHeight",
    "streamed_aoiNFOVStartX",
    "streamed_aoiNFOVWidth",
    "streamed_uniformXStepSize",
    "streamed_displacementY",
    "profilerRotationY",
    "profilerTranslationZ",
    "localRotationX",
    "localRotationZ",
    "umsTranslationX",
    "umsTranslationY",
    "umsTranslationZ",
    "umsRotationX",
    "umsRotationY",
    "umsRotationZ",
    "streamed_profilerRotationY",
    "streamed_profilerTranslationZ",
    "streamed_localRotationX",
    "streamed_localRotationZ",
    "streamed_umsTranslationX",
    "streamed_umsTranslationY",
    "streamed_umsTranslationZ",
    "streamed_umsRotationX",
    "streamed_umsRotationY",
    "streamed_umsRotationZ",
    "Scan3dDistanceUnit",
    "Scan3dCoordinateScale",
    "Scan3dCoordinateOffset",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Load a Z-Trak UserSet, read critical geometry/acquisition values, and save a report."
    )
    parser.add_argument("--serial", required=True, help="Laser serial, e.g. M0006674")
    parser.add_argument("--userset", default="UserSet1")
    parser.add_argument(
        "--expected-displacement-y",
        type=float,
        default=None,
        help="Expected displacementY in device units (normally micrometres).",
    )
    parser.add_argument(
        "--output-dir",
        default="media/Laser_Debug",
        help="Folder for JSON/TXT/CCF reports.",
    )
    return parser.parse_args()


def add_dll_dirs() -> None:
    for raw in DLL_DIRS:
        path = Path(raw)
        if path.exists():
            os.add_dll_directory(str(path))
            print(f"[DLL DIR ADDED] {path}")


def parse_ccf(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    if not path.exists():
        return values
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if "\t" in stripped:
            key, value = stripped.split("\t", 1)
        elif "=" in stripped:
            key, value = stripped.split("=", 1)
        else:
            continue
        values[key.strip()] = value.strip()
    return values


def normalize_number(value: Any) -> float | None:
    try:
        return float(str(value).strip())
    except Exception:
        return None


def main() -> int:
    args = parse_args()
    serial = args.serial.strip()
    userset = args.userset.strip()
    expected_y = (
        args.expected_displacement_y
        if args.expected_displacement_y is not None
        else DEFAULT_EXPECTED_Y.get(serial)
    )

    output_dir = Path(args.output_dir).expanduser().resolve() / serial
    output_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    ccf_path = output_dir / f"{serial}_{userset}_{stamp}_features.ccf"
    json_path = output_dir / f"{serial}_{userset}_{stamp}_readback.json"
    txt_path = output_dir / f"{serial}_{userset}_{stamp}_readback.txt"

    add_dll_dirs()
    from pythonnet import load

    load("netfx")
    import clr
    import System

    clr.AddReference(SAPERA_DOTNET_DLL)
    from DALSA.SaperaLT.SapClassBasic import (
        SapAcqDevice,
        SapLocation,
        SapManager,
        SapManagerBase,
    )

    def try_get_feature(acq_device: Any, feature_name: str) -> tuple[Any, str, bool]:
        attempts = [
            ("String", System.String),
            ("Double", System.Double),
            ("Single", System.Single),
            ("Int32", System.Int32),
            ("UInt32", System.UInt32),
            ("Int64", System.Int64),
            ("UInt64", System.UInt64),
            ("Boolean", System.Boolean),
        ]
        for type_name, dotnet_type in attempts:
            try:
                ref_val = clr.Reference[dotnet_type]()
                ok = acq_device.GetFeatureValue(feature_name, ref_val)
                if ok:
                    return ref_val.Value, type_name, True
            except Exception:
                continue
        return None, "Unreadable", False

    def execute_command(acq_device: Any, name: str) -> bool:
        for value in (True, 1, "Execute"):
            try:
                if not acq_device.IsFeatureAvailable(name):
                    return False
                ok = acq_device.SetFeatureValue(name, value)
                print(f"[COMMAND] {name} using {value!r} -> {ok}")
                if ok:
                    return True
            except Exception as exc:
                print(f"[WARN] {name}={value!r}: {exc}")
        return False

    print("\n" + "=" * 88)
    print("Z-TRAK USERSET READBACK")
    print(f"Serial                : {serial}")
    print(f"UserSet               : {userset}")
    print(f"Expected displacementY: {expected_y}")
    print("Mode                  : LOAD + READBACK ONLY (NO CAPTURE)")
    print("=" * 88)

    SapManager.DetectAllServers(SapManagerBase.DetectServerType.GenCP)
    SapManager.DetectAllServers(SapManagerBase.DetectServerType.All)

    selected: tuple[str, int] | None = None
    for server_idx in range(SapManager.GetServerCount()):
        if not SapManager.IsServerAccessible(server_idx):
            continue
        server_name = SapManager.GetServerName(server_idx)
        count = SapManager.GetResourceCount(server_idx, SapManagerBase.ResourceType.AcqDevice)
        for resource_idx in range(count):
            name = str(
                SapManager.GetResourceName(
                    server_idx, SapManagerBase.ResourceType.AcqDevice, resource_idx
                )
            ).strip()
            available = SapManager.IsResourceAvailable(
                server_idx, SapManagerBase.ResourceType.AcqDevice, resource_idx
            )
            print(f"[DISCOVERY] server={server_name} resource={resource_idx} serial={name} available={available}")
            if name == serial and available:
                selected = (server_name, resource_idx)
                break
        if selected:
            break

    if selected is None:
        raise RuntimeError(
            f"Laser {serial} is not available. Close Apollo, Z-Expert, CamExpert and other Sapera clients."
        )

    acq_device = SapAcqDevice(SapLocation(selected[0], selected[1]))
    report: dict[str, Any] = {
        "serial": serial,
        "userset": userset,
        "timestamp": stamp,
        "expected_displacement_y": expected_y,
        "userset_selector_ok": False,
        "userset_load_ok": False,
        "update_features_ok": False,
        "save_features_ok": False,
        "features": {},
        "verification": {},
        "overall_status": "FAIL",
    }

    try:
        if not acq_device.Create():
            raise RuntimeError("SapAcqDevice.Create() failed")
        print("[CREATE] SapAcqDevice -> True")

        report["userset_selector_ok"] = bool(
            acq_device.SetFeatureValue("UserSetSelector", userset)
        )
        print(f"[SET] UserSetSelector={userset} -> {report['userset_selector_ok']}")

        report["userset_load_ok"] = execute_command(acq_device, "UserSetLoad")
        report["update_features_ok"] = bool(acq_device.UpdateFeaturesFromDevice())
        print(f"[UPDATE FEATURES FROM DEVICE] -> {report['update_features_ok']}")

        try:
            report["save_features_ok"] = bool(acq_device.SaveFeatures(str(ccf_path)))
        except Exception as exc:
            report["save_features_error"] = str(exc)
        print(f"[SAVE FEATURES] {ccf_path} -> {report['save_features_ok']}")

        ccf_values = parse_ccf(ccf_path)
        for name in CRITICAL_FEATURES:
            available = False
            try:
                available = bool(acq_device.IsFeatureAvailable(name))
            except Exception:
                pass

            direct_value, direct_type, direct_ok = try_get_feature(acq_device, name)
            if direct_ok:
                value = str(direct_value)
                source = f"direct:{direct_type}"
            elif name in ccf_values:
                value = ccf_values[name]
                source = "ccf"
            else:
                value = None
                source = "unreadable"

            report["features"][name] = {
                "available": available,
                "value": value,
                "source": source,
            }
            print(f"[READBACK] {name:<38} value={value!s:<20} source={source} available={available}")

        streamed_y = report["features"].get("streamed_displacementY", {}).get("value")
        base_y = report["features"].get("displacementY", {}).get("value")
        actual_y = normalize_number(streamed_y if streamed_y is not None else base_y)
        y_ok = expected_y is None or (
            actual_y is not None and abs(actual_y - float(expected_y)) <= 1e-6
        )
        report["verification"]["displacement_y"] = {
            "expected": expected_y,
            "actual": actual_y,
            "source": "streamed_displacementY" if streamed_y is not None else "displacementY",
            "pass": y_ok,
        }

        x_step = report["features"].get("streamed_uniformXStepSize", {}).get("value")
        if x_step is None:
            x_step = report["features"].get("uniformXStepSize", {}).get("value")
        report["verification"]["x_step_present"] = {
            "actual": normalize_number(x_step),
            "pass": normalize_number(x_step) is not None,
        }

        report["overall_status"] = (
            "PASS"
            if report["userset_selector_ok"]
            and report["userset_load_ok"]
            and report["update_features_ok"]
            and y_ok
            else "FAIL"
        )

    finally:
        try:
            acq_device.Destroy()
            print("[CLEANUP] SapAcqDevice destroyed")
        except Exception as exc:
            print(f"[WARN] Destroy failed: {exc}")

    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    lines = [
        "Z-TRAK USERSET READBACK REPORT",
        "=" * 72,
        f"Serial: {serial}",
        f"UserSet: {userset}",
        f"Overall: {report['overall_status']}",
        f"Expected displacementY: {expected_y}",
        f"Actual displacementY: {report['verification']['displacement_y']['actual']}",
        f"X step: {report['verification']['x_step_present']['actual']}",
        "",
        "Critical readback:",
    ]
    for name, item in report["features"].items():
        lines.append(f"{name}={item['value']} | source={item['source']} | available={item['available']}")
    txt_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print("\n[REPORTS]")
    print(json_path)
    print(txt_path)
    print(ccf_path)
    print(f"\n[FINAL] {report['overall_status']}")
    return 0 if report["overall_status"] == "PASS" else 2


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"\n[FATAL] {exc}", file=sys.stderr)
        raise
