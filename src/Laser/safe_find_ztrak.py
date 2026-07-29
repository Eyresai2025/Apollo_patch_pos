"""Safe, non-capturing Teledyne DALSA Z-Trak connection check.

This module is used by Apollo Test Mode.  It intentionally does only:

1. Load the Sapera .NET SDK.
2. Detect accessible AcqDevice resources.
3. Optionally open and immediately close the requested laser resources.

It does not apply features, turn the laser on, start acquisition, allocate
capture buffers, or save files.  The same discovery/open pattern is used by the
production laser capture runner, but is kept here as an import-safe helper.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, Iterable, List, Optional


DEFAULT_DLL_DIRS = [
    r"C:\Program Files\Teledyne DALSA\Sapera\Bin",
    r"C:\Program Files\Teledyne DALSA\Sapera\Components\NET\Bin",
    r"C:\Program Files\Teledyne DALSA\GenICam 3.20\bin\Win64_x64",
    r"C:\Program Files\Teledyne\Common Components\Bin",
    r"C:\Program Files\Teledyne\GigE Vision Interface\Bin",
]

DEFAULT_SAPERA_DOTNET_DLL = (
    r"C:\Program Files\Teledyne DALSA\Sapera\Components\NET\Bin"
    r"\DALSA.SaperaLT.SapClassBasic.dll"
)


def _normalise_targets(target_serials: Optional[Iterable[str]]) -> List[str]:
    if target_serials is None:
        return []
    result: List[str] = []
    for value in target_serials:
        serial = str(value or "").strip()
        if serial and serial not in result:
            result.append(serial)
    return result


def _load_sapera(sapera_dll: str = ""):
    """Load Sapera lazily so importing Apollo does not require the SDK."""
    dll_path = str(sapera_dll or DEFAULT_SAPERA_DOTNET_DLL).strip()
    if not dll_path:
        dll_path = DEFAULT_SAPERA_DOTNET_DLL

    added_dirs: List[str] = []
    for directory in DEFAULT_DLL_DIRS:
        try:
            if Path(directory).exists():
                os.add_dll_directory(directory)
                added_dirs.append(directory)
        except (AttributeError, FileNotFoundError, OSError):
            # add_dll_directory is Windows-only and can reject duplicate paths.
            pass

    if not Path(dll_path).exists():
        raise FileNotFoundError(f"Sapera .NET DLL not found: {dll_path}")

    from pythonnet import load

    try:
        load("netfx")
    except RuntimeError as exc:
        # pythonnet raises when the runtime is already loaded.  That is safe.
        if "already" not in str(exc).lower():
            raise

    import clr

    clr.AddReference(dll_path)

    from DALSA.SaperaLT.SapClassBasic import (
        SapAcqDevice,
        SapLocation,
        SapManager,
        SapManagerBase,
    )

    return {
        "SapAcqDevice": SapAcqDevice,
        "SapLocation": SapLocation,
        "SapManager": SapManager,
        "SapManagerBase": SapManagerBase,
        "dll_path": dll_path,
        "dll_dirs": added_dirs,
    }


def check_ztrak_connections(
    target_serials: Optional[Iterable[str]] = None,
    *,
    sapera_dll: str = "",
    open_device: bool = True,
) -> Dict[str, object]:
    """Discover requested Z-Trak resources and optionally open/close each one.

    Parameters
    ----------
    target_serials:
        Serial/resource names to require.  When empty, at least one accessible
        AcqDevice must be found.
    sapera_dll:
        Optional absolute Sapera .NET DLL path.
    open_device:
        When True, create and destroy a SapAcqDevice for each selected resource.
        No features are written and no acquisition is started.
    """
    targets = _normalise_targets(target_serials)
    details: Dict[str, object] = {
        "ok": False,
        "targets": targets,
        "detected": [],
        "missing": [],
        "devices": [],
        "sapera_dll": "",
        "dll_dirs": [],
        "server_count": 0,
        "message": "",
    }

    opened_devices = []
    try:
        sdk = _load_sapera(sapera_dll)
        SapManager = sdk["SapManager"]
        SapManagerBase = sdk["SapManagerBase"]
        SapLocation = sdk["SapLocation"]
        SapAcqDevice = sdk["SapAcqDevice"]
        details["sapera_dll"] = sdk["dll_path"]
        details["dll_dirs"] = sdk["dll_dirs"]

        # Use the same detection calls as ztrak_capture_multi_laser.py.
        detect_messages = []
        for detect_name in ("GenCP", "All"):
            try:
                detect_value = getattr(SapManagerBase.DetectServerType, detect_name)
                detect_result = SapManager.DetectAllServers(detect_value)
                detect_messages.append(f"{detect_name}={detect_result}")
            except Exception as exc:
                detect_messages.append(f"{detect_name}=ERROR({exc})")
        details["detect_messages"] = detect_messages

        server_count = int(SapManager.GetServerCount())
        details["server_count"] = server_count
        resources: List[Dict[str, object]] = []

        for server_index in range(server_count):
            try:
                server_name = str(SapManager.GetServerName(server_index))
                server_type = str(SapManager.GetServerType(server_index))
                accessible = bool(SapManager.IsServerAccessible(server_index))
                if not accessible or server_name.strip().lower() == "system":
                    continue

                resource_count = int(
                    SapManager.GetResourceCount(
                        server_index,
                        SapManagerBase.ResourceType.AcqDevice,
                    )
                )
                for resource_index in range(resource_count):
                    resource_name = str(
                        SapManager.GetResourceName(
                            server_index,
                            SapManagerBase.ResourceType.AcqDevice,
                            resource_index,
                        )
                    ).strip()
                    available = bool(
                        SapManager.IsResourceAvailable(
                            server_index,
                            SapManagerBase.ResourceType.AcqDevice,
                            resource_index,
                        )
                    )
                    resources.append(
                        {
                            "serial": resource_name,
                            "server_index": server_index,
                            "server_name": server_name,
                            "server_type": server_type,
                            "resource_index": resource_index,
                            "resource_available": available,
                            "opened": False,
                            "message": "Available" if available else "Resource unavailable",
                        }
                    )
            except Exception as exc:
                resources.append(
                    {
                        "serial": "",
                        "server_index": server_index,
                        "server_name": "",
                        "server_type": "",
                        "resource_index": -1,
                        "resource_available": False,
                        "opened": False,
                        "message": f"Server scan failed: {exc}",
                    }
                )

        detected = [
            str(item.get("serial", ""))
            for item in resources
            if item.get("serial") and item.get("resource_available")
        ]
        details["detected"] = detected

        if targets:
            selected = [item for item in resources if item.get("serial") in targets]
            missing = [serial for serial in targets if serial not in detected]
        else:
            selected = [item for item in resources if item.get("resource_available")]
            missing = []
        details["missing"] = missing

        if open_device:
            for item in selected:
                if not item.get("resource_available"):
                    continue
                acq_device = None
                try:
                    location = SapLocation(
                        str(item["server_name"]),
                        int(item["resource_index"]),
                    )
                    try:
                        acq_device = SapAcqDevice(location)
                    except Exception:
                        acq_device = SapAcqDevice(location, "")
                    created = bool(acq_device.Create())
                    item["opened"] = created
                    item["message"] = (
                        "Connected: SapAcqDevice opened and closed"
                        if created
                        else "SapAcqDevice.Create() returned False"
                    )
                    if created:
                        opened_devices.append(acq_device)
                except Exception as exc:
                    item["opened"] = False
                    item["message"] = f"Open failed: {exc}"
                finally:
                    if acq_device is not None:
                        try:
                            acq_device.Destroy()
                        except Exception:
                            pass
                        if acq_device in opened_devices:
                            opened_devices.remove(acq_device)
        else:
            for item in selected:
                item["opened"] = bool(item.get("resource_available"))
                item["message"] = "Accessible resource detected"

        details["devices"] = selected
        if targets:
            selected_by_serial = {
                str(item.get("serial", "")): item for item in selected
            }
            ok = not missing and all(
                bool(selected_by_serial.get(serial, {}).get("opened", False))
                for serial in targets
            )
        else:
            ok = any(bool(item.get("opened", False)) for item in selected)

        details["ok"] = ok
        if ok:
            connected_names = targets or [
                str(item.get("serial", ""))
                for item in selected
                if item.get("opened")
            ]
            details["message"] = (
                "Laser connection check passed: " + ", ".join(connected_names)
            )
        elif missing:
            details["message"] = (
                "Requested laser serial(s) not detected: " + ", ".join(missing)
            )
        elif not selected:
            details["message"] = "No accessible Sapera AcqDevice laser resources found."
        else:
            failed = [
                f"{item.get('serial')}: {item.get('message')}"
                for item in selected
                if not item.get("opened")
            ]
            details["message"] = "Laser open check failed. " + "; ".join(failed)
        return details

    except Exception as exc:
        details["ok"] = False
        details["message"] = f"Sapera laser connection check error: {exc}"
        details["error"] = str(exc)
        return details
    finally:
        for device in list(opened_devices):
            try:
                device.Destroy()
            except Exception:
                pass


def _print_result(result: Dict[str, object]) -> None:
    print("\n" + "=" * 78)
    print("SAFE Z-TRAK CONNECTION CHECK")
    print("=" * 78)
    print("Sapera DLL :", result.get("sapera_dll", "-"))
    print("Targets    :", result.get("targets", []))
    print("Detected   :", result.get("detected", []))
    print("Missing    :", result.get("missing", []))
    print("Result     :", "PASS" if result.get("ok") else "FAIL")
    print("Message    :", result.get("message", "-"))
    for item in result.get("devices", []) or []:
        print(
            f"  {item.get('serial', '-')} | server={item.get('server_name', '-')} "
            f"| resource={item.get('resource_index', '-')} | opened={item.get('opened')} "
            f"| {item.get('message', '-')}"
        )
    print("=" * 78)


if __name__ == "__main__":
    target_text = os.environ.get("LASER_CONNECTION_TARGET_SERIALS", "").strip()
    target_list = [
        part.strip()
        for part in target_text.replace(";", ",").split(",")
        if part.strip()
    ]
    result = check_ztrak_connections(target_list)
    _print_result(result)
    raise SystemExit(0 if result.get("ok") else 1)
