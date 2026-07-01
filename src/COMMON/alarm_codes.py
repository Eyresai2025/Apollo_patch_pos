from __future__ import annotations

"""Central alarm definitions for Apollo Tyre Inspection V5.

The definitions are deliberately data-only so the monitoring service, UI and
unit tests all use the same alarm code, severity and recommended action.
"""

from dataclasses import dataclass
from typing import Dict


SEVERITIES = ("CRITICAL", "HIGH", "WARNING", "INFO")
ALARM_STATES = ("ACTIVE", "ACKNOWLEDGED", "RECOVERED")


@dataclass(frozen=True)
class AlarmDefinition:
    health_key: str
    code: str
    component: str
    severity: str
    title: str
    recommended_action: str


HEALTH_ALARM_DEFINITIONS: Dict[str, AlarmDefinition] = {
    "plc": AlarmDefinition(
        health_key="plc",
        code="PLC-001",
        component="PLC",
        severity="CRITICAL",
        title="PLC communication unavailable",
        recommended_action=(
            "Check PLC power, Ethernet cable, configured IP/rack/slot and network reachability. "
            "Run the full hardware check after the connection is restored."
        ),
    ),
    "cameras": AlarmDefinition(
        health_key="cameras",
        code="CAM-001",
        component="CAMERAS",
        severity="CRITICAL",
        title="Camera array is not ready",
        recommended_action=(
            "Verify camera power and Ethernet links, confirm expected serial numbers and run the "
            "full hardware check. Do not start inspection until all required cameras are connected."
        ),
    ),
    "laser": AlarmDefinition(
        health_key="laser",
        code="LASER-001",
        component="LASER",
        severity="HIGH",
        title="Laser is unavailable",
        recommended_action=(
            "Check laser power, communication port and controller status. Confirm whether the laser "
            "is configured as mandatory for this deployment."
        ),
    ),
    "gpu": AlarmDefinition(
        health_key="gpu",
        code="GPU-001",
        component="GPU",
        severity="HIGH",
        title="GPU/CUDA is unavailable",
        recommended_action=(
            "Check NVIDIA driver, CUDA runtime and GPU visibility. Restart the application only after "
            "confirming that torch.cuda.is_available() returns True."
        ),
    ),
    "storage": AlarmDefinition(
        health_key="storage",
        code="STORAGE-001",
        component="STORAGE",
        severity="WARNING",
        title="Storage free space is below the configured limit",
        recommended_action=(
            "Archive or remove old captures and reports, then confirm that free disk space is above "
            "STORAGE_MIN_FREE_GB before continuing long inspection runs."
        ),
    ),
    "app_ok": AlarmDefinition(
        health_key="app_ok",
        code="APP-OK-001",
        component="APPLICATION",
        severity="CRITICAL",
        title="Application OK handshake is not verified",
        recommended_action=(
            "Check the configured PLC application-ready bit and confirm the PLC read-back value. "
            "Run the full hardware check and verify the handshake before automatic inspection."
        ),
    ),
    "inspection_sync": AlarmDefinition(
        health_key="inspection_sync",
        code="DB-OUTBOX-001",
        component="DATABASE",
        severity="WARNING",
        title="Inspection records are waiting in the offline outbox",
        recommended_action=(
            "Check MongoDB/network availability. Keep Apollo running so the background recovery "
            "service can synchronize pending inspection records automatically."
        ),
    ),
}


def definition_for_health_key(health_key: str) -> AlarmDefinition | None:
    return HEALTH_ALARM_DEFINITIONS.get(str(health_key or "").strip().lower())
