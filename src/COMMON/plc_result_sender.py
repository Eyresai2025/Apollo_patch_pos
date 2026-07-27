# src/COMMON/plc_result_sender.py

from pathlib import Path
import time
from typing import Any, Dict, Optional


def _project_root() -> Path:
    try:
        return Path(__file__).resolve().parents[2]
    except Exception:
        return Path.cwd()


def _load_env(env_path=None) -> Dict[str, str]:
    env_file = Path(env_path) if env_path else (_project_root() / ".env")
    data: Dict[str, str] = {}

    try:
        if env_file.exists():
            with env_file.open("r", encoding="utf-8") as file:
                for line in file:
                    line = line.strip()
                    if not line or line.startswith("#") or "=" not in line:
                        continue
                    key, value = line.split("=", 1)
                    data[key.strip()] = value.strip().strip('"').strip("'")
    except Exception:
        pass

    return data


def _env_int(env: Dict[str, str], key: str, default: int) -> int:
    try:
        value = env.get(key, "")
        if value is None or str(value).strip() == "":
            return int(default)
        return int(float(str(value).strip()))
    except Exception:
        return int(default)


def _connect_plc(env: Dict[str, str]):
    try:
        import snap7
    except Exception as exc:
        raise RuntimeError(f"snap7 import failed: {exc}") from exc

    plc_ip = env.get("PLC_IP", "192.168.10.1")
    rack = _env_int(env, "PLC_RACK", 0)
    slot = _env_int(env, "PLC_SLOT", 1)

    client = snap7.client.Client()
    client.connect(plc_ip, rack, slot)

    if hasattr(client, "get_connected") and not client.get_connected():
        raise RuntimeError(f"PLC connect failed: {plc_ip}")

    return client


def _read_bit(client, db: int, byte_offset: int, bit: int) -> Optional[bool]:
    data = client.db_read(db, byte_offset, 1)
    if not data:
        return None
    return bool(data[0] & (1 << bit))


def _write_bit(
    client,
    db: int,
    byte_offset: int,
    bit: int,
    value: bool,
) -> Optional[bool]:
    data = bytearray(client.db_read(db, byte_offset, 1))
    if not data:
        data = bytearray([0])

    if value:
        data[0] |= 1 << bit
    else:
        data[0] &= ~(1 << bit)

    client.db_write(db, byte_offset, data)
    return _read_bit(client, db, byte_offset, bit)


def _send_result(
    final_result: str,
    env_path=None,
    require_deployment: bool = True,
) -> Dict[str, Any]:
    """Send and verify one ACCEPT or REJECT pulse using Apollo's PLC settings."""
    env = _load_env(env_path)

    if require_deployment and str(env.get("DEPLOYMENT", "False")).strip() != "True":
        return {
            "sent": False,
            "display": "Demo - Not Sent",
            "detail": "DEPLOYMENT=False",
        }

    final_result = str(final_result or "").strip().upper()

    accept_values = {"OK", "PASS", "GOOD", "ACCEPT"}
    reject_values = {
        "NG", "DEFECT", "SUSPECT", "INVALID",
        "FAILED", "FAIL", "REJECT",
    }

    if final_result in accept_values:
        decision = "ACCEPT"
    elif final_result in reject_values:
        decision = "REJECT"
    else:
        return {
            "sent": False,
            "display": "Result Not Mapped",
            "detail": f"Unknown final_result={final_result}",
        }

    accept_db = _env_int(env, "PLC_ACCEPT_DB", 74)
    accept_byte = _env_int(env, "PLC_ACCEPT_BYTE", 0)
    accept_bit = _env_int(env, "PLC_ACCEPT_BIT", 1)

    reject_db = _env_int(env, "PLC_REJECT_DB", 74)
    reject_byte = _env_int(env, "PLC_REJECT_BYTE", 0)
    reject_bit = _env_int(env, "PLC_REJECT_BIT", 2)

    pulse_ms = max(0, _env_int(env, "PLC_RESULT_PULSE_MS", 300))
    plc_ip = env.get("PLC_IP", "192.168.10.1")

    target = (
        (accept_db, accept_byte, accept_bit)
        if decision == "ACCEPT"
        else (reject_db, reject_byte, reject_bit)
    )
    target_db, target_byte, target_bit = target

    client = None
    try:
        client = _connect_plc(env)

        # Start every test/result from a known state.
        accept_clear = _write_bit(
            client, accept_db, accept_byte, accept_bit, False
        )
        reject_clear = _write_bit(
            client, reject_db, reject_byte, reject_bit, False
        )
        time.sleep(0.05)

        pre_accept = _read_bit(client, accept_db, accept_byte, accept_bit)
        pre_reject = _read_bit(client, reject_db, reject_byte, reject_bit)

        set_readback = _write_bit(
            client, target_db, target_byte, target_bit, True
        )

        if pulse_ms > 0:
            time.sleep(pulse_ms / 1000.0)

        clear_readback = _write_bit(
            client, target_db, target_byte, target_bit, False
        )

        post_accept = _read_bit(client, accept_db, accept_byte, accept_bit)
        post_reject = _read_bit(client, reject_db, reject_byte, reject_bit)

        set_verified = set_readback is True
        clear_verified = clear_readback is False
        both_low_after = post_accept is False and post_reject is False
        sent = bool(set_verified and clear_verified and both_low_after)

        return {
            "sent": sent,
            "decision": decision,
            "display": (
                f"{decision} Pulse Verified"
                if sent
                else f"{decision} Verification Failed"
            ),
            "detail": (
                f"PLC={plc_ip} | target=DB{target_db}.DBX{target_byte}.{target_bit} | "
                f"pulse_ms={pulse_ms} | "
                f"initial_clear_accept={accept_clear} | "
                f"initial_clear_reject={reject_clear} | "
                f"pre_accept={pre_accept} | pre_reject={pre_reject} | "
                f"set_readback={set_readback} | "
                f"clear_readback={clear_readback} | "
                f"post_accept={post_accept} | post_reject={post_reject}"
            ),
            "plc_ip": plc_ip,
            "db": target_db,
            "byte": target_byte,
            "bit": target_bit,
            "pulse_ms": pulse_ms,
            "set_readback": set_readback,
            "clear_readback": clear_readback,
            "post_accept": post_accept,
            "post_reject": post_reject,
        }

    except Exception as exc:
        return {
            "sent": False,
            "decision": decision,
            "display": "PLC Send Failed",
            "detail": str(exc),
            "plc_ip": plc_ip,
            "db": target_db,
            "byte": target_byte,
            "bit": target_bit,
            "pulse_ms": pulse_ms,
        }

    finally:
        try:
            if client is not None:
                client.disconnect()
        except Exception:
            pass


def send_tyre_result_to_plc(final_result, env_path=None) -> Dict[str, Any]:
    """Production result path. DEPLOYMENT=True is required."""
    return _send_result(
        final_result=final_result,
        env_path=env_path,
        require_deployment=True,
    )


def test_plc_result_bit(decision: str, env_path=None) -> Dict[str, Any]:
    """Integrated Test Mode pulse.

    This uses the same Apollo PLC result implementation and .env configuration
    as Live inference, but it is allowed from Test Mode even when DEPLOYMENT is
    temporarily False.
    """
    normalized = str(decision or "").strip().upper()
    if normalized not in {"ACCEPT", "REJECT"}:
        raise ValueError("decision must be ACCEPT or REJECT")

    return _send_result(
        final_result=normalized,
        env_path=env_path,
        require_deployment=False,
    )
