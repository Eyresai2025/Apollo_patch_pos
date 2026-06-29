from __future__ import annotations

"""Alarm lifecycle service for Apollo VIT V5.

The service converts lightweight component-health snapshots into deduplicated
PostgreSQL alarms. It deliberately does no camera/PLC reconnection and performs no
AI loading, so callers can run it from a background executor during inspection.
"""

import hashlib
import threading
from typing import Any, Dict, Mapping, Optional

from src.COMMON.alarm_codes import HEALTH_ALARM_DEFINITIONS, AlarmDefinition
from src.COMMON.alarm_repository import AlarmRepository
from src.COMMON.structured_logging import get_logger

logger = get_logger(__name__, component="ALARM_SERVICE")


class AlarmService:
    def __init__(
        self,
        repository: AlarmRepository,
        *,
        failure_confirmations: int = 2,
        recovery_confirmations: int = 1,
    ) -> None:
        self.repository = repository
        self.failure_confirmations = max(1, int(failure_confirmations))
        self.recovery_confirmations = max(1, int(recovery_confirmations))
        self._failure_counts: Dict[str, int] = {}
        self._recovery_counts: Dict[str, int] = {}
        self._lock = threading.RLock()
        self._indexes_ready = False

    def ensure_indexes(self):
        with self._lock:
            if self._indexes_ready:
                return []
            names = self.repository.ensure_indexes()
            self._indexes_ready = True
            return names

    @staticmethod
    def fingerprint(code: str, component: str, source: str = "SYSTEM_MONITOR") -> str:
        payload = f"{source}|{component}|{code}".strip().upper().encode("utf-8")
        digest = hashlib.sha1(payload).hexdigest()[:16]
        return f"{source}:{component}:{code}:{digest}".upper()

    def process_health_snapshot(
        self,
        health: Mapping[str, Any],
        *,
        context: Optional[Mapping[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Create/update/recover alarms from one non-invasive health snapshot."""
        self.ensure_indexes()
        context_dict = dict(context or {})
        items = dict(health.get("items") or {})
        created = updated = recovered = 0
        affected = []

        for health_key, definition in HEALTH_ALARM_DEFINITIONS.items():
            if health_key not in items:
                continue
            item = dict(items.get(health_key) or {})

            # A component can be unhealthy-looking while it is still merely
            # uninitialized (for example, "Not checked" immediately after app
            # launch). Such states must not create nuisance startup alarms.
            if not self._is_alarm_eligible(item):
                with self._lock:
                    self._failure_counts[health_key] = 0
                    self._recovery_counts[health_key] = 0
                continue

            is_ok = bool(item.get("ok", False))
            fingerprint = self.fingerprint(definition.code, definition.component)

            with self._lock:
                if is_ok:
                    self._failure_counts[health_key] = 0
                    self._recovery_counts[health_key] = self._recovery_counts.get(health_key, 0) + 1
                    # Process only the transition confirmation. Later identical
                    # OK snapshots do not keep writing the recovered record.
                    if self._recovery_counts[health_key] != self.recovery_confirmations:
                        continue
                else:
                    self._recovery_counts[health_key] = 0
                    self._failure_counts[health_key] = self._failure_counts.get(health_key, 0) + 1
                    # Process only the first confirmed failure. This avoids one
                    # PostgreSQL write and occurrence increment on every health poll.
                    if self._failure_counts[health_key] != self.failure_confirmations:
                        continue

            if is_ok:
                document = self.repository.recover_by_fingerprint(
                    fingerprint,
                    message=f"{definition.component} health returned to OK: {item.get('text', 'OK')}",
                    context={"health_item": item, **context_dict},
                )
                if document:
                    recovered += 1
                    affected.append(document)
                    logger.info(
                        f"Alarm recovered: {definition.code} {definition.component}",
                        extra={
                            "event_code": "ALARM_RECOVERED",
                            "error_code": definition.code,
                            "status": "RECOVERED",
                            "details": {"fingerprint": fingerprint, "health": item},
                        },
                    )
                continue

            document = self.repository.open_or_update(
                self._alarm_payload(definition, item, context_dict)
            )
            if document.get("created"):
                created += 1
                event_code = "ALARM_OPENED"
            else:
                updated += 1
                event_code = "ALARM_REOCCURRED"
            affected.append(document)
            logger.warning(
                f"{definition.code} {definition.title}: {item.get('text', '-')}",
                extra={
                    "event_code": event_code,
                    "error_code": definition.code,
                    "status": document.get("state", "ACTIVE"),
                    "cycle_id": context_dict.get("cycle_id", "-"),
                    "tyre_id": context_dict.get("tyre_id", "-"),
                    "sku_name": context_dict.get("sku_name", "-"),
                    "details": {"fingerprint": fingerprint, "health": item},
                },
            )

        summary = self.repository.summary()
        return {
            "created": created,
            "updated": updated,
            "recovered": recovered,
            "affected": affected,
            "summary": summary,
        }

    def raise_alarm(
        self,
        *,
        code: str,
        component: str,
        severity: str,
        title: str,
        message: str,
        recommended_action: str,
        source: str = "APPLICATION",
        context: Optional[Mapping[str, Any]] = None,
        cycle_id: str = "-",
        tyre_id: str = "-",
        sku_name: str = "-",
        zone: str = "-",
    ) -> Dict[str, Any]:
        self.ensure_indexes()
        payload = {
            "fingerprint": self.fingerprint(code, component, source),
            "code": code,
            "component": component,
            "severity": severity,
            "title": title,
            "message": message,
            "recommended_action": recommended_action,
            "source": source,
            "context": dict(context or {}),
            "cycle_id": cycle_id,
            "tyre_id": tyre_id,
            "sku_name": sku_name,
            "zone": zone,
        }
        document = self.repository.open_or_update(payload)
        logger.warning(
            f"Manual/application alarm recorded: {code} {message}",
            extra={
                "event_code": "ALARM_OPENED" if document.get("created") else "ALARM_REOCCURRED",
                "error_code": code,
                "status": document.get("state", "ACTIVE"),
                "cycle_id": cycle_id,
                "tyre_id": tyre_id,
                "sku_name": sku_name,
                "zone": zone,
                "details": dict(context or {}),
            },
        )
        return document

    def recover_alarm(self, *, code: str, component: str, source: str = "APPLICATION"):
        self.ensure_indexes()
        return self.repository.recover_by_fingerprint(
            self.fingerprint(code, component, source),
            message="Alarm recovered by application event",
        )

    def acknowledge(self, alarm_id: Any, *, user: Mapping[str, Any], note: str = ""):
        document = self.repository.acknowledge(alarm_id, user=user, note=note)
        if document:
            logger.info(
                f"Alarm acknowledged: {document.get('code', '-')}",
                extra={
                    "event_code": "ALARM_ACKNOWLEDGED",
                    "error_code": document.get("code", "-"),
                    "status": "ACKNOWLEDGED",
                    "details": {"alarm_id": str(document.get("_id")), "user": dict(user)},
                },
            )
        return document

    def manual_clear(self, alarm_id: Any, *, user: Mapping[str, Any], note: str):
        document = self.repository.recover_by_id(alarm_id, user=user, note=note)
        if document:
            logger.info(
                f"Alarm manually cleared: {document.get('code', '-')}",
                extra={
                    "event_code": "ALARM_MANUALLY_CLEARED",
                    "error_code": document.get("code", "-"),
                    "status": "RECOVERED",
                    "details": {"alarm_id": str(document.get("_id")), "user": dict(user)},
                },
            )
        return document

    def list_alarms(self, filters=None, *, page: int = 1, page_size: int = 25):
        self.ensure_indexes()
        payload = self.repository.list_alarms(filters, page=page, page_size=page_size)
        summary_filters = dict(filters or {})
        # Summary cards represent the selected search/component/severity scope,
        # but are not collapsed by the table's state selector.
        summary_filters.pop("state", None)
        payload["summary"] = self.repository.summary(summary_filters)
        payload["filter_options"] = self.repository.filter_options()
        return payload

    def get_alarm(self, alarm_id: Any):
        return self.repository.get_by_id(alarm_id)

    def summary(self, filters=None):
        self.ensure_indexes()
        return self.repository.summary(filters)

    @staticmethod
    def _is_alarm_eligible(item: Mapping[str, Any]) -> bool:
        """Return False for startup/unknown states that are not real failures."""
        explicit = item.get("alarm_eligible")
        if explicit is not None:
            return bool(explicit)

        # Backward-compatible protection when an older health-service file is
        # accidentally used with this hotfix.
        text = str(item.get("text") or "").strip().lower()
        startup_placeholders = {
            "not checked",
            "demo not checked",
            "not verified",
            "demo not verified",
            "unknown",
            "mode: unknown",
        }
        return text not in startup_placeholders

    def _alarm_payload(
        self,
        definition: AlarmDefinition,
        item: Mapping[str, Any],
        context: Mapping[str, Any],
    ) -> Dict[str, Any]:
        return {
            "fingerprint": self.fingerprint(definition.code, definition.component),
            "code": definition.code,
            "component": definition.component,
            "severity": definition.severity,
            "title": definition.title,
            "message": str(item.get("text") or "Component health check failed"),
            "recommended_action": definition.recommended_action,
            "source": "SYSTEM_MONITOR",
            "context": {"health_item": dict(item), **dict(context)},
            "cycle_id": context.get("cycle_id", "-"),
            "tyre_id": context.get("tyre_id", "-"),
            "sku_name": context.get("sku_name", "-"),
            "zone": context.get("zone", "-"),
        }
