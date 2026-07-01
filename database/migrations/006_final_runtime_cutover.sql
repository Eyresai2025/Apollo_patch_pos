-- Apollo Tyre Inspection PostgreSQL Phase 5
-- Final runtime cutover tables for alarms, repeatability and hardware-test data.
-- MongoDB is no longer required by normal application startup/runtime.

CREATE TABLE IF NOT EXISTS {{schema}}.alarm_events (
    id                  UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    schema_version      VARCHAR(20) NOT NULL DEFAULT '5.0',
    fingerprint         VARCHAR(300) NOT NULL,
    code                VARCHAR(100) NOT NULL,
    component           VARCHAR(100) NOT NULL,
    severity            VARCHAR(20) NOT NULL,
    severity_rank       SMALLINT NOT NULL DEFAULT 3,
    title               TEXT NOT NULL,
    message             TEXT NOT NULL,
    recommended_action  TEXT,
    source              VARCHAR(100) NOT NULL DEFAULT 'SYSTEM_MONITOR',
    state               VARCHAR(30) NOT NULL DEFAULT 'ACTIVE',
    is_open             BOOLEAN NOT NULL DEFAULT TRUE,
    opened_at           TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    first_seen_at       TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    last_seen_at        TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    recovered_at        TIMESTAMPTZ,
    updated_at          TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    occurrence_count    INTEGER NOT NULL DEFAULT 1,
    cycle_id            VARCHAR(200),
    tyre_id             VARCHAR(200),
    sku_name            VARCHAR(200),
    zone                VARCHAR(50),
    acknowledgement     JSONB,
    recovery            JSONB,
    context             JSONB NOT NULL DEFAULT '{}'::jsonb,
    legacy_mongo_id     VARCHAR(80),
    CONSTRAINT ck_alarm_severity CHECK (severity IN ('CRITICAL', 'HIGH', 'WARNING', 'INFO')),
    CONSTRAINT ck_alarm_state CHECK (state IN ('ACTIVE', 'ACKNOWLEDGED', 'RECOVERED')),
    CONSTRAINT ck_alarm_occurrence_positive CHECK (occurrence_count >= 1)
);

CREATE UNIQUE INDEX IF NOT EXISTS uq_alarm_open_fingerprint
    ON {{schema}}.alarm_events (fingerprint)
    WHERE is_open = TRUE;
CREATE INDEX IF NOT EXISTS idx_alarm_open_severity_time
    ON {{schema}}.alarm_events (is_open DESC, severity_rank, opened_at DESC);
CREATE INDEX IF NOT EXISTS idx_alarm_component_code_time
    ON {{schema}}.alarm_events (component, code, opened_at DESC);
CREATE INDEX IF NOT EXISTS idx_alarm_cycle_time
    ON {{schema}}.alarm_events (cycle_id, opened_at DESC);
CREATE UNIQUE INDEX IF NOT EXISTS uq_alarm_legacy_mongo_id
    ON {{schema}}.alarm_events (legacy_mongo_id)
    WHERE legacy_mongo_id IS NOT NULL;

CREATE TABLE IF NOT EXISTS {{schema}}.repeatability_events (
    id                  UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    event_type          VARCHAR(100) NOT NULL,
    run_id              VARCHAR(200),
    cycle_no            INTEGER,
    target_cycles       INTEGER,
    folder_path         TEXT,
    operator_name       VARCHAR(150),
    images              JSONB NOT NULL DEFAULT '{}'::jsonb,
    event_document      JSONB NOT NULL DEFAULT '{}'::jsonb,
    legacy_mongo_id     VARCHAR(80),
    created_at          TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_repeatability_event_time
    ON {{schema}}.repeatability_events (event_type, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_repeatability_run_cycle
    ON {{schema}}.repeatability_events (run_id, cycle_no);
CREATE UNIQUE INDEX IF NOT EXISTS uq_repeatability_legacy_mongo_id
    ON {{schema}}.repeatability_events (legacy_mongo_id)
    WHERE legacy_mongo_id IS NOT NULL;

CREATE TABLE IF NOT EXISTS {{schema}}.test_mode_results (
    id                      UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    operator_name           VARCHAR(150),
    overall_ok              BOOLEAN NOT NULL DEFAULT FALSE,
    overall_status          VARCHAR(20) NOT NULL DEFAULT 'FAIL',
    deployment              VARCHAR(100),
    lights_ok               BOOLEAN NOT NULL DEFAULT FALSE,
    plc_ok                  BOOLEAN NOT NULL DEFAULT FALSE,
    camera_ok               BOOLEAN NOT NULL DEFAULT FALSE,
    laser_ok                BOOLEAN NOT NULL DEFAULT FALSE,
    app_ok_sent             BOOLEAN NOT NULL DEFAULT FALSE,
    connected_camera_count  INTEGER,
    total_camera_count      INTEGER,
    result_document         JSONB NOT NULL DEFAULT '{}'::jsonb,
    legacy_mongo_id         VARCHAR(80),
    created_at              TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    CONSTRAINT ck_test_mode_status CHECK (overall_status IN ('PASS', 'FAIL'))
);

CREATE INDEX IF NOT EXISTS idx_test_mode_created_status
    ON {{schema}}.test_mode_results (created_at DESC, overall_status);
CREATE UNIQUE INDEX IF NOT EXISTS uq_test_mode_legacy_mongo_id
    ON {{schema}}.test_mode_results (legacy_mongo_id)
    WHERE legacy_mongo_id IS NOT NULL;

INSERT INTO {{schema}}.application_settings (
    setting_key, setting_value, description, updated_at
) VALUES (
    'postgres_phase',
    '{"phase": 5, "status": "final_runtime_cutover_ready", "data_backend": "postgresql", "mongodb_required_at_runtime": false}'::jsonb,
    'Apollo PostgreSQL migration phase status',
    NOW()
)
ON CONFLICT (setting_key) DO UPDATE SET
    setting_value = EXCLUDED.setting_value,
    description = EXCLUDED.description,
    updated_at = NOW();
