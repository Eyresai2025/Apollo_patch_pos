-- Apollo VIT PostgreSQL Phase 3: inspection-cycle metadata and traceability.
-- Actual inspection image binaries remain in the existing MongoDB GridFS
-- buckets during this phase. The GridFS references are retained inside the
-- inspection_document JSONB payload.
-- Do not edit this migration after it has been applied.

CREATE TABLE IF NOT EXISTS {{schema}}.inspection_cycles (
    id                       UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    cycle_uid                VARCHAR(300) NOT NULL UNIQUE,
    cycle_id                 VARCHAR(150) NOT NULL,
    cycle_no                 VARCHAR(100),
    sku_id                   UUID REFERENCES {{schema}}.skus(id)
                                 ON UPDATE CASCADE ON DELETE SET NULL,
    sku_name                 VARCHAR(150),
    tyre_name                VARCHAR(200),
    inspection_datetime      TIMESTAMPTZ NOT NULL,
    inspection_date          DATE NOT NULL,
    operator_username        VARCHAR(150),
    operator_full_name       VARCHAR(200),
    operator_role            VARCHAR(100),
    final_result             VARCHAR(30) NOT NULL DEFAULT 'UNKNOWN',
    total_defect_count       INTEGER NOT NULL DEFAULT 0,
    cycle_time_ms            DOUBLE PRECISION,
    plc_sent                 BOOLEAN NOT NULL DEFAULT FALSE,
    plc_display              VARCHAR(150),
    lifecycle_status         VARCHAR(40) NOT NULL DEFAULT 'AI_COMPLETED',
    schema_version           VARCHAR(30),
    storage_status           VARCHAR(40) NOT NULL DEFAULT 'POSTGRESQL',
    offline_recovered        BOOLEAN NOT NULL DEFAULT FALSE,
    gridfs_linked            BOOLEAN NOT NULL DEFAULT FALSE,
    gridfs_input_count       INTEGER NOT NULL DEFAULT 0,
    gridfs_output_count      INTEGER NOT NULL DEFAULT 0,
    gridfs_failed_count      INTEGER NOT NULL DEFAULT 0,
    document_revision        INTEGER NOT NULL DEFAULT 1,
    inspection_document      JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at               TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at               TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    CONSTRAINT ck_inspection_cycles_defects_nonnegative
        CHECK (total_defect_count >= 0),
    CONSTRAINT ck_inspection_cycles_gridfs_counts_nonnegative
        CHECK (
            gridfs_input_count >= 0
            AND gridfs_output_count >= 0
            AND gridfs_failed_count >= 0
        ),
    CONSTRAINT ck_inspection_cycles_revision_positive
        CHECK (document_revision > 0),
    CONSTRAINT ck_inspection_cycles_document_object
        CHECK (jsonb_typeof(inspection_document) = 'object')
);

CREATE INDEX IF NOT EXISTS idx_inspection_cycles_datetime
    ON {{schema}}.inspection_cycles (inspection_datetime DESC);

CREATE INDEX IF NOT EXISTS idx_inspection_cycles_date
    ON {{schema}}.inspection_cycles (inspection_date DESC);

CREATE INDEX IF NOT EXISTS idx_inspection_cycles_sku_datetime
    ON {{schema}}.inspection_cycles (sku_name, inspection_datetime DESC);

CREATE INDEX IF NOT EXISTS idx_inspection_cycles_result_datetime
    ON {{schema}}.inspection_cycles (final_result, inspection_datetime DESC);

CREATE INDEX IF NOT EXISTS idx_inspection_cycles_lifecycle
    ON {{schema}}.inspection_cycles (lifecycle_status, inspection_datetime DESC);

CREATE INDEX IF NOT EXISTS idx_inspection_cycles_operator
    ON {{schema}}.inspection_cycles (operator_username, inspection_datetime DESC);

CREATE INDEX IF NOT EXISTS idx_inspection_cycles_document_gin
    ON {{schema}}.inspection_cycles USING GIN (inspection_document);

DROP TRIGGER IF EXISTS trg_inspection_cycles_updated_at
    ON {{schema}}.inspection_cycles;
CREATE TRIGGER trg_inspection_cycles_updated_at
BEFORE UPDATE ON {{schema}}.inspection_cycles
FOR EACH ROW EXECUTE FUNCTION {{schema}}.set_updated_at();

CREATE TABLE IF NOT EXISTS {{schema}}.inspection_cycle_events (
    id                       BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    inspection_cycle_id      UUID NOT NULL REFERENCES {{schema}}.inspection_cycles(id)
                                 ON UPDATE CASCADE ON DELETE CASCADE,
    cycle_uid                VARCHAR(300) NOT NULL,
    event_type               VARCHAR(80) NOT NULL,
    event_status             VARCHAR(40) NOT NULL,
    lifecycle_status         VARCHAR(40),
    event_data               JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at               TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    CONSTRAINT ck_inspection_cycle_events_data_object
        CHECK (jsonb_typeof(event_data) = 'object')
);

CREATE INDEX IF NOT EXISTS idx_inspection_cycle_events_cycle_created
    ON {{schema}}.inspection_cycle_events (inspection_cycle_id, created_at DESC);

CREATE INDEX IF NOT EXISTS idx_inspection_cycle_events_uid_created
    ON {{schema}}.inspection_cycle_events (cycle_uid, created_at DESC);

CREATE INDEX IF NOT EXISTS idx_inspection_cycle_events_type_status
    ON {{schema}}.inspection_cycle_events (event_type, event_status, created_at DESC);

INSERT INTO {{schema}}.application_settings (
    setting_key,
    setting_value,
    description
)
VALUES (
    'postgres_phase',
    '{"phase": 3, "status": "inspection_metadata_ready", "image_backend": "mongodb_gridfs"}'::jsonb,
    'Tracks the active PostgreSQL migration phase for Apollo VIT.'
)
ON CONFLICT (setting_key) DO UPDATE SET
    setting_value = EXCLUDED.setting_value,
    description = EXCLUDED.description,
    updated_at = NOW();
