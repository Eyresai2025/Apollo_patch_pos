-- Apollo Tyre Inspection PostgreSQL Phase 4B
-- Action/OSC catalog metadata + images and AI model registry + binaries.
-- Existing MongoDB collections and GridFS files remain untouched as rollback/fallback sources.
-- Do not edit this migration after it has been applied.

CREATE TABLE IF NOT EXISTS {{schema}}.action_catalog_versions (
    version_id          VARCHAR(180) PRIMARY KEY,
    revision_no         VARCHAR(50) NOT NULL,
    local_version_no    VARCHAR(80) NOT NULL DEFAULT '00',
    source              VARCHAR(250) NOT NULL DEFAULT 'manual',
    status              VARCHAR(30) NOT NULL DEFAULT 'DRAFT',
    is_current          BOOLEAN NOT NULL DEFAULT FALSE,
    locked              BOOLEAN NOT NULL DEFAULT FALSE,
    header              JSONB NOT NULL DEFAULT '{}'::jsonb,
    notes               TEXT NOT NULL DEFAULT '',
    created_by          VARCHAR(150) NOT NULL DEFAULT 'system',
    created_at          TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at          TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    published_at        TIMESTAMPTZ,
    legacy_mongo_id     VARCHAR(80),
    CONSTRAINT ck_action_catalog_versions_status CHECK (status IN ('DRAFT', 'ACTIVE', 'ARCHIVED')),
    CONSTRAINT ck_action_catalog_versions_header_object CHECK (jsonb_typeof(header) = 'object')
);

CREATE UNIQUE INDEX IF NOT EXISTS uq_action_catalog_one_current
    ON {{schema}}.action_catalog_versions (is_current)
    WHERE is_current = TRUE;
CREATE INDEX IF NOT EXISTS idx_action_catalog_versions_created
    ON {{schema}}.action_catalog_versions (created_at DESC);
DROP TRIGGER IF EXISTS trg_action_catalog_versions_updated_at ON {{schema}}.action_catalog_versions;
CREATE TRIGGER trg_action_catalog_versions_updated_at
BEFORE UPDATE ON {{schema}}.action_catalog_versions
FOR EACH ROW EXECUTE FUNCTION {{schema}}.set_updated_at();

CREATE TABLE IF NOT EXISTS {{schema}}.action_catalog_rows (
    id                      UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    version_id              VARCHAR(180) NOT NULL REFERENCES {{schema}}.action_catalog_versions(version_id)
                                ON UPDATE CASCADE ON DELETE CASCADE,
    catalog_code            VARCHAR(80) NOT NULL,
    section_name            TEXT NOT NULL DEFAULT '',
    side                    VARCHAR(80) NOT NULL DEFAULT 'general',
    condition_code          VARCHAR(150) NOT NULL,
    row_order               INTEGER NOT NULL DEFAULT 0,
    section_order           INTEGER NOT NULL DEFAULT 0,
    description             TEXT NOT NULL DEFAULT '',
    action_code             TEXT NOT NULL DEFAULT '',
    classification          VARCHAR(80) NOT NULL DEFAULT '',
    oe                      BOOLEAN NOT NULL DEFAULT FALSE,
    replacement             BOOLEAN NOT NULL DEFAULT FALSE,
    scrap                   BOOLEAN NOT NULL DEFAULT FALSE,
    critical_characteristic BOOLEAN NOT NULL DEFAULT FALSE,
    is_note                 BOOLEAN NOT NULL DEFAULT FALSE,
    active                  BOOLEAN NOT NULL DEFAULT TRUE,
    source_page             INTEGER,
    row_document            JSONB NOT NULL DEFAULT '{}'::jsonb,
    updated_by              VARCHAR(150),
    legacy_mongo_id         VARCHAR(80),
    created_at              TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at              TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    CONSTRAINT uq_action_catalog_row_condition UNIQUE (version_id, condition_code),
    CONSTRAINT ck_action_catalog_rows_orders CHECK (row_order >= 0 AND section_order >= 0),
    CONSTRAINT ck_action_catalog_rows_document_object CHECK (jsonb_typeof(row_document) = 'object')
);
CREATE INDEX IF NOT EXISTS idx_action_catalog_rows_section
    ON {{schema}}.action_catalog_rows (version_id, section_order, catalog_code, row_order);
CREATE INDEX IF NOT EXISTS idx_action_catalog_rows_lookup
    ON {{schema}}.action_catalog_rows (version_id, catalog_code, active);
DROP TRIGGER IF EXISTS trg_action_catalog_rows_updated_at ON {{schema}}.action_catalog_rows;
CREATE TRIGGER trg_action_catalog_rows_updated_at
BEFORE UPDATE ON {{schema}}.action_catalog_rows
FOR EACH ROW EXECUTE FUNCTION {{schema}}.set_updated_at();

CREATE TABLE IF NOT EXISTS {{schema}}.action_catalog_images (
    id                      UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    version_id              VARCHAR(180) NOT NULL REFERENCES {{schema}}.action_catalog_versions(version_id)
                                ON UPDATE CASCADE ON DELETE CASCADE,
    catalog_code            VARCHAR(80) NOT NULL,
    section_name            TEXT NOT NULL DEFAULT '',
    side                    VARCHAR(80) NOT NULL DEFAULT 'general',
    description             TEXT NOT NULL DEFAULT '',
    condition_code          VARCHAR(150),
    action_code             TEXT,
    classification          VARCHAR(80),
    image_order             INTEGER NOT NULL DEFAULT 0,
    page_no                 INTEGER,
    asset_id                UUID REFERENCES {{schema}}.file_assets(id)
                                ON UPDATE CASCADE ON DELETE RESTRICT,
    image_path              TEXT,
    legacy_gridfs_bucket    VARCHAR(150),
    legacy_gridfs_file_id   VARCHAR(100),
    content_type            VARCHAR(150),
    file_size_bytes         BIGINT,
    bbox                    JSONB,
    active                  BOOLEAN NOT NULL DEFAULT TRUE,
    metadata                JSONB NOT NULL DEFAULT '{}'::jsonb,
    legacy_mongo_id         VARCHAR(80),
    created_at              TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at              TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    CONSTRAINT uq_action_catalog_image_order UNIQUE (version_id, catalog_code, image_order),
    CONSTRAINT ck_action_catalog_images_order CHECK (image_order >= 0),
    CONSTRAINT ck_action_catalog_images_size CHECK (file_size_bytes IS NULL OR file_size_bytes >= 0),
    CONSTRAINT ck_action_catalog_images_metadata_object CHECK (jsonb_typeof(metadata) = 'object')
);
CREATE INDEX IF NOT EXISTS idx_action_catalog_images_asset
    ON {{schema}}.action_catalog_images (asset_id);
CREATE INDEX IF NOT EXISTS idx_action_catalog_images_gridfs
    ON {{schema}}.action_catalog_images (legacy_gridfs_bucket, legacy_gridfs_file_id)
    WHERE legacy_gridfs_file_id IS NOT NULL;
DROP TRIGGER IF EXISTS trg_action_catalog_images_updated_at ON {{schema}}.action_catalog_images;
CREATE TRIGGER trg_action_catalog_images_updated_at
BEFORE UPDATE ON {{schema}}.action_catalog_images
FOR EACH ROW EXECUTE FUNCTION {{schema}}.set_updated_at();

CREATE TABLE IF NOT EXISTS {{schema}}.action_catalog_audit_log (
    id                  BIGSERIAL PRIMARY KEY,
    event_type          VARCHAR(100) NOT NULL,
    version_id          VARCHAR(180),
    operator_name       VARCHAR(150) NOT NULL DEFAULT 'system',
    event_document      JSONB NOT NULL DEFAULT '{}'::jsonb,
    legacy_mongo_id     VARCHAR(80),
    created_at          TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    CONSTRAINT ck_action_catalog_audit_document_object CHECK (jsonb_typeof(event_document) = 'object')
);
CREATE INDEX IF NOT EXISTS idx_action_catalog_audit_version_created
    ON {{schema}}.action_catalog_audit_log (version_id, created_at DESC);
CREATE UNIQUE INDEX IF NOT EXISTS uq_action_catalog_audit_legacy
    ON {{schema}}.action_catalog_audit_log (legacy_mongo_id)
    WHERE legacy_mongo_id IS NOT NULL;

CREATE TABLE IF NOT EXISTS {{schema}}.ai_defect_catalog_map (
    id                  UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    ai_label            VARCHAR(180) NOT NULL,
    side                VARCHAR(80) NOT NULL,
    model_version       VARCHAR(120) NOT NULL DEFAULT 'v1.0',
    catalog_code        VARCHAR(80) NOT NULL,
    min_confidence      DOUBLE PRECISION NOT NULL DEFAULT 0,
    active              BOOLEAN NOT NULL DEFAULT TRUE,
    updated_by          VARCHAR(150),
    mapping_document    JSONB NOT NULL DEFAULT '{}'::jsonb,
    legacy_mongo_id     VARCHAR(80),
    created_at          TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at          TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    CONSTRAINT uq_ai_defect_catalog_map UNIQUE (ai_label, side, model_version),
    CONSTRAINT ck_ai_defect_catalog_confidence CHECK (min_confidence >= 0 AND min_confidence <= 1),
    CONSTRAINT ck_ai_defect_catalog_document_object CHECK (jsonb_typeof(mapping_document) = 'object')
);
CREATE INDEX IF NOT EXISTS idx_ai_defect_catalog_lookup
    ON {{schema}}.ai_defect_catalog_map (model_version, side, ai_label, active);
DROP TRIGGER IF EXISTS trg_ai_defect_catalog_map_updated_at ON {{schema}}.ai_defect_catalog_map;
CREATE TRIGGER trg_ai_defect_catalog_map_updated_at
BEFORE UPDATE ON {{schema}}.ai_defect_catalog_map
FOR EACH ROW EXECUTE FUNCTION {{schema}}.set_updated_at();

CREATE TABLE IF NOT EXISTS {{schema}}.action_decision_rules (
    id                  UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    rule_id             VARCHAR(180) NOT NULL UNIQUE,
    version_id          VARCHAR(180) NOT NULL REFERENCES {{schema}}.action_catalog_versions(version_id)
                            ON UPDATE CASCADE ON DELETE CASCADE,
    catalog_code        VARCHAR(80) NOT NULL,
    condition_code      VARCHAR(150),
    measurement_field   VARCHAR(180),
    comparison_operator VARCHAR(10) NOT NULL DEFAULT '>=',
    comparison_value    DOUBLE PRECISION,
    final_decision      VARCHAR(100),
    priority            INTEGER NOT NULL DEFAULT 0,
    active              BOOLEAN NOT NULL DEFAULT TRUE,
    rule_document       JSONB NOT NULL DEFAULT '{}'::jsonb,
    legacy_mongo_id     VARCHAR(80),
    created_at          TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at          TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    CONSTRAINT ck_action_decision_rules_document_object CHECK (jsonb_typeof(rule_document) = 'object')
);
CREATE INDEX IF NOT EXISTS idx_action_decision_rules_lookup
    ON {{schema}}.action_decision_rules (version_id, catalog_code, active, priority DESC);
DROP TRIGGER IF EXISTS trg_action_decision_rules_updated_at ON {{schema}}.action_decision_rules;
CREATE TRIGGER trg_action_decision_rules_updated_at
BEFORE UPDATE ON {{schema}}.action_decision_rules
FOR EACH ROW EXECUTE FUNCTION {{schema}}.set_updated_at();

CREATE TABLE IF NOT EXISTS {{schema}}.inspection_action_decisions (
    id                  UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    cycle_id            VARCHAR(200),
    cycle_uid           VARCHAR(300),
    sku_name            VARCHAR(150),
    tyre_name           VARCHAR(200),
    side                VARCHAR(80),
    ai_label            VARCHAR(180),
    final_decision      VARCHAR(100),
    resolved            BOOLEAN NOT NULL DEFAULT FALSE,
    version_id          VARCHAR(180),
    decision_document   JSONB NOT NULL DEFAULT '{}'::jsonb,
    legacy_mongo_id     VARCHAR(80),
    created_at          TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    CONSTRAINT ck_inspection_action_decision_document_object CHECK (jsonb_typeof(decision_document) = 'object')
);
CREATE INDEX IF NOT EXISTS idx_inspection_action_decisions_cycle
    ON {{schema}}.inspection_action_decisions (cycle_id, side, ai_label, created_at DESC);
CREATE UNIQUE INDEX IF NOT EXISTS uq_inspection_action_decisions_legacy
    ON {{schema}}.inspection_action_decisions (legacy_mongo_id)
    WHERE legacy_mongo_id IS NOT NULL;

CREATE TABLE IF NOT EXISTS {{schema}}.ai_models (
    id                  UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    model_name          VARCHAR(220) NOT NULL,
    model_version       VARCHAR(150) NOT NULL,
    model_type          VARCHAR(120) NOT NULL DEFAULT 'UNSPECIFIED',
    framework           VARCHAR(120),
    sku_name            VARCHAR(150),
    zone                VARCHAR(80),
    camera_serial       VARCHAR(150),
    asset_id            UUID REFERENCES {{schema}}.file_assets(id)
                            ON UPDATE CASCADE ON DELETE RESTRICT,
    status              VARCHAR(40) NOT NULL DEFAULT 'VALIDATION_PENDING',
    active              BOOLEAN NOT NULL DEFAULT FALSE,
    validation_status   VARCHAR(40),
    validation_score    DOUBLE PRECISION,
    model_document      JSONB NOT NULL DEFAULT '{}'::jsonb,
    legacy_gridfs_bucket VARCHAR(150),
    legacy_gridfs_file_id VARCHAR(100),
    legacy_mongo_id     VARCHAR(80),
    created_by          VARCHAR(150) NOT NULL DEFAULT 'system',
    created_at          TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at          TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    published_at        TIMESTAMPTZ,
    activated_at        TIMESTAMPTZ,
    CONSTRAINT uq_ai_models_identity UNIQUE NULLS NOT DISTINCT (model_name, model_version, sku_name, zone, camera_serial),
    CONSTRAINT ck_ai_models_status CHECK (
        status IN ('VALIDATION_PENDING', 'VALIDATED', 'PUBLISHED', 'READY', 'ACTIVE', 'REJECTED', 'FAILED', 'MISSING_BINARY', 'ARCHIVED')
    ),
    CONSTRAINT ck_ai_models_document_object CHECK (jsonb_typeof(model_document) = 'object')
);
CREATE INDEX IF NOT EXISTS idx_ai_models_lookup
    ON {{schema}}.ai_models (sku_name, zone, active, updated_at DESC);
CREATE INDEX IF NOT EXISTS idx_ai_models_asset
    ON {{schema}}.ai_models (asset_id);
CREATE INDEX IF NOT EXISTS idx_ai_models_status
    ON {{schema}}.ai_models (status, updated_at DESC);
CREATE UNIQUE INDEX IF NOT EXISTS uq_ai_models_legacy_mongo_id
    ON {{schema}}.ai_models (legacy_mongo_id)
    WHERE legacy_mongo_id IS NOT NULL;
DROP TRIGGER IF EXISTS trg_ai_models_updated_at ON {{schema}}.ai_models;
CREATE TRIGGER trg_ai_models_updated_at
BEFORE UPDATE ON {{schema}}.ai_models
FOR EACH ROW EXECUTE FUNCTION {{schema}}.set_updated_at();

CREATE TABLE IF NOT EXISTS {{schema}}.ai_model_deployments (
    id                  UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    model_id            UUID NOT NULL REFERENCES {{schema}}.ai_models(id)
                            ON UPDATE CASCADE ON DELETE CASCADE,
    deployment_target   VARCHAR(180) NOT NULL DEFAULT 'EDGE_LOCAL',
    deployment_status   VARCHAR(40) NOT NULL DEFAULT 'PENDING',
    local_cache_path    TEXT,
    checksum_verified   BOOLEAN NOT NULL DEFAULT FALSE,
    error_message       TEXT,
    deployment_document JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at          TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at          TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    loaded_at           TIMESTAMPTZ,
    activated_at        TIMESTAMPTZ,
    CONSTRAINT ck_ai_model_deployments_status CHECK (
        deployment_status IN ('PENDING', 'DOWNLOADING', 'READY', 'LOADED', 'ACTIVE', 'FAILED', 'ROLLED_BACK')
    ),
    CONSTRAINT ck_ai_model_deployment_document_object CHECK (jsonb_typeof(deployment_document) = 'object')
);
CREATE INDEX IF NOT EXISTS idx_ai_model_deployments_model_created
    ON {{schema}}.ai_model_deployments (model_id, created_at DESC);
DROP TRIGGER IF EXISTS trg_ai_model_deployments_updated_at ON {{schema}}.ai_model_deployments;
CREATE TRIGGER trg_ai_model_deployments_updated_at
BEFORE UPDATE ON {{schema}}.ai_model_deployments
FOR EACH ROW EXECUTE FUNCTION {{schema}}.set_updated_at();

INSERT INTO {{schema}}.application_settings (setting_key, setting_value, description)
VALUES (
    'postgres_phase',
    '{"phase": "4B", "status": "catalog_and_ai_models_ready", "catalog_backend": "postgresql", "model_backend": "postgresql_chunked", "mongodb_fallback": true}'::jsonb,
    'Tracks the active PostgreSQL migration phase for Apollo Tyre Inspection.'
)
ON CONFLICT (setting_key) DO UPDATE SET
    setting_value = EXCLUDED.setting_value,
    description = EXCLUDED.description,
    updated_at = NOW();
