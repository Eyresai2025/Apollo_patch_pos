-- Apollo VIT PostgreSQL Phase 4A: chunked binary assets for inspection and New SKU images.
-- Existing MongoDB GridFS content remains untouched and can be used as a read fallback.
-- Do not edit this migration after it has been applied.

CREATE TABLE IF NOT EXISTS {{schema}}.file_assets (
    id                  UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    asset_type          VARCHAR(60) NOT NULL,
    filename            VARCHAR(255) NOT NULL,
    content_type        VARCHAR(150),
    file_size_bytes     BIGINT NOT NULL DEFAULT 0,
    checksum_sha256     CHAR(64),
    storage_status      VARCHAR(30) NOT NULL DEFAULT 'UPLOADING',
    source_backend      VARCHAR(50) NOT NULL DEFAULT 'LOCAL_FILE',
    source_id           VARCHAR(600),
    original_path       TEXT,
    source_mtime_ns     BIGINT,
    metadata            JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at          TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at          TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    CONSTRAINT ck_file_assets_size_nonnegative CHECK (file_size_bytes >= 0),
    CONSTRAINT ck_file_assets_metadata_object CHECK (jsonb_typeof(metadata) = 'object'),
    CONSTRAINT ck_file_assets_status CHECK (
        storage_status IN ('UPLOADING', 'READY', 'FAILED', 'DELETING')
    )
);

CREATE UNIQUE INDEX IF NOT EXISTS uq_file_assets_source
    ON {{schema}}.file_assets (source_backend, source_id)
    WHERE source_id IS NOT NULL AND source_id <> '';

CREATE INDEX IF NOT EXISTS idx_file_assets_type_created
    ON {{schema}}.file_assets (asset_type, created_at DESC);

CREATE INDEX IF NOT EXISTS idx_file_assets_checksum
    ON {{schema}}.file_assets (checksum_sha256)
    WHERE checksum_sha256 IS NOT NULL;

CREATE INDEX IF NOT EXISTS idx_file_assets_status
    ON {{schema}}.file_assets (storage_status, created_at DESC);

DROP TRIGGER IF EXISTS trg_file_assets_updated_at ON {{schema}}.file_assets;
CREATE TRIGGER trg_file_assets_updated_at
BEFORE UPDATE ON {{schema}}.file_assets
FOR EACH ROW EXECUTE FUNCTION {{schema}}.set_updated_at();

CREATE TABLE IF NOT EXISTS {{schema}}.file_asset_chunks (
    asset_id            UUID NOT NULL REFERENCES {{schema}}.file_assets(id)
                            ON UPDATE CASCADE ON DELETE CASCADE,
    chunk_index         INTEGER NOT NULL,
    chunk_size_bytes    INTEGER NOT NULL,
    chunk_data          BYTEA NOT NULL,
    created_at          TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY (asset_id, chunk_index),
    CONSTRAINT ck_file_asset_chunks_index_nonnegative CHECK (chunk_index >= 0),
    CONSTRAINT ck_file_asset_chunks_size_nonnegative CHECK (chunk_size_bytes >= 0),
    CONSTRAINT ck_file_asset_chunks_size_matches CHECK (octet_length(chunk_data) = chunk_size_bytes)
);

CREATE TABLE IF NOT EXISTS {{schema}}.inspection_images (
    id                  UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    cycle_uid           VARCHAR(300) NOT NULL REFERENCES {{schema}}.inspection_cycles(cycle_uid)
                            ON UPDATE CASCADE ON DELETE CASCADE,
    zone                VARCHAR(30) NOT NULL,
    image_type          VARCHAR(10) NOT NULL,
    asset_id            UUID NOT NULL REFERENCES {{schema}}.file_assets(id)
                            ON UPDATE CASCADE ON DELETE RESTRICT,
    image_status        VARCHAR(30) NOT NULL DEFAULT 'READY',
    metadata            JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at          TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at          TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    CONSTRAINT uq_inspection_images_cycle_zone_type UNIQUE (cycle_uid, zone, image_type),
    CONSTRAINT ck_inspection_images_zone CHECK (
        zone IN ('sidewall1', 'sidewall2', 'innerwall', 'tread', 'bead')
    ),
    CONSTRAINT ck_inspection_images_type CHECK (image_type IN ('INPUT', 'OUTPUT')),
    CONSTRAINT ck_inspection_images_metadata_object CHECK (jsonb_typeof(metadata) = 'object')
);

CREATE INDEX IF NOT EXISTS idx_inspection_images_asset
    ON {{schema}}.inspection_images (asset_id);

CREATE INDEX IF NOT EXISTS idx_inspection_images_cycle
    ON {{schema}}.inspection_images (cycle_uid, zone, image_type);

DROP TRIGGER IF EXISTS trg_inspection_images_updated_at ON {{schema}}.inspection_images;
CREATE TRIGGER trg_inspection_images_updated_at
BEFORE UPDATE ON {{schema}}.inspection_images
FOR EACH ROW EXECUTE FUNCTION {{schema}}.set_updated_at();

CREATE TABLE IF NOT EXISTS {{schema}}.new_sku_images (
    id                  UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    sku_id              UUID REFERENCES {{schema}}.skus(id)
                            ON UPDATE CASCADE ON DELETE SET NULL,
    sku_name            VARCHAR(150) NOT NULL,
    capture_id          VARCHAR(200) NOT NULL,
    camera_serial       VARCHAR(150),
    capture_index       INTEGER,
    save_group          VARCHAR(60),
    label               VARCHAR(150),
    asset_id            UUID NOT NULL REFERENCES {{schema}}.file_assets(id)
                            ON UPDATE CASCADE ON DELETE RESTRICT,
    image_status        VARCHAR(30) NOT NULL DEFAULT 'READY',
    metadata            JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at          TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at          TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    CONSTRAINT ck_new_sku_images_capture_index CHECK (
        capture_index IS NULL OR capture_index >= 0
    ),
    CONSTRAINT ck_new_sku_images_metadata_object CHECK (jsonb_typeof(metadata) = 'object')
);

CREATE UNIQUE INDEX IF NOT EXISTS uq_new_sku_images_capture_camera_index
    ON {{schema}}.new_sku_images (capture_id, camera_serial, capture_index)
    WHERE camera_serial IS NOT NULL AND capture_index IS NOT NULL;

CREATE INDEX IF NOT EXISTS idx_new_sku_images_sku_created
    ON {{schema}}.new_sku_images (sku_name, created_at DESC);

CREATE INDEX IF NOT EXISTS idx_new_sku_images_asset
    ON {{schema}}.new_sku_images (asset_id);

DROP TRIGGER IF EXISTS trg_new_sku_images_updated_at ON {{schema}}.new_sku_images;
CREATE TRIGGER trg_new_sku_images_updated_at
BEFORE UPDATE ON {{schema}}.new_sku_images
FOR EACH ROW EXECUTE FUNCTION {{schema}}.set_updated_at();

ALTER TABLE {{schema}}.inspection_cycles
    ADD COLUMN IF NOT EXISTS asset_linked BOOLEAN NOT NULL DEFAULT FALSE,
    ADD COLUMN IF NOT EXISTS asset_input_count INTEGER NOT NULL DEFAULT 0,
    ADD COLUMN IF NOT EXISTS asset_output_count INTEGER NOT NULL DEFAULT 0,
    ADD COLUMN IF NOT EXISTS asset_failed_count INTEGER NOT NULL DEFAULT 0;

ALTER TABLE {{schema}}.inspection_cycles
    DROP CONSTRAINT IF EXISTS ck_inspection_cycles_asset_counts_nonnegative;
ALTER TABLE {{schema}}.inspection_cycles
    ADD CONSTRAINT ck_inspection_cycles_asset_counts_nonnegative CHECK (
        asset_input_count >= 0
        AND asset_output_count >= 0
        AND asset_failed_count >= 0
    );

INSERT INTO {{schema}}.application_settings (
    setting_key,
    setting_value,
    description
)
VALUES (
    'postgres_phase',
    '{"phase": "4A", "status": "binary_assets_ready", "image_backend": "postgresql_chunked", "gridfs_fallback": true}'::jsonb,
    'Tracks the active PostgreSQL migration phase for Apollo VIT.'
)
ON CONFLICT (setting_key) DO UPDATE SET
    setting_value = EXCLUDED.setting_value,
    description = EXCLUDED.description,
    updated_at = NOW();
