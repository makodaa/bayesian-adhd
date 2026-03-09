-- Migration 005: Persist ACOS-C scoring outputs with EEG result rows

ALTER TABLE results ADD COLUMN IF NOT EXISTS acos_total_score INTEGER;
ALTER TABLE results ADD COLUMN IF NOT EXISTS acos_average_score NUMERIC(4,3);
ALTER TABLE results ADD COLUMN IF NOT EXISTS acos_severity VARCHAR(40);
ALTER TABLE results ADD COLUMN IF NOT EXISTS acos_subscale_scores JSONB;
ALTER TABLE results ADD COLUMN IF NOT EXISTS acos_item_scores JSONB;
