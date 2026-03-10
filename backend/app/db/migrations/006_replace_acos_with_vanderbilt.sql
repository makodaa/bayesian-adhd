-- Migration 006: Replace ACOS fields with Vanderbilt ADHD-core fields

ALTER TABLE results DROP COLUMN IF EXISTS acos_total_score;
ALTER TABLE results DROP COLUMN IF EXISTS acos_average_score;
ALTER TABLE results DROP COLUMN IF EXISTS acos_severity;
ALTER TABLE results DROP COLUMN IF EXISTS acos_subscale_scores;
ALTER TABLE results DROP COLUMN IF EXISTS acos_item_scores;

ALTER TABLE results ADD COLUMN IF NOT EXISTS vanderbilt_scale_type VARCHAR(20);
ALTER TABLE results ADD COLUMN IF NOT EXISTS vanderbilt_inattentive_count INTEGER;
ALTER TABLE results ADD COLUMN IF NOT EXISTS vanderbilt_hyperactive_impulsive_count INTEGER;
ALTER TABLE results ADD COLUMN IF NOT EXISTS vanderbilt_performance_impairment_count INTEGER;
ALTER TABLE results ADD COLUMN IF NOT EXISTS vanderbilt_adhd_inattentive_met BOOLEAN;
ALTER TABLE results ADD COLUMN IF NOT EXISTS vanderbilt_adhd_hyperactive_impulsive_met BOOLEAN;
ALTER TABLE results ADD COLUMN IF NOT EXISTS vanderbilt_adhd_combined_met BOOLEAN;
ALTER TABLE results ADD COLUMN IF NOT EXISTS vanderbilt_interpretation TEXT;
ALTER TABLE results ADD COLUMN IF NOT EXISTS vanderbilt_domain_scores JSONB;
ALTER TABLE results ADD COLUMN IF NOT EXISTS vanderbilt_symptom_scores JSONB;
ALTER TABLE results ADD COLUMN IF NOT EXISTS vanderbilt_performance_scores JSONB;
