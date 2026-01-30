-- Migration: Add fast_alpha and high_beta to band_powers CHECK constraint
-- Date: 2026-01-30
-- Description: Updates the band_powers table to allow all 7 frequency bands
--              (delta, theta, alpha, beta, gamma, fast_alpha, high_beta)

-- Drop the existing CHECK constraint
ALTER TABLE band_powers DROP CONSTRAINT IF EXISTS band_powers_frequency_band_check;

-- Add the updated CHECK constraint with all 7 bands
ALTER TABLE band_powers ADD CONSTRAINT band_powers_frequency_band_check 
    CHECK (frequency_band IN ('delta', 'theta', 'alpha', 'beta', 'gamma', 'fast_alpha', 'high_beta'));
