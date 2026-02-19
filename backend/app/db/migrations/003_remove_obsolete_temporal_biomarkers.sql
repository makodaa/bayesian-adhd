-- Migration: Remove obsolete temporal biomarker data
-- Keeps only "Band Power Ratios" and "Relative Band Powers" groups/keys.
-- Removes: Spectral Features, Hjorth Parameters, Regional Power,
--          Hemispheric Asymmetry, Anterior-Posterior Ratios.

-- Delete old temporal plot groups that no longer exist
DELETE FROM temporal_plots
WHERE group_name NOT IN ('Band Power Ratios', 'Relative Band Powers');

-- Delete old temporal summary keys that are no longer computed
DELETE FROM temporal_summaries
WHERE biomarker_key NOT IN (
    'theta_beta_ratio',
    'theta_alpha_ratio',
    'alpha_theta_ratio',
    'alpha_beta_ratio',
    'delta_theta_ratio',
    'low_beta_high_beta_ratio',
    'combined_ratio',
    'relative_delta',
    'relative_theta',
    'relative_alpha',
    'relative_beta',
    'relative_gamma'
);
