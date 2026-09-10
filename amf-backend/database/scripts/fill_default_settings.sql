-- Default values for every persisted setting.
--
-- Executed on EVERY API startup (run_amf_api.py) and idempotent, so it both
-- seeds a fresh database and adds any newly-introduced key to an existing one.
-- A new setting therefore only needs a row here - there is no separate
-- migration step, and nothing else to remember.
--
-- ON CONFLICT DO NOTHING is what makes that safe: a key already present keeps
-- whatever value the user has set. Note it also means a change to an existing
-- row's value_type/default_value will NOT reach databases that already have
-- that key; such a correction needs its own guarded UPDATE at the end of this
-- file (see dropoutRate below for the pattern).

INSERT INTO public.Settings (key, value_type, value, default_value) VALUES
    ('outdir', 'string', '', ''),
    ('imageDirectory', 'string', '', ''),
    ('batchSize', 'integer', '32', '32'),
    ('epochs', 'integer', '50', '50'),
    ('model', 'string', 'am_252_efficientnet.pth', 'am_252_efficientnet.pth'),
    ('modelErm', 'string', 'erm_126_efficientnet.pth', 'erm_126_efficientnet.pth'),
    ('modelPath', 'string', '', ''),
    ('vfrac', 'float', '0.2', '0.2'),
    ('dataAugm', 'boolean', 'false', 'false'),
    ('summary', 'boolean', 'false', 'false'),
    ('threshold', 'float', '0.5', '0.5'),
    ('useDb', 'boolean', 'true', 'true'),
    ('aggregateTiles', 'boolean', 'false', 'false'),
    ('device', 'string', 'automatic', 'automatic'),
    ('numWorkers', 'integer', '0', '0'),
    ('learningRate', 'float', '0.000004218361045', '0.000004218361045'),
    ('learningRateActiveLearning', 'float', '0.0000004', '0.0000004'),
    ('getTilesForLabellingUsingActiveLearning', 'boolean', 'false', 'false'),
    ('activeLearningMethod', 'string', 'bald', 'bald'),
    ('numSamplesForLabelling', 'integer', '10', '10'),
    ('mcSamples', 'integer', '50', '50'),
    ('dropoutRate', 'float', '0.0', '0.0'),
    ('dropPathRate', 'float', '0.0', '0.0'),
    ('epochsActiveLearning', 'integer', '5', '5'),
    ('trainActiveLearning', 'boolean', 'false', 'false'),
    ('adamBeta1', 'float', '0.906450740008503', '0.906450740008503'),
    ('adamBeta2', 'float', '0.986390310777448', '0.986390310777448'),
    ('balanceFactor', 'float', '1.24895925434138', '1.24895925434138'),
    ('mlFlowFlag', 'boolean', 'false', 'false'),
    ('modelType', 'string', 'efficientnet', 'efficientnet'),
    ('preTrained', 'boolean', 'true', 'true'),
    ('filterBackground', 'boolean', 'false', 'false'),
    ('dynamicLoading', 'boolean', 'false', 'false'),
    ('pretiledDir', 'string', '', ''),
    ('checkpointPath', 'string', '', ''),
    ('resizeDim', 'integer', '', ''),
    ('weightDecay', 'float', '0.0', '0.0'),
    ('backboneLrMult', 'float', '1.0', '1.0'),
    ('freezeEpochs', 'integer', '0', '0'),
    ('earlyBreakEpoch', 'integer', '', ''),
    ('patienceE', 'integer', '10', '10'),
    ('patienceR', 'integer', '5', '5'),
    ('convertImageFileType', 'string', 'jpg', 'jpg'),
    ('temperatureFactorPath', 'string', 'am_252_efficientnet_temperature_value.txt', 'am_252_efficientnet_temperature_value.txt'),
    ('temperatureFactorPathErm', 'string', 'erm_126_efficientnet_temperature_value.txt', 'erm_126_efficientnet_temperature_value.txt'),
    ('useContextualConfidence', 'boolean', 'true', 'true'),
    ('contextualConfidenceThreshold', 'float', '1.0', '1.0'),
    ('ciMethod', 'string', 'analytic', 'analytic'),
    ('tagPalette', 'string', '{}', '{}')
ON CONFLICT (key) DO NOTHING;

-- One-time corrections to rows seeded by an earlier version. Each is guarded on
-- the old value so it repairs the stale default exactly once and can never
-- prevent a deliberate later change, even though this runs on every startup.

UPDATE public.Settings SET default_value = '0.0'
WHERE key = 'dropoutRate' AND default_value = '0.25';
