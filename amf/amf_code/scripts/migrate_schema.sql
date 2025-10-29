-- Step 1: Add the UpdatedAt column
ALTER TABLE public.ImageReference
ADD COLUMN UpdatedAt timestamp default current_timestamp;

-- Step 2: Initialize UpdatedAt with the current UploadTimestamp values
UPDATE public.ImageReference
SET UpdatedAt = UploadTimestamp;

-- Step 3: Add contextual column
ALTER TABLE public.Cnn1PredictionsAm
ADD COLUMN ContextualLabel text;

ALTER TABLE public.Cnn1PredictionsErm
ADD COLUMN ContextualLabel text;
