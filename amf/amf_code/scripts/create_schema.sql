-- Create necessary tables under amf DB
CREATE TABLE IF NOT EXISTS public.ImageReference
(
  Id                        bigint  GENERATED ALWAYS AS IDENTITY,
  FileNameReference         text    NOT NULL,
  UploadTimestamp           timestamp default current_timestamp,
  TileEdge                  integer,
  UpdatedAt                 timestamp default current_timestamp,
  Enabled                   boolean,
  CONSTRAINT PK_ImageReference  PRIMARY KEY (Id)
);

CREATE TABLE IF NOT EXISTS public.Cnn1PredictionsAm
(
    ImageReferenceId    bigint  NOT NULL,
    RowNum              integer NOT NULL,
    ColNum              integer NOT NULL,
    AmColonised         numeric,
    Uncolonised         numeric,
    Background          numeric,
    Unreadable          numeric,
    DSE                 numeric,
    Hybrid              numeric,
    ContextualLabel      text,
    CONSTRAINT PK_row_col_predictions_cnn1_am       PRIMARY KEY (ImageReferenceId, RowNum, ColNum),
    CONSTRAINT FK_Prediction_ImageReferenceId_Am    FOREIGN KEY (ImageReferenceId) REFERENCES public.ImageReference(Id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS public.Cnn1AnnotationsAm
(
    ImageReferenceId    bigint  NOT NULL,
    RowNum              integer NOT NULL,
    ColNum              integer NOT NULL,
    AmColonised         integer,
    Uncolonised         integer,
    Background          integer,
    Unreadable          integer,
    DSE                 integer,
    Hybrid              integer,
    Question            integer,
    QuestionComment     text,
    CONSTRAINT PK_row_col_annotations_cnn1_am       PRIMARY KEY (ImageReferenceId, RowNum, ColNum),
    CONSTRAINT FK_Annotation_ImageReferenceId_Am    FOREIGN KEY (ImageReferenceId) REFERENCES public.ImageReference(Id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS public.Cnn1PredictionsErm
(
    ImageReferenceId    bigint  NOT NULL,
    RowNum              integer NOT NULL,
    ColNum              integer NOT NULL,
    BlueCoils           numeric,
    BrownCoils          numeric,
    TypeTwo             numeric,
    Uncolonised         numeric,
    Background          numeric,
    MainRoot            numeric,
    Unreadable          numeric,
    DSE                 numeric,
    HybridErm           numeric,
    HybridDse           numeric,
    ContextualLabel      text,
    CONSTRAINT PK_row_col_predictions_cnn1_erm       PRIMARY KEY (ImageReferenceId, RowNum, ColNum),
    CONSTRAINT FK_Prediction_ImageReferenceId_Erm    FOREIGN KEY (ImageReferenceId) REFERENCES public.ImageReference(Id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS public.Cnn1AnnotationsErm
(
    ImageReferenceId    bigint  NOT NULL,
    RowNum              integer NOT NULL,
    ColNum              integer NOT NULL,
    BlueCoils           integer,
    BrownCoils          integer,
    TypeTwo             integer,
    Uncolonised         integer,
    Background          integer,
    MainRoot            integer,
    Unreadable          integer,
    DSE                 integer,
    HybridErm           integer,
    HybridDse           integer,
    Question            integer,
    QuestionComment     text,
    CONSTRAINT PK_row_col_annotations_cnn1_erm       PRIMARY KEY (ImageReferenceId, RowNum, ColNum),
    CONSTRAINT FK_Annotation_ImageReferenceId_Erm    FOREIGN KEY (ImageReferenceId) REFERENCES public.ImageReference(Id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS public.Cnn2PredictionsAm
(
    ImageReferenceId    bigint  NOT NULL,
    RowNum              integer NOT NULL,
    ColNum              integer NOT NULL,
    Arbuscule           numeric,
    Vesicle             numeric,
    Hyphopodium         numeric,
    Hypha               numeric,
    CONSTRAINT PK_row_col_predictions_cnn2_am      PRIMARY KEY (ImageReferenceId, RowNum, ColNum),
    CONSTRAINT FK_Prediction_ImageReferenceId_Am   FOREIGN KEY (ImageReferenceId) REFERENCES public.ImageReference(Id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS public.Cnn2AnnotationsAm
(
    ImageReferenceId    bigint  NOT NULL,
    RowNum              integer NOT NULL,
    ColNum              integer NOT NULL,
    Arbuscule           numeric,
    Vesicle             numeric,
    Hyphopodium         numeric,
    Hypha               numeric,
    CONSTRAINT PK_row_col_annotations_cnn2_am      PRIMARY KEY (ImageReferenceId, RowNum, ColNum),
    CONSTRAINT FK_Annotation_ImageReferenceId_Am   FOREIGN KEY (ImageReferenceId) REFERENCES public.ImageReference(Id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS public.Settings (
    id              SERIAL PRIMARY KEY,
    key             TEXT UNIQUE,
    value_type      TEXT CHECK (value_type IN ('integer', 'string', 'boolean', 'float')),
    value           TEXT,
    default_value   TEXT
);

ALTER TABLE IF EXISTS public.Cnn1PredictionsAm
    OWNER to postgres;

ALTER TABLE IF EXISTS public.Cnn1AnnotationsAm
    OWNER to postgres;

ALTER TABLE IF EXISTS public.Cnn2PredictionsAm
    OWNER to postgres;

ALTER TABLE IF EXISTS public.Cnn2AnnotationsAm
    OWNER to postgres;

ALTER TABLE IF EXISTS public.Cnn1PredictionsErm
    OWNER to postgres;

ALTER TABLE IF EXISTS public.Cnn1AnnotationsErm
    OWNER to postgres;

ALTER TABLE IF EXISTS public.ImageReference
    OWNER to postgres;

ALTER TABLE IF EXISTS public.Settings
    OWNER to postgres;
