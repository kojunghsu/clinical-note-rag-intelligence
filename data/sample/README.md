# Synthetic Discharge Note Sample

This folder contains synthetic, fictitious sample data for testing the project pipeline without redistributing protected MIMIC-IV-Note data.

## File

- `discharge_sample.csv`: 1,000 fake discharge summary records with the same column structure as MIMIC-IV-Note `discharge.csv`.

## Schema

```text
note_id, subject_id, hadm_id, note_type, note_seq, charttime, storetime, text
```

## Important Notice

The records in this folder are entirely synthetic. They do not describe real patients, do not contain protected health information, and were not copied from MIMIC-IV-Note or any other clinical dataset.

This sample is provided only so reviewers can test CSV loading, chunking, embedding, retrieval, structured lookup, and RAG pipeline behavior. It should not be used for clinical inference, medical decision-making, model evaluation conclusions, or research claims about real patient populations.

To run the full project with real data, users must obtain authorized access to MIMIC-IV and MIMIC-IV-Note through PhysioNet and follow all applicable Data Use Agreement requirements.
