# data/

This folder is git-ignored. Put the PhysioNet files here after getting credentialed access:

| File | Source |
|---|---|
| `discharge.csv` | [MIMIC-IV-Note](https://physionet.org/content/mimic-iv-note/) (columns include `note_id`, `subject_id`, `text`) |
| `mimic-iv-bhc.csv` | [MIMIC-IV BHC labelled notes](https://physionet.org/content/labelled-notes-hospital-course/) (columns include `note_id`, `target`) |

Never commit these files, generated summaries, or any output that contains note text. That is a requirement of the PhysioNet Data Use Agreement.
