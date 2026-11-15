# CurriculumLens — Full SVM Project

AI-based syllabus similarity, curriculum comparison and novelty detection using MiniLM embeddings and a Linear SVM.

## Project structure

```text
CurriculumLens_Full_Project/
├── app.py
├── config.py
├── database.py
├── minilm_engine.py
├── model_training.py
├── comparison.py
├── syllabus_utils.py
├── reporting.py
├── requirements.txt
├── syllabi.db
├── data/
│   └── syllabus_pairs.csv
└── models/                 # trained model artifact is saved here
```

## 1. Open the project

Extract the ZIP file, open the `CurriculumLens_Full_Project` folder in VS Code, then open a terminal in that folder.

## 2. Install packages

```bash
py -m pip install -r requirements.txt
```

If `py` is not recognized, try `python -m pip install -r requirements.txt`.

## 3. Train and evaluate the Linear SVM

```bash
py model_training.py
```

This loads the MiniLM encoder, evaluates using a course-held-out split where possible, trains the final Linear SVM on all labelled rows, and saves:

```text
models/curriculumlens_linear_svm.joblib
```

The first run downloads `sentence-transformers/all-MiniLM-L6-v2`, so an internet connection is required. Feature extraction can take several minutes.

## 4. Run the Streamlit website

```bash
py -m streamlit run app.py
```

Streamlit will print a local URL, normally `http://localhost:8501`.

The app loads the built-in CSV and trains its in-memory SVM when it starts. The saved `.joblib` artifact from step 3 is an output for reuse; the current app startup path retrains from the CSV rather than loading that artifact.

## Dataset note

`data/syllabus_pairs.csv` contains 595 curated syllabus-pair examples derived from 35 course profiles. The labels are provisional taxonomy-based draft annotations, not independently collected or faculty-verified ground truth. Treat evaluation scores as a pipeline demonstration, not proof of real-world accuracy. The dataset mostly contains course titles and topic outlines; it does not provide independently annotated learning outcomes for every pair.

## Notes

- The classifier is Linear SVM only. MiniLM is the semantic encoder, not a second classifier.
- The model predicts `Similar` (1) or `Different` (0).
- The Streamlit app also provides catalog management, syllabus comparison, topic alignment, novelty estimates and PDF reports.
- The included SQLite database is a starter catalog. You can add more courses through the Catalog section.
