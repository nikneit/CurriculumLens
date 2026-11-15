"""CurriculumLens: Linear SVM training and leakage-aware evaluation.

Expected CSV columns:
    syllabus_a, syllabus_b, label

Optional course identity columns:
    course_a_id, course_b_id

For pair datasets with course IDs, evaluation holds out whole courses:
training pairs contain only training-course IDs and test pairs contain only
test-course IDs. Cross-partition pairs are excluded from that evaluation split.
"""

from pathlib import Path
import re
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from minilm_engine import pair_features, load_encoder


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_CSV = BASE_DIR / "data" / "syllabus_pairs.csv"
DEFAULT_MODEL_PATH = BASE_DIR / "models" / "curriculumlens_linear_svm.joblib"
FEATURE_NAMES = [
    "description_similarity",
    "topic_similarity",
    "outcome_similarity",
    "jaccard_similarity",
    "overall_similarity",
]


class CachedEncoder:
    """Cache repeated MiniLM sentence embeddings while building pair features."""

    def __init__(self, encoder):
        self.encoder = encoder
        self.cache = {}

    def encode(self, sentences, **kwargs):
        if isinstance(sentences, str):
            sentences = [sentences]
        sentences = list(sentences)
        normalize = bool(kwargs.get("normalize_embeddings", False))
        vectors = []

        for sentence in sentences:
            text = str(sentence)
            key = (text, normalize)
            if key not in self.cache:
                encoded = self.encoder.encode(
                    [text],
                    normalize_embeddings=normalize,
                    convert_to_numpy=True,
                    show_progress_bar=False,
                )
                self.cache[key] = np.asarray(encoded, dtype=np.float32)[0]
            vectors.append(self.cache[key])

        if not vectors:
            return np.empty((0, 0), dtype=np.float32)
        return np.vstack(vectors)


def normalize_label(value):
    """Convert common Similar/Different label forms to 1/0."""
    if isinstance(value, str):
        value = value.strip().lower()
        if value in {"1", "similar", "same", "match", "yes"}:
            return 1
        if value in {"0", "different", "dissimilar", "not similar", "no"}:
            return 0

    if value in (0, 1, 0.0, 1.0):
        return int(value)

    raise ValueError(
        f"Unrecognized label {value!r}. Use 1/0 or Similar/Different."
    )


def validate_pair_dataframe(df):
    """Validate and clean the required syllabus-pair CSV columns."""
    required = {"syllabus_a", "syllabus_b", "label"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            "CSV is missing required columns: " + ", ".join(sorted(missing))
        )

    clean = df.copy()
    clean = clean.dropna(subset=["syllabus_a", "syllabus_b", "label"])
    clean["syllabus_a"] = clean["syllabus_a"].astype(str).str.strip()
    clean["syllabus_b"] = clean["syllabus_b"].astype(str).str.strip()
    clean = clean[
        (clean["syllabus_a"].str.len() > 0)
        & (clean["syllabus_b"].str.len() > 0)
    ].copy()

    if clean.empty:
        raise ValueError("No complete syllabus pairs were found in the CSV.")

    clean["label"] = clean["label"].apply(normalize_label)

    if clean["label"].nunique() != 2:
        raise ValueError(
            "The CSV must contain both classes: Similar (1) and Different (0)."
        )

    if clean["label"].value_counts().min() < 2:
        raise ValueError("Include at least two examples of each label.")

    return clean.reset_index(drop=True)


def _course_from_text(raw_text, fallback_title):
    """Convert one CSV syllabus cell into the structure expected by pair_features.

    The supplied 595-row CSV uses:
        Course: <title>. Syllabus: topic one; topic two; topic three

    Its syllabus items are therefore split on semicolons. If another CSV format
    is supplied, the complete text is retained as a single topic.
    """
    text = str(raw_text).strip()
    match = re.search(
        r"^\s*Course:\s*(.*?)\.\s*Syllabus:\s*(.*)$",
        text,
        flags=re.IGNORECASE | re.DOTALL,
    )

    if match:
        title = match.group(1).strip() or fallback_title
        syllabus_body = match.group(2).strip()
    else:
        title = fallback_title
        syllabus_body = text

    topics = [
        item.strip(" \t\r\n.-")
        for item in re.split(r"[;\n]+", syllabus_body)
        if item.strip(" \t\r\n.-")
    ]

    if not topics:
        topics = [syllabus_body or text or fallback_title]

    # This CSV contains course topic outlines, not separate learning outcomes.
    # Outcomes remain empty rather than inventing outcome annotations.
    return {
        "title": title,
        "description": title,
        "outcomes": [],
        "topics": topics,
    }


def build_feature_matrix(df, encoder):
    """Build the five MiniLM/Jaccard features for every syllabus pair."""
    cached_encoder = CachedEncoder(encoder)
    features = []

    for index, row in df.iterrows():
        course_a = _course_from_text(
            row["syllabus_a"], f"Syllabus A {index + 1}"
        )
        course_b = _course_from_text(
            row["syllabus_b"], f"Syllabus B {index + 1}"
        )
        feature_row = pair_features(course_a, course_b, cached_encoder)
        features.append(feature_row)

    matrix = np.asarray(features, dtype=float)

    if matrix.ndim != 2 or matrix.shape[1] != len(FEATURE_NAMES):
        raise ValueError(
            f"Expected {len(FEATURE_NAMES)} features per pair; got shape {matrix.shape}."
        )

    if not np.isfinite(matrix).all():
        raise ValueError("Feature extraction produced NaN or infinite values.")

    return matrix


def load_training_data(encoder, csv_path=DEFAULT_CSV):
    """Load the labelled CSV and generate X/y for model training."""
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(
            f"Training CSV not found: {csv_path}\n"
            "Place the downloaded dataset at data/syllabus_pairs.csv."
        )

    df = validate_pair_dataframe(
        pd.read_csv(csv_path, encoding="utf-8-sig")
    )
    X = build_feature_matrix(df, encoder)
    y = df["label"].to_numpy(dtype=int)
    return X, y, df


def classifier_set():
    """Return the project's single supervised classifier: Linear SVM."""
    return {
        "SVM": SVC(
            kernel="linear",
            probability=True,
            class_weight="balanced",
            random_state=42,
        )
    }


def train_models(encoder, csv_path=DEFAULT_CSV):
    """Train the final application model on all labelled rows.

    The app expects this return structure:
        models, scaler, X, y, source_dataframe
    Evaluation is performed separately by evaluate_dataframe().
    """
    X, y, source_df = load_training_data(encoder, csv_path)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = classifier_set()["SVM"]
    model.fit(X_scaled, y)

    return {"SVM": model}, scaler, X, y, source_df


def _course_holdout_split(df, test_size=0.25, random_state=42):
    """Split pair rows by course identity so no course appears on both sides.

    A pair is retained in training only when both course IDs are in the
    training-course partition. It is retained in testing only when both IDs
    are in the test-course partition. Cross-partition pairs are excluded.
    """
    required_ids = {"course_a_id", "course_b_id"}
    if not required_ids.issubset(df.columns):
        return None

    ids_a = df["course_a_id"].astype(str)
    ids_b = df["course_b_id"].astype(str)
    unique_ids = np.array(sorted(set(ids_a) | set(ids_b)))

    if len(unique_ids) < 8:
        return None

    splitter = GroupShuffleSplit(
        n_splits=600,
        test_size=test_size,
        random_state=random_state,
    )

    best = None
    dummy_x = np.zeros(len(unique_ids))

    for train_course_idx, test_course_idx in splitter.split(
        dummy_x, groups=unique_ids
    ):
        train_courses = set(unique_ids[train_course_idx])
        test_courses = set(unique_ids[test_course_idx])

        train_idx = np.flatnonzero(
            ids_a.isin(train_courses).to_numpy()
            & ids_b.isin(train_courses).to_numpy()
        )
        test_idx = np.flatnonzero(
            ids_a.isin(test_courses).to_numpy()
            & ids_b.isin(test_courses).to_numpy()
        )

        if len(train_idx) < 10 or len(test_idx) < 10:
            continue

        y_train = df.iloc[train_idx]["label"].to_numpy(dtype=int)
        y_test = df.iloc[test_idx]["label"].to_numpy(dtype=int)

        if len(np.unique(y_train)) != 2 or len(np.unique(y_test)) != 2:
            continue

        test_counts = np.bincount(y_test, minlength=2)
        train_counts = np.bincount(y_train, minlength=2)

        # Prefer both classes represented evenly in test, then a larger test set.
        score = (
            int(min(test_counts)),
            int(min(train_counts)),
            int(len(test_idx)),
        )

        if best is None or score > best[0]:
            best = (score, train_idx, test_idx, len(train_courses), len(test_courses))

    if best is None:
        return None

    _, train_idx, test_idx, n_train_courses, n_test_courses = best
    return (
        train_idx,
        test_idx,
        n_train_courses,
        n_test_courses,
    )


def _split_indices(df, test_size=0.25, random_state=42):
    """Choose the safest available split and explain the choice."""
    y = df["label"].to_numpy(dtype=int)

    # Preferred split for the supplied 595-row dataset: disjoint course IDs.
    course_split = _course_holdout_split(
        df, test_size=test_size, random_state=random_state
    )
    if course_split is not None:
        train_idx, test_idx, n_train_courses, n_test_courses = course_split
        excluded = len(df) - len(train_idx) - len(test_idx)
        note = (
            "Course-held-out evaluation: no course ID appears in both sets. "
            f"{n_train_courses} courses in training, {n_test_courses} in testing. "
            f"{excluded} cross-partition pairs were excluded to prevent course leakage."
        )
        return train_idx, test_idx, note

    # Alternative: use a supplied group/domain column if it has enough groups.
    group_col = next(
        (
            col
            for col in ["source_group", "group", "domain", "course_group"]
            if col in df.columns and df[col].nunique() >= 4
        ),
        None,
    )

    if group_col:
        splitter = GroupShuffleSplit(
            n_splits=100,
            test_size=test_size,
            random_state=random_state,
        )
        groups = df[group_col].fillna("unknown").astype(str).to_numpy()

        for train_idx, test_idx in splitter.split(
            np.zeros(len(y)), y, groups
        ):
            if (
                len(np.unique(y[train_idx])) == 2
                and len(np.unique(y[test_idx])) == 2
            ):
                return (
                    train_idx,
                    test_idx,
                    f"Group-based split using '{group_col}'.",
                )

    # Last-resort fallback for CSVs without usable group identifiers.
    train_idx, test_idx = train_test_split(
        np.arange(len(y)),
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )
    return (
        train_idx,
        test_idx,
        "WARNING: stratified pair-row split used because course/group IDs "
        "were unavailable. Repeated syllabus content may occur in both sets.",
    )


def evaluate_dataframe(df, encoder):
    """Train on the training partition and evaluate on the held-out test set.

    Return values preserve the interface expected by the Streamlit app:
        metrics_dataframe, confusion_matrices, split_note, n_train, n_test
    """
    clean = validate_pair_dataframe(df)
    X = build_feature_matrix(clean, encoder)
    y = clean["label"].to_numpy(dtype=int)

    train_idx, test_idx, split_note = _split_indices(clean)

    if len(train_idx) < 3 or len(test_idx) < 2:
        raise ValueError(
            "Not enough training/testing pairs after splitting. "
            "Add more labelled syllabus pairs or more distinct course IDs."
        )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X[train_idx])
    X_test = scaler.transform(X[test_idx])

    model = classifier_set()["SVM"]
    model.fit(X_train, y[train_idx])

    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, 1]

    auc = (
        roc_auc_score(y[test_idx], probabilities)
        if len(np.unique(y[test_idx])) == 2
        else np.nan
    )

    metrics = pd.DataFrame(
        [
            {
                "Model": "Linear SVM",
                "Accuracy": accuracy_score(y[test_idx], predictions),
                "Precision": precision_score(
                    y[test_idx], predictions, zero_division=0
                ),
                "Recall": recall_score(
                    y[test_idx], predictions, zero_division=0
                ),
                "F1-score": f1_score(
                    y[test_idx], predictions, zero_division=0
                ),
                "ROC-AUC": auc,
            }
        ]
    )

    matrix = pd.DataFrame(
        confusion_matrix(y[test_idx], predictions, labels=[0, 1]),
        index=["Actual Different", "Actual Similar"],
        columns=["Predicted Different", "Predicted Similar"],
    )

    return metrics, {"SVM": matrix}, split_note, len(train_idx), len(test_idx)


def evaluate_models(encoder, csv_path=DEFAULT_CSV):
    """Evaluate the configured training CSV; used by the Model Comparison tab."""
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(
            f"Evaluation CSV not found: {csv_path}. "
            "Place the dataset at data/syllabus_pairs.csv."
        )
    source_df = pd.read_csv(csv_path, encoding="utf-8-sig")
    return evaluate_dataframe(source_df, encoder)


def train_and_save(
    encoder=None,
    csv_path=DEFAULT_CSV,
    model_path=DEFAULT_MODEL_PATH,
):
    """Evaluate on held-out courses, then fit and save the final all-data model."""
    if encoder is None:
        encoder = load_encoder()

    metrics, matrices, split_note, n_train, n_test = evaluate_models(
        encoder, csv_path
    )
    models, scaler, X, y, source_df = train_models(encoder, csv_path)

    model_path = Path(model_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)

    bundle = {
        "model": models["SVM"],
        "scaler": scaler,
        "feature_names": FEATURE_NAMES,
        "training_rows": int(len(source_df)),
        "class_counts": {
            "different_0": int((y == 0).sum()),
            "similar_1": int((y == 1).sum()),
        },
        "trained_at": datetime.now().isoformat(timespec="seconds"),
        "dataset_path": str(Path(csv_path).resolve()),
    }
    joblib.dump(bundle, model_path)

    return metrics, matrices, split_note, n_train, n_test, model_path


if __name__ == "__main__":
    print("Loading MiniLM encoder...")
    encoder = load_encoder()

    print(f"Reading dataset: {DEFAULT_CSV}")
    metrics, matrices, split_note, n_train, n_test, saved_path = train_and_save(
        encoder=encoder
    )

    print("\nEvaluation results (held-out test set):")
    print(metrics.to_string(index=False))
    print(f"\nTraining pairs: {n_train}")
    print(f"Testing pairs: {n_test}")
    print(f"Split method: {split_note}")
    print("\nConfusion matrix:")
    print(matrices["SVM"])
    print(f"\nFinal SVM retrained on all labelled rows and saved to:\n{saved_path}")
