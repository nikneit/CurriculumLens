"""Pair comparison, SVM prediction, topic/outcome alignment and novelty."""
import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.metrics.pairwise import cosine_similarity
from config import THRESHOLD
from minilm_engine import pair_features


def align_items(proposed_items, existing_items, encoder):
    if not proposed_items or not existing_items:
        return [], list(proposed_items)
    A = encoder.encode(proposed_items, normalize_embeddings=True)
    B = encoder.encode(existing_items, normalize_embeddings=True)
    sim = cosine_similarity(A, B)
    rows, cols = linear_sum_assignment(-sim)
    pairs = [{"Proposed": proposed_items[i], "Existing": existing_items[j],
              "Similarity": float(sim[i, j])} for i, j in zip(rows, cols)]
    matched = {i for i, j in zip(rows, cols) if sim[i, j] >= THRESHOLD}
    return pairs, [x for i, x in enumerate(proposed_items) if i not in matched]


def compare_course(proposed, existing, encoder, models, scaler):
    features = pair_features(proposed, existing, encoder).reshape(1, -1)
    scaled = scaler.transform(features)
    model = models["SVM"]
    label = int(model.predict(scaled)[0])
    probability = float(model.predict_proba(scaled)[0][1])
    predictions = {
        "SVM": {
            "prediction": "Similar" if label else "Different",
            "probability": probability
        }
    }
    topic_pairs, new_topics = align_items(proposed.get("topics", []), existing.get("topics", []), encoder)
    outcome_pairs, new_outcomes = align_items(proposed.get("outcomes", []), existing.get("outcomes", []), encoder)
    topic_coverage = sum(p["Similarity"] >= THRESHOLD for p in topic_pairs) / max(1, len(proposed.get("topics", [])))
    outcome_coverage = sum(p["Similarity"] >= THRESHOLD for p in outcome_pairs) / max(1, len(proposed.get("outcomes", [])))
    similarity = float(features[0, 4])
    novelty = float(1 - topic_coverage)
    rank = 100 * (0.35 * similarity + 0.35 * outcome_coverage + 0.30 * topic_coverage)
    if similarity >= 0.75:
        tagline = "Strong thematic overlap"
    elif similarity >= 0.55:
        tagline = "Moderate thematic overlap"
    elif similarity >= 0.35:
        tagline = "Limited thematic overlap"
    else:
        tagline = "Distinct curriculum focus"
    return {
        "course": existing.get("title", "Untitled"), "code": existing.get("code", ""),
        "similarity": similarity, "topic_coverage": topic_coverage,
        "outcome_coverage": outcome_coverage, "novelty": novelty, "rank": rank,
        "tagline": tagline, "predictions": predictions, "topic_pairs": topic_pairs,
        "outcome_pairs": outcome_pairs, "new_topics": new_topics, "new_outcomes": new_outcomes
    }
