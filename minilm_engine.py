"""MiniLM sentence embeddings and syllabus-pair feature extraction."""
import re
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from config import MODEL_NAME

def load_encoder():
    return SentenceTransformer(MODEL_NAME)

def course_text(course):
    return " ".join([course.get("title", ""), course.get("description", ""),
                     " ".join(course.get("outcomes", [])), " ".join(course.get("topics", []))]).strip()

def pair_features(a, b, encoder):
    a_parts = [a.get("description", "") or a.get("title", ""), *a.get("outcomes", []), *a.get("topics", [])]
    b_parts = [b.get("description", "") or b.get("title", ""), *b.get("outcomes", []), *b.get("topics", [])]
    a_parts = [x for x in a_parts if x.strip()] or [a.get("title", "")]
    b_parts = [x for x in b_parts if x.strip()] or [b.get("title", "")]
    ea = encoder.encode(a_parts, normalize_embeddings=True)
    eb = encoder.encode(b_parts, normalize_embeddings=True)
    sim = cosine_similarity(ea, eb)
    da = encoder.encode([a.get("description", "") or a.get("title", "")], normalize_embeddings=True)
    db = encoder.encode([b.get("description", "") or b.get("title", "")], normalize_embeddings=True)
    desc_sim = float(cosine_similarity(da, db)[0, 0])
    topic_start_a = 1 + len(a.get("outcomes", []))
    topic_start_b = 1 + len(b.get("outcomes", []))
    # Compare the proposed course's topic/outcome segments to all segments of the catalog course.
    topic_rows = sim[topic_start_a:] if len(a.get("topics", [])) else sim
    outcome_rows = sim[1:topic_start_a] if len(a.get("outcomes", [])) else sim
    topic_sim = float(np.mean(np.max(topic_rows, axis=1))) if len(topic_rows) else float(np.mean(np.max(sim, axis=1)))
    out_sim = float(np.mean(np.max(outcome_rows, axis=1))) if len(outcome_rows) else float(np.mean(np.max(sim, axis=1)))
    wa = set(re.findall(r"[a-z0-9]+", course_text(a).lower()))
    wb = set(re.findall(r"[a-z0-9]+", course_text(b).lower()))
    jaccard = len(wa & wb) / max(1, len(wa | wb))
    overall = float(np.mean(np.max(sim, axis=1)))
    return np.array([desc_sim, topic_sim, out_sim, jaccard, overall], dtype=float)
