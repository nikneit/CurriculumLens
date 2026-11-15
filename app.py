"""CurriculumLens: white dashboard for syllabus comparison and curriculum novelty."""
import pathlib
import pandas as pd
import streamlit as st
import plotly.express as px
from config import MODEL_NAME, THRESHOLD, TOP_MATCHES
from database import db_conn, load_courses, add_course, migrate_legacy_courses, save_analysis
from syllabus_utils import parse_syllabus, read_upload, split_lines
from minilm_engine import load_encoder
from model_training import train_models, evaluate_models, evaluate_dataframe
from comparison import compare_course
from reporting import build_pdf

st.set_page_config(page_title="CurriculumLens | Syllabus Intelligence", page_icon="🎓", layout="wide")

st.markdown("""
<style>
:root { color-scheme: light; }
.stApp { background: #f5f7fb; color: #182230; }
[data-testid="stHeader"] { background: rgba(245,247,251,.92); }
[data-testid="stSidebar"] { background: #ffffff; border-right: 1px solid #e4e8f0; }
.block-container { max-width: 1500px; padding-top: 1.3rem; padding-bottom: 3rem; }
h1,h2,h3 { color: #17243a; letter-spacing: -.02em; }
p, label, span, div { }
[data-testid="stMetric"] { background: #ffffff; border: 1px solid #e4e8f0; border-radius: 14px; padding: 16px 18px; box-shadow: 0 2px 8px rgba(24,39,75,.04); }
[data-testid="stMetricLabel"] { color: #66758a; font-size: .78rem; text-transform: uppercase; letter-spacing: .04em; }
[data-testid="stMetricValue"] { color: #17243a; font-weight: 700; }
div[data-testid="stExpander"] { background: #fff; border: 1px solid #e2e7ef; border-radius: 12px; }
.stButton > button[kind="primary"] { background: #304e9b; border: 0; color: white; border-radius: 9px; }
.stButton > button { border-radius: 9px; }
[data-testid="stDataFrame"] { background: white; border-radius: 12px; }
section[data-testid="stSidebar"] .stButton > button { width: 100%; }
.small-muted { color: #718096; font-size: 13px; }
.match-heading { background: #edf1ff; border: 1px solid #dce3ff; padding: 9px 13px; border-radius: 9px; color: #394b91; font-weight: 650; margin: 12px 0; }
.pill { display:inline-block; padding:3px 9px; border-radius:20px; background:#eff8f3; color:#23734c; border:1px solid #ccebd9; font-size:12px; }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def get_encoder():
    return load_encoder()

@st.cache_resource
def get_training(_encoder):
    return train_models(_encoder)

con = db_conn()
migrate_legacy_courses(con)

with st.sidebar:
    st.title("🎓 CurriculumLens")
    st.caption("AI-based syllabus similarity, curriculum comparison and novelty detection")
    st.markdown("**Semantic encoder**")
    st.write("all-MiniLM-L6-v2")
    st.markdown("**Classifiers**")
    st.write("Linear SVM")
    st.caption("MiniLM downloads its pretrained model on first launch. Internet is needed for the initial download.")
    if st.button("Load example course catalog"):
        sample_dir = pathlib.Path("data/sample_syllabi")
        added = 0
        for path in sample_dir.glob("*.txt"):
            item = parse_syllabus(path.read_text(encoding="utf-8"), path.name)
            add_course(con, item, str(path))
            added += 1
        st.success(f"Added {added} example courses.")
        st.rerun()

st.title("CurriculumLens")
st.caption("AI-based syllabus similarity, curriculum comparison and novelty detection")
try:
    with st.spinner("Loading MiniLM and training the Linear SVM..."):
        encoder = get_encoder()
        models, scaler, train_X, train_y, training_df = get_training(encoder)
except Exception as exc:
    st.error("The semantic model or training data could not be loaded. Check requirements, internet access and data/syllabus_pairs.csv.")
    st.exception(exc)
    st.stop()

courses = load_courses(con)
tab_analyze, tab_catalog, tab_models, tab_method = st.tabs(
    ["Analyze", "Catalog", "Model comparison", "Methodology"]
)

with tab_analyze:
    st.header("Analyze a group of syllabi")
    st.caption("Select any number of courses. CurriculumLens produces ONE overall group-comparison result, not a separate result card for every pair.")
    source_mode = st.radio(
        "Choose where your courses come from",
        ["Select courses from catalog", "Upload multiple syllabi"],
        horizontal=True,
        key="proposal_source_mode"
    )

    selected_courses = []
    if source_mode == "Select courses from catalog":
        if courses:
            course_options = {
                f'{c["code"] or "—"}  {c["title"]}  (ID {c["id"]})': c
                for c in courses
            }
            selected_labels = st.multiselect(
                "Select TWO OR MORE courses for one group analysis",
                options=list(course_options.keys()),
                default=[],
                key="selected_catalog_courses",
                placeholder="Choose 2, 3, 4 or more courses..."
            )
            selected_courses = [course_options[label].copy() for label in selected_labels]
            st.caption(f"{len(selected_courses)} course(s) selected.")
        else:
            st.info("The catalog is empty. Add courses in Catalog or load example courses from the sidebar.")
    else:
        uploaded_files = st.file_uploader(
            "Upload TWO OR MORE syllabi (PDF, DOCX or TXT)",
            type=["pdf", "docx", "txt"],
            accept_multiple_files=True,
            key="multiple_proposal_uploads"
        )
        if uploaded_files:
            selected_courses = [
                parse_syllabus(read_upload(file), file.name)
                for file in uploaded_files
            ]
            st.caption(f"{len(selected_courses)} syllabus file(s) uploaded.")

    if selected_courses:
        with st.expander("Review selected courses", expanded=False):
            for idx, course in enumerate(selected_courses):
                st.markdown(f"**{idx + 1}. {course.get('code', '')} — {course.get('title', 'Untitled')}**")
                st.write(course.get("description", ""))
                st.caption(f"Topics: {len(course.get('topics', []))} | Learning outcomes: {len(course.get('outcomes', []))}")

    if len(selected_courses) >= 2:
        if st.button("Run group curriculum analysis", type="primary", key="run_curriculum_analysis"):
            import itertools
            pair_results = []
            for course_a, course_b in itertools.combinations(selected_courses, 2):
                pair_result = compare_course(course_a, course_b, encoder, models, scaler)
                pair_result["course_a"] = course_a.get("title", "Course A")
                pair_result["course_b"] = course_b.get("title", "Course B")
                pair_result["course"] = f'{pair_result["course_a"]} vs {pair_result["course_b"]}'
                pair_results.append(pair_result)

            # Aggregate the internal pair calculations into ONE group-level result.
            # Similarity/coverage/novelty are the mean across all unique course pairs.
            group_similarity = float(sum(r["similarity"] for r in pair_results) / len(pair_results))
            group_topic_coverage = float(sum(r["topic_coverage"] for r in pair_results) / len(pair_results))
            group_outcome_coverage = float(sum(r["outcome_coverage"] for r in pair_results) / len(pair_results))
            group_novelty = float(sum(r["novelty"] for r in pair_results) / len(pair_results))
            group_rank = float(sum(r["rank"] for r in pair_results) / len(pair_results))

            group_predictions = {}
            for model_name in ["SVM"]:
                probabilities = [r["predictions"][model_name]["probability"] for r in pair_results]
                similar_votes = sum(r["predictions"][model_name]["prediction"] == "Similar" for r in pair_results)
                mean_probability = float(sum(probabilities) / len(probabilities))
                group_predictions[model_name] = {
                    "prediction": "Similar" if mean_probability >= 0.5 else "Different",
                    "probability": mean_probability,
                    "similar_pair_percentage": similar_votes / len(pair_results)
                }

            all_topic_pairs = [item for r in pair_results for item in r["topic_pairs"]]
            all_outcome_pairs = [item for r in pair_results for item in r["outcome_pairs"]]
            all_new_topics = list(dict.fromkeys(item for r in pair_results for item in r["new_topics"]))
            all_new_outcomes = list(dict.fromkeys(item for r in pair_results for item in r["new_outcomes"]))

            names = [c.get("title", "Untitled") for c in selected_courses]
            group_result = {
                "course": "Overall selected-course group",
                "code": "",
                "similarity": group_similarity,
                "topic_coverage": group_topic_coverage,
                "outcome_coverage": group_outcome_coverage,
                "novelty": group_novelty,
                "rank": group_rank,
                "tagline": "Strong overall overlap" if group_similarity >= .75 else
                           "Moderate overall overlap" if group_similarity >= .55 else
                           "Limited overall overlap" if group_similarity >= .35 else
                           "Distinct overall curriculum focus",
                "predictions": group_predictions,
                "topic_pairs": all_topic_pairs,
                "outcome_pairs": all_outcome_pairs,
                "new_topics": all_new_topics,
                "new_outcomes": all_new_outcomes,
                "pair_count": len(pair_results),
                "course_count": len(selected_courses),
                "course_names": names,
                "min_pair_similarity": min(r["similarity"] for r in pair_results),
                "max_pair_similarity": max(r["similarity"] for r in pair_results)
            }
            report_proposed = {
                "title": f"Group comparison ({len(selected_courses)} courses)",
                "code": "",
                "description": "Courses included: " + ", ".join(names)
            }
            st.session_state["analysis_result"] = {
                "proposed": report_proposed,
                "results": [group_result],
                "group_result": group_result,
                "pair_results": pair_results,
                "selected_courses": names
            }
            save_analysis(con, report_proposed, [group_result])
    elif selected_courses:
        st.warning("Please select at least TWO courses for group analysis.")

    if "analysis_result" in st.session_state:
        analysis = st.session_state["analysis_result"]
        group = analysis.get("group_result")
        if group:
            st.markdown(f"### Overall result: {group['course_count']} selected courses")
            st.caption("This is one consolidated result for the entire selected group.")
            st.markdown("**Courses included:** " + " · ".join(group["course_names"]))
            m = st.columns(4)
            m[0].metric("Courses included", group["course_count"])
            m[1].metric("Overall similarity", f"{group['similarity']:.0%}")
            m[2].metric("Average topic coverage", f"{group['topic_coverage']:.0%}")
            m[3].metric("Overall novelty", f"{group['novelty']:.0%}")
            st.markdown('<div class="match-heading">Consolidated group comparison</div>', unsafe_allow_html=True)
            st.write(f"**Group rank:** {group['rank']:.1f}/100  ·  **Outcome alignment:** {group['outcome_coverage']:.0%}  ·  **Internal pair calculations:** {group['pair_count']} (combined into this one result)")
            st.caption(f"Pair similarity range used to calculate the group average: {group['min_pair_similarity']:.0%} to {group['max_pair_similarity']:.0%}.")
            st.subheader("Linear SVM result for the selected group")
            model_rows = []
            for name, value in group["predictions"].items():
                model_rows.append({
                    "Model": name,
                    "Overall group prediction": value["prediction"],
                    "Mean Similar probability": f'{value["probability"]:.1%}',
                    "Share of internal pairs predicted Similar": f'{value["similar_pair_percentage"]:.1%}'
                })
            st.dataframe(pd.DataFrame(model_rows), use_container_width=True, hide_index=True)
            st.subheader("Group-level interpretation")
            st.write(f"**Overall similarity:** {group['similarity']:.1%}. This is the mean semantic similarity across all unique pairs in the selected group.")
            st.write(f"**Novelty estimate:** {group['novelty']:.1%}. This is the average uncovered-topic proportion across the pair comparisons.")
            st.write("**How to interpret model results:** each classifier predicts Similar/Different for the internal course pairs; the displayed group prediction is based on the mean Similar probability across those pairs.")
            with st.expander("View combined topic and outcome matching details"):
                left, right = st.columns(2)
                with left:
                    st.markdown("**Topic matches from all internal comparisons**")
                    st.dataframe(pd.DataFrame(group["topic_pairs"]), use_container_width=True, hide_index=True)
                    st.markdown("**Uncovered topics**")
                    st.write(", ".join(group["new_topics"]) or "None identified")
                with right:
                    st.markdown("**Learning outcome matches from all internal comparisons**")
                    st.dataframe(pd.DataFrame(group["outcome_pairs"]), use_container_width=True, hide_index=True)
                    st.markdown("**Uncovered learning outcomes**")
                    st.write(", ".join(group["new_outcomes"]) or "None identified")
            st.download_button(
                "Download group comparison report (PDF)",
                build_pdf(analysis["proposed"], [group]),
                file_name="CurriculumLens_Group_Comparison_Report.pdf",
                mime="application/pdf"
            )

            st.markdown("---")
            st.header("One-to-one comparison results for every pair")
            st.caption("Every unique pair from the selected courses is shown below automatically. No pair needs to be selected from a dropdown.")

            pair_results = analysis.get("pair_results", [])
            pair_table = pd.DataFrame([
                {
                    "Course pair": r["course"],
                    "Semantic similarity": f'{r["similarity"]:.1%}',
                    "Topic coverage": f'{r["topic_coverage"]:.1%}',
                    "Outcome alignment": f'{r["outcome_coverage"]:.1%}',
                    "Novelty": f'{r["novelty"]:.1%}',
                    "Linear SVM": r["predictions"]["SVM"]["prediction"]
                }
                for r in pair_results
            ])
            st.dataframe(pair_table, use_container_width=True, hide_index=True)

            for pair_index, pair in enumerate(pair_results, start=1):
                with st.expander(f'{pair_index}. {pair["course"]} | Similarity {pair["similarity"]:.1%}'):
                    p1, p2, p3, p4 = st.columns(4)
                    p1.metric("Semantic similarity", f'{pair["similarity"]:.1%}')
                    p2.metric("Topic coverage", f'{pair["topic_coverage"]:.1%}')
                    p3.metric("Outcome alignment", f'{pair["outcome_coverage"]:.1%}')
                    p4.metric("Novelty", f'{pair["novelty"]:.1%}')
                    st.markdown("**Linear SVM prediction for this exact pair**")
                    pair_model_rows = [
                        {
                            "Model": model_name,
                            "Prediction": values["prediction"],
                            "Similar probability": f'{values["probability"]:.1%}'
                        }
                        for model_name, values in pair["predictions"].items()
                    ]
                    st.dataframe(pd.DataFrame(pair_model_rows), use_container_width=True, hide_index=True)
                    st.caption(f'Pair rank: {pair["rank"]:.1f}/100 · {pair["tagline"]}')
                    with st.expander("View matched topics and learning outcomes"):
                        left, right = st.columns(2)
                        with left:
                            st.markdown("**Topic alignment**")
                            st.dataframe(pd.DataFrame(pair["topic_pairs"]), use_container_width=True, hide_index=True)
                            st.markdown("**Uncovered topics from the first course**")
                            st.write(", ".join(pair["new_topics"]) or "None identified")
                        with right:
                            st.markdown("**Learning-outcome alignment**")
                            st.dataframe(pd.DataFrame(pair["outcome_pairs"]), use_container_width=True, hide_index=True)
                            st.markdown("**Uncovered outcomes from the first course**")
                            st.write(", ".join(pair["new_outcomes"]) or "None identified")

            st.download_button(
                "Download all one-to-one comparison results (PDF)",
                build_pdf(
                    {"title": "All one-to-one course comparisons",
                     "description": "All unique course pairs generated from the selected course group."},
                    pair_results
                ),
                file_name="CurriculumLens_All_Pair_Comparisons.pdf",
                mime="application/pdf",
                key="download_all_pair_reports"
            )

with tab_catalog:
    st.header("Course catalog")
    st.caption(f"{len(courses)} courses currently available for comparison.")
    with st.expander("Add a course manually", expanded=False):
        code = st.text_input("Course code", key="new_code")
        title = st.text_input("Course title", key="new_title")
        desc = st.text_area("Course description", key="new_description")
        outs = st.text_area("Learning outcomes (one per line)", key="new_outcomes")
        tops = st.text_area("Topics (one per line)", key="new_topics")
        if st.button("Save course to catalog"):
            if title.strip():
                add_course(con, {"code":code,"title":title,"description":desc,
                                 "outcomes":split_lines(outs),"topics":split_lines(tops)})
                st.success("Course saved.")
                st.rerun()
            else:
                st.error("Please enter a course title.")
    uploads = st.file_uploader("Import multiple existing syllabi (PDF, DOCX, TXT)", type=["pdf","docx","txt"],
                               accept_multiple_files=True, key="catalog_uploads")
    if uploads and st.button("Import uploaded syllabi"):
        for file in uploads:
            add_course(con, parse_syllabus(read_upload(file), file.name), file.name)
        st.success(f"Imported {len(uploads)} syllabi.")
        st.rerun()
    for course in load_courses(con):
        with st.expander(f'{course["code"] or "—"} — {course["title"]} (ID {course["id"]})'):
            st.write(course["description"])
            st.markdown("**Topics**")
            st.write("\n".join(f"- {x}" for x in course["topics"]) or "Not provided")
            st.markdown("**Learning outcomes**")
            st.write("\n".join(f"- {x}" for x in course["outcomes"]) or "Not provided")
            if st.button("Delete course", key=f'delete_{course["id"]}'):
                con.execute("DELETE FROM courses WHERE id=?", (course["id"],))
                con.commit()
                st.rerun()

with tab_models:
    st.header("Model comparison and evaluation")
    st.info("Evaluate the project's primary classifier, Linear SVM, using either the built-in labelled dataset or your own labelled syllabus-pair CSV. This tab measures classifier performance; it is different from predicting a label for one pair in Analyze.")
    st.markdown("**MiniLM features:** description cosine similarity, topic semantic similarity, outcome semantic similarity, lexical Jaccard overlap and overall semantic similarity.")
    st.markdown("### Evaluation method 1 — Built-in labelled dataset")
    st.caption(f"Dataset rows: {len(training_df)} | Similar: {(training_df.label==1).sum()} | Different: {(training_df.label==0).sum()}. The current labels are provisional draft annotations.")
    if st.button("Evaluate built-in dataset", type="primary", key="evaluate_builtin"):
        with st.spinner("Extracting MiniLM features and evaluating Linear SVM..."):
            metrics, matrices, split_note, n_train, n_test = evaluate_models(encoder)
            st.session_state["model_metrics"] = metrics
            st.session_state["model_matrices"] = matrices
            st.session_state["model_split_note"] = split_note
            st.session_state["model_split_counts"] = (n_train, n_test)
            st.session_state["model_eval_source"] = "Built-in dataset"

    st.markdown("---")
    st.markdown("### Evaluation method 2 — Upload your own labelled CSV")
    st.write("Upload a CSV containing syllabus pairs and their correct labels. The model learns from the training portion and evaluates predictions on the held-out test portion.")
    with st.expander("Required CSV format and example", expanded=True):
        st.code(
            "syllabus_a,syllabus_b,label,domain\n"
            '"AI fundamentals and neural networks","Machine learning and deep learning",1,Computing\n'
            '"Database normalization and SQL","Operating systems and CPU scheduling",0,Computing\n'
            '"Classification and model evaluation","Supervised learning and classifiers",1,Computing\n'
            '"Financial auditing and taxation","Computer network routing and protocols",0,Finance',
            language="csv"
        )
        st.caption("Required columns: syllabus_a, syllabus_b, label. Labels can be 1/0 or Similar/Different. Optional domain or source_group column enables group-based splitting. Include both labels and enough examples (preferably 30+ pairs per class) for meaningful evaluation.")
    uploaded_eval_csv = st.file_uploader(
        "Upload labelled syllabus-pair CSV",
        type=["csv"],
        key="labelled_eval_csv"
    )
    if uploaded_eval_csv is not None:
        try:
            uploaded_df = pd.read_csv(uploaded_eval_csv)
            st.caption(f"Uploaded {len(uploaded_df)} rows. Preview:")
            st.dataframe(uploaded_df.head(5), use_container_width=True, hide_index=True)
            if st.button("Train and evaluate uploaded dataset", type="primary", key="evaluate_uploaded"):
                with st.spinner("Extracting MiniLM features and evaluating Linear SVM..."):
                    metrics, matrices, split_note, n_train, n_test = evaluate_dataframe(uploaded_df, encoder)
                    st.session_state["model_metrics"] = metrics
                    st.session_state["model_matrices"] = matrices
                    st.session_state["model_split_note"] = split_note
                    st.session_state["model_split_counts"] = (n_train, n_test)
                    st.session_state["model_eval_source"] = uploaded_eval_csv.name
                    st.success("Linear SVM evaluation completed on the uploaded labelled dataset.")
        except Exception as exc:
            st.error(f"Could not read the uploaded CSV: {exc}")

    if "model_metrics" in st.session_state:
        metrics = st.session_state["model_metrics"]
        n_train, n_test = st.session_state.get("model_split_counts", (0, 0))
        st.markdown("### Evaluation results")
        st.caption(f"Evaluation source: {st.session_state.get('model_eval_source', 'Dataset')} · Training rows: {n_train} · Test rows: {n_test}")
        st.info(st.session_state.get("model_split_note", ""))
        st.dataframe(metrics.style.format({c:"{:.3f}" for c in metrics.columns if c!="Model"}),
                     use_container_width=True, hide_index=True)
        chart = metrics.melt(id_vars="Model", var_name="Metric", value_name="Score")
        st.plotly_chart(px.bar(chart, x="Metric", y="Score", color="Model", barmode="group",
                               range_y=[0,1], title="Linear SVM performance evaluation"), use_container_width=True)
        st.markdown("### Linear SVM confusion matrix")
        st.markdown("**Linear SVM**")
        st.dataframe(st.session_state["model_matrices"]["SVM"], use_container_width=True)
        st.caption("Confusion matrix rows are actual labels; columns are model predictions. 0 = Different and 1 = Similar.")

    st.warning("Evaluation scores depend on the quality and independence of the labelled examples. The built-in labels are provisional draft annotations, not independently validated expert labels. Do not present their scores as proven real-world accuracy.")

with tab_method:
    st.header("Methodology")
    st.markdown("""
    **1. Syllabus extraction and parsing** — reads PDF, DOCX and TXT content and separates description, learning outcomes and topics.

    **2. MiniLM semantic encoding** — `all-MiniLM-L6-v2` converts text segments into dense numerical embeddings that represent contextual meaning.

    **3. Cosine similarity and feature engineering** — compares embedding vectors and creates five pair-level features for the Linear SVM classifier.

    **4. Linear SVM classification** — a supervised Support Vector Machine learns a maximum-margin decision boundary between Similar and Different syllabus pairs.

    **5. Topic and outcome alignment** — the Hungarian algorithm finds one-to-one high-similarity matches between syllabus items.

    **6. Novelty estimation** — one minus the proportion of proposed topics matched above the configured threshold.

    **7. Evaluation** — Accuracy, Precision, Recall, F1-score and ROC-AUC measure the Linear SVM on a held-out labelled test set.
    """)
    st.markdown("**Important distinction:** MiniLM is the semantic encoder, cosine similarity is a comparison measure, and Linear SVM is the supervised classifier.")
    st.caption(f"Encoder: {MODEL_NAME} | Topic matching threshold: {THRESHOLD:.2f}")
