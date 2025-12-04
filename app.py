import os
from typing import Optional

import streamlit as st

from orchestrator import run_bio_paper_council, build_report_text
from paper_processing import extract_text_from_pdf, prepare_paper_context
from llm_client import get_openrouter_api_key, MODEL_REGISTRY


APP_TITLE = "🧪 Biological Paper Critic – Multi-Model LLM Peer Reviewer"


def get_api_key_status() -> bool:
    """Return True if OPENROUTER_API_KEY is set."""
    return bool(get_openrouter_api_key(raise_on_missing=False))


def main() -> None:
    st.set_page_config(page_title="Biological Paper Critic", layout="wide")

    if "council_result" not in st.session_state:
        st.session_state.council_result = None
    if "report_text" not in st.session_state:
        st.session_state.report_text = None

    # Title and description
    st.title(APP_TITLE)
    st.markdown(
        "This app helps you critically review biology/biotech research papers using multiple LLM roles "
        "plus a final judge model. **It is an assistive tool and does *not* replace real expert peer review.**"
    )

    # Sidebar configuration
    with st.sidebar:
        st.header("Settings")

        api_key_present = get_api_key_status()
        if api_key_present:
            st.success("`OPENROUTER_API_KEY` detected.")
        else:
            st.error("`OPENROUTER_API_KEY` not found. Set it in your environment before running reviews.")

        skip_stats = st.checkbox("Skip statistics review", value=False)
        short_run = st.checkbox("Short run (more aggressive truncation)", value=False)

        st.markdown("### Paper type")
        paper_type_choice = st.selectbox(
            "Detected paper type (you can override)",
            options=[
                "Auto-detect",
                "primary_experimental",
                "methods",
                "computational",
                "review",
                "clinical",
                "meta_analysis",
            ],
            index=0,
            help="By default the app auto-detects paper type; choose a value to override.",
        )

        st.markdown("### Models per role (read-only)")
        for role_key, model_name in MODEL_REGISTRY.items():
            st.text(f"{role_key}: {model_name}")

    st.markdown("### Input")

    uploaded_pdf = st.file_uploader(
        "Upload PDF of your paper", type=["pdf"], help="If both PDF and text are provided, the PDF will be used."
    )

    text_fallback = st.text_area(
        "Or paste your paper text / abstract here",
        height=250,
        help="Used only if no PDF is uploaded.",
    )

    run_review = st.button("Run Review", type="primary")

    if run_review:
        if not get_api_key_status():
            st.error("Cannot run review: `OPENROUTER_API_KEY` is missing.")
            return

        raw_text: Optional[str] = None
        source_label: str = ""

        if uploaded_pdf is not None:
            try:
                raw_text = extract_text_from_pdf(uploaded_pdf)
                source_label = "PDF upload"
            except Exception as e:  # noqa: BLE001
                st.error(f"Failed to extract text from PDF: {e}")
                return
        elif text_fallback.strip():
            raw_text = text_fallback
            source_label = "Pasted text"
        else:
            st.error("Please upload a PDF or paste your paper text before running the review.")
            return

        max_chars = 15000 if short_run else 30000

        with st.spinner("Processing paper and running expert council..."):
            prepared_text = prepare_paper_context(raw_text, max_chars=max_chars)
            council_result = run_bio_paper_council(
                paper_text=prepared_text,
                original_source=source_label,
                skip_stats=skip_stats,
                override_paper_type=None if paper_type_choice == "Auto-detect" else paper_type_choice,
            )

        st.session_state.council_result = council_result
        st.session_state.report_text = build_report_text(council_result)

    # Output section
    council_result = st.session_state.get("council_result")
    report_text = st.session_state.get("report_text")

    if council_result:
        st.markdown("### Review Result")

        metadata = council_result.get("metadata") or {}
        classifier_info = metadata.get("classifier") or {}
        detected_type = metadata.get("paper_type", "primary_experimental")
        detected_conf = classifier_info.get("confidence")
        detected_notes = classifier_info.get("notes", "")

        st.markdown(
            f"**Paper type:** `{detected_type}`"
            + (
                f"  •  **Classifier confidence:** {detected_conf:.2f}"
                if isinstance(detected_conf, (int, float))
                else ""
            )
        )
        if detected_notes:
            with st.expander("Paper type classification notes", expanded=False):
                st.markdown(detected_notes)

        judge = council_result.get("judge", {})
        judge_content = judge.get("content", "")
        judge_error = judge.get("error")

        if judge_error:
            st.error(f"Judge step failed: {judge_error}")
        else:
            verdict = council_result.get("verdict", "Unknown verdict")
            reason = council_result.get("verdict_reason", "")

            verdict_color = {
                "Accept": "green",
                "Minor revision": "orange",
                "Major revision": "red",
                "Reject": "red",
            }.get(verdict, "gray")

            st.markdown(
                f"<div style='border:1px solid {verdict_color}; padding:0.75rem; border-radius:0.5rem;'>"
                f"<strong>Verdict:</strong> <span style='color:{verdict_color}'>{verdict}</span><br/>"
                f"<strong>Reason:</strong> {reason or 'See judge review below for details.'}"
                "</div>",
                unsafe_allow_html=True,
            )

        experts = council_result.get("experts", {})

        with st.expander("Paper summary", expanded=True):
            summary = experts.get("summariser", {})
            if summary.get("error"):
                st.error(summary["error"])
            else:
                st.markdown(summary.get("content", "No summary available."))

        with st.expander("Methods review", expanded=False):
            methods = experts.get("methods_reviewer", {})
            if methods.get("error"):
                st.error(methods["error"])
            else:
                st.markdown(methods.get("content", "No methods review available."))

        with st.expander("Statistics review", expanded=False):
            stats_info = experts.get("stats_checker")
            if stats_info is None and council_result.get("stats_skipped"):
                st.info("Statistics review was skipped.")
            elif stats_info and stats_info.get("error"):
                st.error(stats_info["error"])
            elif stats_info:
                st.markdown(stats_info.get("content", "No statistics review available."))
            else:
                st.info("No statistics review available.")

        with st.expander("Suggested next experiments", expanded=False):
            next_exp = experts.get("next_experiments_designer", {})
            if next_exp.get("error"):
                st.error(next_exp["error"])
            else:
                st.markdown(next_exp.get("content", "No experiments suggestions available."))

        with st.expander("Judge's full review", expanded=False):
            if judge_error:
                st.error(judge_error)
            else:
                st.markdown(judge_content or "No judge review available.")

        st.markdown("### Model usage")
        model_info_lines = []
        for role_id, info in experts.items():
            model_name = info.get("model", "unknown")
            model_info_lines.append(f"- **{role_id}** → `{model_name}`")
        judge_model = judge.get("model")
        if judge_model:
            model_info_lines.append(f"- **judge** → `{judge_model}`")
        if model_info_lines:
            st.markdown("\n".join(model_info_lines))

        if report_text:
            st.markdown("### Download full report")
            st.download_button(
                label="Download review as Markdown",
                data=report_text,
                file_name="biological_paper_critic_review.md",
                mime="text/markdown",
            )


if __name__ == "__main__":
    main()


