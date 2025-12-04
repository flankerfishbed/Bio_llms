from __future__ import annotations

from dataclasses import dataclass, asdict
import json
from typing import Any, Dict, List, Optional, Tuple

from council_roles import EXPERT_ROLES, JUDGE_ROLE, RoleConfig
from llm_client import call_openrouter, get_model_for_key


@dataclass
class ExpertResult:
    role_id: str
    content: str
    model: str
    error: Optional[str] = None
    usage: Optional[Dict[str, Any]] = None


@dataclass
class JudgeResult:
    content: str
    model: str
    error: Optional[str] = None
    usage: Optional[Dict[str, Any]] = None


@dataclass
class CouncilResult:
    paper_text: str
    original_source: str
    experts: Dict[str, ExpertResult]
    judge: Optional[JudgeResult]
    verdict: str
    verdict_reason: str
    stats_skipped: bool = False
    # Optional free-form metadata about the run.
    metadata: Optional[Dict[str, Any]] = None


ALLOWED_PAPER_TYPES = {
    "primary_experimental",
    "methods",
    "computational",
    "review",
    "clinical",
    "meta_analysis",
}


def classify_paper_type(paper_text: str) -> Dict[str, Any]:
    """
    Classify the paper into a coarse type using an LLM.

    Returns a dict with keys:
      - paper_type: one of ALLOWED_PAPER_TYPES (default: 'primary_experimental')
      - confidence: float in [0, 1]
      - notes: free-text explanation
    """
    model_name = get_model_for_key("BIO_CLASSIFIER_MODEL")

    system_prompt = (
        "You are a biological and biomedical manuscript classifier.\n"
        "Given the text of a research paper, classify it into exactly one of the "
        "following types:\n"
        "- primary_experimental\n"
        "- methods\n"
        "- computational\n"
        "- review\n"
        "- clinical\n"
        "- meta_analysis\n\n"
        "Respond in strict JSON with the following keys:\n"
        "{\n"
        '  "paper_type": "<one of the labels above>",\n'
        '  "confidence": <float between 0 and 1>,\n'
        '  "notes": "<short explanation of your reasoning>"\n'
        "}\n"
    )

    messages = [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": (
                "Classify the following biology/biotech manuscript:\n\n"
                f"{paper_text}"
            ),
        },
    ]

    result = call_openrouter(model=model_name, messages=messages)
    if not isinstance(result, dict):
        return {
            "paper_type": "primary_experimental",
            "confidence": 0.0,
            "notes": "Classification failed: no response from OpenRouter.",
        }
    if result.get("error"):
        return {
            "paper_type": "primary_experimental",
            "confidence": 0.0,
            "notes": f"Classification failed: {result['error']}",
        }

    raw_text = result.get("text", "") or ""
    paper_type = "primary_experimental"
    confidence = 0.5
    notes = ""

    try:
        parsed = json.loads(raw_text)
        if isinstance(parsed, dict):
            pt = str(parsed.get("paper_type", "")).strip().lower()
            if pt in ALLOWED_PAPER_TYPES:
                paper_type = pt
            conf = parsed.get("confidence")
            if isinstance(conf, (int, float)):
                confidence = float(conf)
                if confidence < 0.0 or confidence > 1.0:
                    confidence = max(0.0, min(confidence, 1.0))
            notes_val = parsed.get("notes")
            if isinstance(notes_val, str):
                notes = notes_val.strip()
    except json.JSONDecodeError:
        notes = f"Could not parse classifier JSON. Raw model output: {raw_text[:500]}"

    return {
        "paper_type": paper_type,
        "confidence": confidence,
        "notes": notes,
    }


def _get_type_role_instruction(paper_type: str, role_id: str) -> str:
    """
    Additional guidance for a role based on the paper type.
    Kept simple so _run_role can remain stateless and parallelisable later.
    """
    pt = paper_type.lower()

    if pt in {"methods", "computational"}:
        if role_id == "methods_reviewer":
            return (
                "This manuscript focuses on methods/algorithms. Emphasise validation, "
                "benchmarking, and robustness of the proposed methods rather than "
                "traditional wet-lab controls.\n"
            )
        if role_id == "stats_checker":
            return (
                "Focus on evaluation metrics, statistical soundness of benchmarks, and "
                "comparisons between methods.\n"
            )
    if pt == "review":
        if role_id == "methods_reviewer":
            return (
                "Treat 'methods' as how well the review covers, organises, and critically "
                "analyses existing methods in the field. Do not penalise for lack of "
                "original experimental protocols.\n"
            )
        if role_id == "stats_checker":
            return (
                "If statistics are present, assess them; otherwise focus on how the review "
                "summarises statistical evidence from prior work without penalising for "
                "lack of new datasets.\n"
            )
    if pt in {"clinical", "meta_analysis"}:
        if role_id == "methods_reviewer":
            return (
                "Pay particular attention to inclusion/exclusion criteria, patient "
                "populations, endpoints, and sources of bias.\n"
            )
        if role_id == "stats_checker":
            return (
                "Focus on clinical/statistical endpoints, handling of confounders, and "
                "choices of effect-size or meta-analytic models.\n"
            )

    return ""


def get_roles_for_paper_type(paper_type: str) -> List[RoleConfig]:
    """
    Map paper type to the list of expert roles to run.

    - primary_experimental: run all expert roles.
    - methods / computational: run all roles, with type-specific instructions.
    - review: run summariser, methods reviewer, and next-experiments designer
      (statistics may be less central).
    - clinical / meta_analysis: run all roles.
    """
    pt = (paper_type or "").lower()

    if pt == "review":
        return [r for r in EXPERT_ROLES if r.id in {"summariser", "methods_reviewer", "next_experiments_designer"}]

    # Default: all expert roles.
    return list(EXPERT_ROLES)


def _run_role(role: RoleConfig, paper_text: str, paper_type: str) -> ExpertResult:
    """Run a single expert-style role on the paper text."""
    model_name = get_model_for_key(role.model_key)
    type_instruction = _get_type_role_instruction(paper_type, role.id)

    messages = [
        {"role": "system", "content": role.system_prompt},
        {
            "role": "user",
            "content": (
                "Here is the text of a biology/biotech research paper. "
                f"The paper has been classified as: {paper_type.upper()}.\n"
                "Perform your assigned role on this text using criteria appropriate to this "
                "paper type.\n\n"
                f"{type_instruction}"
                f"{paper_text}"
            ),
        },
    ]

    result = call_openrouter(model=model_name, messages=messages)
    if not isinstance(result, dict):
        return ExpertResult(
            role_id=role.id,
            content="",
            model=model_name,
            error="No response from OpenRouter.",
            usage=None,
        )

    return ExpertResult(
        role_id=role.id,
        content=result.get("text", "") or "",
        model=result.get("model", model_name),
        error=result.get("error"),
        usage=result.get("usage"),
    )


def _run_judge(
    paper_text: str,
    expert_results: Dict[str, ExpertResult],
    stats_skipped: bool,
    paper_type: str,
) -> JudgeResult:
    """Run the judge model over the paper text and expert outputs."""
    model_name = get_model_for_key(JUDGE_ROLE.model_key)

    # Build a structured summary of expert outputs for the judge.
    expert_sections: List[str] = []
    for role_id, expert in expert_results.items():
        header = role_id.replace("_", " ").title()
        if expert.error:
            body = f"[ERROR] {expert.error}"
        else:
            body = expert.content or "[No content returned.]"
        expert_sections.append(f"### {header}\n{body}")

    if stats_skipped and "stats_checker" not in expert_results:
        expert_sections.append("### Statistics Checker\n[Skipped at user request.]")

    expert_block = "\n\n".join(expert_sections)

    user_content = (
        "You are the editorial Judge in a biology/biotech peer-review process.\n\n"
        f"The paper has been classified as: {paper_type.upper()} article. "
        "Please evaluate it using criteria appropriate for this paper type.\n\n"
        "You will receive the (possibly truncated) paper text, followed by outputs from several expert roles.\n"
        "Use this information to write a final, coherent review and a clear verdict.\n\n"
        "=== PAPER TEXT START ===\n"
        f"{paper_text}\n"
        "=== PAPER TEXT END ===\n\n"
        "=== EXPERT OUTPUTS START ===\n"
        f"{expert_block}\n"
        "=== EXPERT OUTPUTS END ==="
    )

    messages = [
        {"role": "system", "content": JUDGE_ROLE.system_prompt},
        {"role": "user", "content": user_content},
    ]

    result = call_openrouter(model=model_name, messages=messages)
    if not isinstance(result, dict):
        return JudgeResult(
            content="",
            model=model_name,
            error="No response from OpenRouter.",
            usage=None,
        )

    return JudgeResult(
        content=result.get("text", "") or "",
        model=result.get("model", model_name),
        error=result.get("error"),
        usage=result.get("usage"),
    )


def _extract_verdict_and_reason(judge_text: str) -> Tuple[str, str]:
    """
    Heuristically extract the verdict label and a short reason from the judge text.

    Strategy:
    - Prefer a dedicated verdict line starting with a label like "Verdict:", "Final verdict:", or "Decision:".
    - If not found, fall back to scanning any line containing an allowed verdict token.
    - For the reason, prefer text under a heading mentioning "reason".
    - Otherwise, derive the reason from sentences immediately following the verdict line.
    """
    verdict_options = ["Accept", "Minor revision", "Major revision", "Reject"]
    verdict = "Unknown verdict"
    reason = ""

    lines = [line.strip() for line in judge_text.splitlines() if line.strip()]

    # Helper to find a verdict token in a line.
    def find_verdict_in_line(line: str) -> Optional[str]:
        lower = line.lower()
        for candidate in verdict_options:
            if candidate.lower() in lower:
                return candidate
        return None

    verdict_line_index: Optional[int] = None

    # 1) Prefer explicit verdict lines with a clear prefix.
    verdict_prefixes = ("verdict:", "final verdict:", "decision:")
    for idx, line in enumerate(lines):
        lower = line.lower()
        if any(lower.startswith(pref) for pref in verdict_prefixes):
            candidate = find_verdict_in_line(line)
            if candidate:
                verdict = candidate
                verdict_line_index = idx
                break

    # 2) Fallback: scan any line for a verdict token.
    if verdict_line_index is None and verdict == "Unknown verdict":
        for idx, line in enumerate(lines):
            candidate = find_verdict_in_line(line)
            if candidate:
                verdict = candidate
                verdict_line_index = idx
                break

    # Reason extraction
    text_for_reason = ""

    # Prefer a dedicated "reason" heading.
    reason_start_index: Optional[int] = None
    for i, line in enumerate(lines):
        lower = line.lower()
        if "reason for verdict" in lower or lower.startswith("reason:"):
            reason_start_index = i + 1
            break

    if reason_start_index is not None:
        # Collect lines until the next likely heading or end.
        collected: List[str] = []
        for line in lines[reason_start_index:]:
            if line.startswith("#"):
                break
            if line.endswith(":") and len(line.split()) <= 6:
                # Likely another section heading, stop.
                break
            collected.append(line)
        text_for_reason = " ".join(collected).strip()
    else:
        # No explicit reason heading: take text after the verdict line, if known.
        if verdict_line_index is not None and verdict_line_index + 1 < len(lines):
            tail_lines = lines[verdict_line_index + 1 :]
        else:
            tail_lines = lines
        text_for_reason = " ".join(tail_lines).strip()

    # Simple sentence split from the selected region.
    sentences = [s.strip() for s in text_for_reason.replace("\n", " ").split(".") if s.strip()]
    if sentences:
        reason = ". ".join(sentences[:3])
        if not reason.endswith("."):
            reason += "."

    return verdict, reason


def run_bio_paper_council(
    paper_text: str,
    original_source: str = "",
    skip_stats: bool = False,
    override_paper_type: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Run the multi-model council plus judge on the given paper text.

    Returns a serialisable dict suitable for Streamlit session state and UI.
    """
    # First classify the paper type.
    classifier_result = classify_paper_type(paper_text)
    detected_type = classifier_result.get("paper_type") or "primary_experimental"
    if detected_type not in ALLOWED_PAPER_TYPES:
        detected_type = "primary_experimental"

    effective_type = (override_paper_type or detected_type).strip().lower()
    if effective_type not in ALLOWED_PAPER_TYPES:
        effective_type = detected_type

    roles_to_run = get_roles_for_paper_type(effective_type)

    experts: Dict[str, ExpertResult] = {}

    # Sanity check: roles should contain a statistics role with id 'stats_checker', if used.
    has_stats_role = any(r.id == "stats_checker" for r in roles_to_run)

    for role in roles_to_run:
        if skip_stats and role.id == "stats_checker":
            continue
        expert_result = _run_role(role, paper_text, effective_type)
        experts[role.id] = expert_result

    judge_result = _run_judge(
        paper_text=paper_text,
        expert_results=experts,
        stats_skipped=skip_stats,
        paper_type=effective_type,
    )

    if judge_result.error or not judge_result.content:
        verdict = "Unknown verdict"
        reason = ""
    else:
        verdict, reason = _extract_verdict_and_reason(judge_result.content)

    council = CouncilResult(
        paper_text=paper_text,
        original_source=original_source,
        experts=experts,
        judge=judge_result,
        verdict=verdict,
        verdict_reason=reason,
        stats_skipped=skip_stats,
        metadata={
            "has_stats_role": has_stats_role,
            "paper_type": effective_type,
            "classifier": classifier_result,
        },
    )

    # Convert dataclasses to plain dicts for Streamlit.
    experts_dict = {rid: asdict(res) for rid, res in council.experts.items()}
    judge_dict = asdict(council.judge) if council.judge else None

    return {
        "paper_text": council.paper_text,
        "original_source": council.original_source,
        "experts": experts_dict,
        "judge": judge_dict,
        "verdict": council.verdict,
        "verdict_reason": council.verdict_reason,
        "stats_skipped": council.stats_skipped,
        "metadata": council.metadata,
    }


def build_report_text(council_result: Dict[str, Any]) -> str:
    """
    Build a markdown report combining all expert and judge outputs.
    """
    lines: List[str] = []

    verdict = council_result.get("verdict", "Unknown verdict")
    verdict_reason = council_result.get("verdict_reason", "")

    original_source = council_result.get("original_source")

    lines.append("# Biological Paper Critic – Review Report")
    lines.append("")
    if original_source:
        lines.append(f"Source: {original_source}")
        lines.append("")
    lines.append(f"**Verdict:** {verdict}")
    if verdict_reason:
        lines.append("")
        lines.append(f"**Reason for verdict:** {verdict_reason}")
    lines.append("")

    experts = council_result.get("experts", {})

    summary = experts.get("summariser", {})
    lines.append("## Paper Summary")
    if summary.get("error"):
        lines.append(f"_Error: {summary['error']}_")
    else:
        lines.append(summary.get("content", "No summary available."))
    lines.append("")

    methods = experts.get("methods_reviewer", {})
    lines.append("## Methods Review")
    if methods.get("error"):
        lines.append(f"_Error: {methods['error']}_")
    else:
        lines.append(methods.get("content", "No methods review available."))
    lines.append("")

    stats = experts.get("stats_checker")
    lines.append("## Statistics Review")
    if council_result.get("stats_skipped") and stats is None:
        lines.append("_Statistics review was skipped at user request._")
    elif stats and stats.get("error"):
        lines.append(f"_Error: {stats['error']}_")
    elif stats:
        lines.append(stats.get("content", "No statistics review available."))
    else:
        lines.append("No statistics review available.")
    lines.append("")

    next_exp = experts.get("next_experiments_designer", {})
    lines.append("## Suggested Next Experiments")
    if next_exp.get("error"):
        lines.append(f"_Error: {next_exp['error']}_")
    else:
        lines.append(next_exp.get("content", "No suggested experiments available."))
    lines.append("")

    judge = council_result.get("judge", {})
    lines.append("## Judge's Full Review")
    if judge.get("error"):
        lines.append(f"_Judge step failed: {judge['error']}_")
    else:
        lines.append(judge.get("content", "No judge review available."))
    lines.append("")

    # Models used
    lines.append("## Models Used")
    model_lines: List[str] = []
    for role_id, info in experts.items():
        model_name = info.get("model", "unknown")
        model_lines.append(f"- **{role_id}** → `{model_name}`")
    judge_model = judge.get("model")
    if judge_model:
        model_lines.append(f"- **judge** → `{judge_model}`")
    if model_lines:
        lines.extend(model_lines)
    lines.append("")

    lines.append("---")
    lines.append("_Generated by the Biological Paper Critic. This does not replace expert peer review._")

    return "\n".join(lines)


