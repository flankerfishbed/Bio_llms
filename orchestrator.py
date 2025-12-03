from __future__ import annotations

from dataclasses import dataclass, asdict
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


def _run_role(role: RoleConfig, paper_text: str) -> ExpertResult:
    """Run a single expert-style role on the paper text."""
    model_name = get_model_for_key(role.model_key)
    messages = [
        {"role": "system", "content": role.system_prompt},
        {
            "role": "user",
            "content": (
                "Here is the text of a biology/biotech research paper. "
                "Perform your assigned role on this text.\n\n"
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
) -> Dict[str, Any]:
    """
    Run the multi-model council plus judge on the given paper text.

    Returns a serialisable dict suitable for Streamlit session state and UI.
    """
    experts: Dict[str, ExpertResult] = {}

    # Sanity check: EXPERT_ROLES should contain a statistics role with id 'stats_checker'.
    has_stats_role = any(r.id == "stats_checker" for r in EXPERT_ROLES)

    for role in EXPERT_ROLES:
        if skip_stats and role.id == "stats_checker":
            continue
        expert_result = _run_role(role, paper_text)
        experts[role.id] = expert_result

    judge_result = _run_judge(
        paper_text=paper_text,
        expert_results=experts,
        stats_skipped=skip_stats,
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
        metadata={"has_stats_role": has_stats_role},
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


