"""Select Content node — decides editorial strategy before any rewriting.

Applies deterministic suppression rules where possible, then uses LLM
for the emphasis plan. Includes experience-block-level suppression to
keep only the most relevant experiences.
"""

import json
import logging
import re
from collections import defaultdict

from langchain_core.messages import HumanMessage, SystemMessage

from resume_tailor.config import get_config
from resume_tailor.models import (
    EmphasisPlan,
    ExperienceBlock,
    MatchStrength,
    PriorityTier,
    RequirementMapping,
    SuppressionEntry,
)
from resume_tailor.prompts.select_content import (
    SELECT_CONTENT_SYSTEM,
    SELECT_CONTENT_USER,
)
from resume_tailor.state import ResumeState

logger = logging.getLogger(__name__)

# Pattern for detecting metrics/numbers in bullets (%, $, numbers, etc.)
_METRIC_PATTERN = re.compile(r'\d+[%xX]|\$[\d,]+|\d{2,}|#\d+')


def _score_experience_block(
    block: ExperienceBlock,
    evidence_map: list[RequirementMapping],
) -> float:
    """Score an experience block by how well its bullets match JD requirements.

    Scoring:
    - Each strong match to a must_have requirement: 10 points
    - Each strong match to a strong_preference: 6 points
    - Each strong match to a nice_to_have: 3 points
    - Weak matches get half the points
    - Bonus for number of distinct requirements covered
    """
    bullet_texts = {b.text for b in block.bullets}
    score = 0.0
    requirements_covered = set()

    priority_weights = {
        PriorityTier.MUST_HAVE: 10.0,
        PriorityTier.STRONG_PREFERENCE: 6.0,
        PriorityTier.NICE_TO_HAVE: 3.0,
    }
    strength_multiplier = {
        MatchStrength.STRONG: 1.0,
        MatchStrength.WEAK: 0.5,
        MatchStrength.NONE: 0.0,
    }

    for mapping in evidence_map:
        for entry in mapping.evidence:
            if entry.source_bullet in bullet_texts:
                weight = priority_weights.get(mapping.priority, 3.0)
                mult = strength_multiplier.get(entry.match_strength, 0.0)
                score += weight * mult
                if mult > 0:
                    requirements_covered.add(mapping.requirement)

    # Bonus for breadth — covering more distinct requirements is valuable
    score += len(requirements_covered) * 2.0

    return score


def _deterministic_suppressions(
    evidence_map: list[RequirementMapping],
    parsed_resume,
) -> list[SuppressionEntry]:
    """Apply rule-based suppression logic — no LLM needed.

    Operates at two levels:
    1. Experience-block level: rank blocks by relevance, suppress the weakest
    2. Bullet level: within kept blocks, suppress unmatched/duplicate bullets
    """
    config = get_config()
    suppressions = []
    all_bullets = parsed_resume.all_bullets()

    # --- Phase 1: Experience block ranking and suppression ---
    blocks = parsed_resume.experience_blocks
    suppressed_block_ids: set[str] = set()

    if config.max_experiences > 0 and len(blocks) > config.max_experiences:
        # Score each block
        block_scores = []
        for block in blocks:
            score = _score_experience_block(block, evidence_map)
            block_scores.append((block, score))
            logger.debug(
                f"Block '{block.experience_id}' "
                f"({block.company or block.title}): score={score:.1f}"
            )

        # Sort by score descending
        block_scores.sort(key=lambda x: x[1], reverse=True)

        # Keep top N, suppress the rest
        kept = block_scores[:config.max_experiences]
        cut = block_scores[config.max_experiences:]

        kept_ids = {b.experience_id for b, _ in kept}
        logger.info(
            f"Experience selection: keeping {len(kept)}/{len(blocks)} blocks "
            f"(top scores: {[f'{s:.0f}' for _, s in kept]})"
        )

        for block, score in cut:
            suppressed_block_ids.add(block.experience_id)
            for bullet in block.bullets:
                suppressions.append(SuppressionEntry(
                    source_bullet=bullet.text,
                    experience_id=bullet.experience_id,
                    reason=f"experience block suppressed (relevance score: {score:.0f}, "
                           f"below top {config.max_experiences})",
                ))

    # --- Phase 2: Compute per-block bullet targets ---
    # Higher-scoring blocks get more bullets; weaker blocks get fewer.
    kept_blocks = [b for b in blocks if b.experience_id not in suppressed_block_ids]
    if not kept_blocks:
        kept_blocks = blocks  # no suppression happened (fewer than max)

    block_score_map = {b.experience_id: _score_experience_block(b, evidence_map)
                       for b in kept_blocks}
    max_score = max(block_score_map.values()) if block_score_map else 1.0
    max_score = max(max_score, 1.0)  # avoid division by zero

    # target = base + extra bullets scaled by relative score
    # e.g. base=2.5, max_target=5 → top block gets 5, weakest gets ~2
    extra_range = config.max_bullet_target - config.base_bullet_target
    bullet_targets: dict[str, int] = {}
    for exp_id, score in block_score_map.items():
        normalized = score / max_score
        raw_target = config.base_bullet_target + extra_range * normalized
        target = max(config.min_bullets_per_block,
                     min(config.max_bullet_target, round(raw_target)))
        bullet_targets[exp_id] = target

    logger.info(
        f"Bullet targets: "
        + ", ".join(f"{eid}={t}" for eid, t in bullet_targets.items())
    )

    # --- Phase 3: Bullet-level suppression within kept blocks ---

    # Build mapping of bullets → requirements
    mapped_bullets: set[str] = set()
    bullet_to_requirements: dict[str, list[tuple[str, MatchStrength]]] = {}

    for mapping in evidence_map:
        for entry in mapping.evidence:
            mapped_bullets.add(entry.source_bullet)
            if entry.source_bullet not in bullet_to_requirements:
                bullet_to_requirements[entry.source_bullet] = []
            bullet_to_requirements[entry.source_bullet].append(
                (mapping.requirement, entry.match_strength)
            )

    already_suppressed = {s.source_bullet for s in suppressions}

    # Group bullets by experience block for target-aware suppression
    block_bullets: dict[str, list] = defaultdict(list)
    for bullet in all_bullets:
        if bullet.experience_id not in suppressed_block_ids:
            block_bullets[bullet.experience_id].append(bullet)

    for exp_id, exp_bullets in block_bullets.items():
        target = bullet_targets.get(exp_id, config.min_bullets_per_block)

        # Score each bullet for retention priority:
        #   matched strong > matched weak > unmatched with metrics > unmatched
        def _bullet_retention_score(b):
            reqs = bullet_to_requirements.get(b.text, [])
            if reqs:
                best = min(
                    (0 if s == MatchStrength.STRONG else 1)
                    for _, s in reqs
                )
                return best  # 0 = strong match, 1 = weak match
            # Unmatched — prefer bullets with metrics/numbers
            has_metric = bool(_METRIC_PATTERN.search(b.text))
            return 2 if has_metric else 3

        ranked = sorted(exp_bullets, key=_bullet_retention_score)

        # Keep up to target, suppress the rest (worst-ranked first)
        kept_count = 0
        for bullet in ranked:
            if bullet.text in already_suppressed:
                continue
            kept_count += 1
            if kept_count > target:
                suppressions.append(SuppressionEntry(
                    source_bullet=bullet.text,
                    experience_id=bullet.experience_id,
                    reason="below bullet target (low relevance)",
                ))
                already_suppressed.add(bullet.text)

    # Rule: Duplicate evidence (multiple bullets → same requirement, within budget)
    req_to_bullets: dict[str, list[tuple[str, str, MatchStrength]]] = {}
    for mapping in evidence_map:
        for entry in mapping.evidence:
            if entry.experience_id in suppressed_block_ids:
                continue
            key = mapping.requirement
            if key not in req_to_bullets:
                req_to_bullets[key] = []
            req_to_bullets[key].append(
                (entry.source_bullet, entry.experience_id, entry.match_strength)
            )

    for req, bullets in req_to_bullets.items():
        if len(bullets) <= 1:
            continue
        strength_order = {
            MatchStrength.STRONG: 0,
            MatchStrength.WEAK: 1,
            MatchStrength.NONE: 2,
        }
        sorted_bullets = sorted(
            bullets, key=lambda x: strength_order.get(x[2], 2)
        )
        for bullet_text, exp_id, strength in sorted_bullets[1:]:
            if bullet_text not in already_suppressed:
                # Only suppress duplicates if the block is still above minimum
                block_remaining = sum(
                    1 for b in block_bullets.get(exp_id, [])
                    if b.text not in already_suppressed
                )
                if block_remaining <= config.min_bullets_per_block:
                    continue
                other_reqs = [
                    r for r, s in bullet_to_requirements.get(bullet_text, [])
                    if r != req
                ]
                if not other_reqs:
                    suppressions.append(SuppressionEntry(
                        source_bullet=bullet_text,
                        experience_id=exp_id,
                        reason=f"duplicate evidence for: {req}",
                    ))
                    already_suppressed.add(bullet_text)

    return suppressions


def select_content_node(state: ResumeState) -> dict:
    """Decide suppressions and emphasis plan."""
    config = get_config()
    evidence_map = state["evidence_map"]
    parsed_resume = state["parsed_resume"]

    # Step 1: Deterministic suppressions (block-level + bullet-level)
    suppressions = _deterministic_suppressions(evidence_map, parsed_resume)
    logger.info(f"Total suppressions: {len(suppressions)} bullets")

    # Step 2: LLM-driven emphasis plan
    evidence_map_json = json.dumps(
        [m.model_dump() for m in evidence_map], indent=2
    )
    constraints_json = json.dumps(state.get("constraints", {}), indent=2)
    research_json = json.dumps(state.get("research_context", {}), indent=2)

    llm = config.get_llm()
    structured_llm = llm.with_structured_output(EmphasisPlan)

    prompt_text = SELECT_CONTENT_USER.format(
        evidence_map_json=evidence_map_json,
        baseline_resume=state["baseline_resume"],
        constraints_json=constraints_json,
        research_context_json=research_json,
    )

    emphasis_plan: EmphasisPlan = structured_llm.invoke([
        SystemMessage(content=SELECT_CONTENT_SYSTEM),
        HumanMessage(content=prompt_text),
    ])

    return {
        "suppressions": suppressions,
        "emphasis_plan": emphasis_plan,
    }
