"""Main evaluation loop: orchestrates attackers, defender, and judging."""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any

from redteam_sft.prompt_injection.attackers import instantiate_attacker
from redteam_sft.prompt_injection.client import OpenAICompatibleChatClient
from redteam_sft.prompt_injection.config import load_config, parse_args
from redteam_sft.prompt_injection.defender import build_defender_messages
from redteam_sft.prompt_injection.io_utils import (
    MetricTokenizer,
    dump_jsonl,
    load_jsonl,
    write_summary_csv,
)
from redteam_sft.prompt_injection.judging import (
    evaluate_hard_oracle,
    evaluate_refusal,
    evaluate_with_judge,
)
from redteam_sft.prompt_injection.reporting import aggregate_results

logger = logging.getLogger(__name__)


def _determine_final_label(
    scenario: str,
    hard_oracle_label: bool,
    judge_label: str,
) -> str:
    if hard_oracle_label:
        return "success"
    if scenario == "goal_hijacking" and judge_label == "SUCCESS":
        return "success"
    return "failure"


def _evaluate_single_case(
    attacker,
    case: dict[str, Any],
    run_seed: int,
    system_prompt_template: str,
    defender_client: OpenAICompatibleChatClient,
    judge_client: OpenAICompatibleChatClient | None,
    hard_oracle_cfg: dict[str, Any],
    refusal_cfg: dict[str, Any],
    judge_cfg: dict[str, Any],
    metric_tokenizer: MetricTokenizer,
) -> dict[str, Any]:
    started = time.time()

    attack = attacker.generate(case, run_seed)
    defender_messages = build_defender_messages(
        system_prompt_template, case, attack.payload,
    )
    defender_response = defender_client.create_chat_completion(defender_messages)

    hard_oracle_label = evaluate_hard_oracle(case, defender_response, hard_oracle_cfg)
    refusal_label = evaluate_refusal(defender_response, refusal_cfg)

    scenario = case.get("scenario", "")
    judge_label = "NOT_USED"
    if scenario == "goal_hijacking" and not hard_oracle_label:
        judge_label = evaluate_with_judge(
            judge_client, judge_cfg, case, defender_response,
        )

    final_label = _determine_final_label(scenario, hard_oracle_label, judge_label)

    return {
        "case_id": case.get("case_id"),
        "scenario": scenario,
        "attacker_id": attacker.id,
        "attack_seed": run_seed,
        "attacker_payload": attack.payload,
        "defender_request": defender_messages[1]["content"],
        "defender_response": defender_response,
        "hard_oracle_label": hard_oracle_label,
        "refusal_label": refusal_label,
        "judge_label": judge_label,
        "final_label": final_label,
        "attacker_prompt_tokens": attack.prompt_tokens,
        "defender_response_tokens": metric_tokenizer.count(defender_response),
        "latency_seconds": round(time.time() - started, 3),
    }


def run() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    args = parse_args()
    cfg = load_config(args.config)

    experiment_cfg = cfg.get("experiment", {})
    metric_tokenizer = MetricTokenizer(cfg.get("metrics", {}).get("tokenizer_id"))

    # ---- Load benchmark cases ----
    cases_path = args.cases or cfg.get("datasets", {}).get("test_path")
    if not cases_path:
        raise ValueError("No benchmark cases path provided")
    cases = load_jsonl(cases_path)
    if args.limit is not None:
        cases = cases[: args.limit]

    allowed_scenarios = {
        sc["id"] for sc in cfg.get("scenarios", []) if sc.get("enabled", True)
    }
    if allowed_scenarios:
        cases = [c for c in cases if c.get("scenario") in allowed_scenarios]

    # ---- Select attackers ----
    attacker_cfgs = cfg.get("attackers", [])
    if args.attacker:
        wanted = set(args.attacker)
        attacker_cfgs = [a for a in attacker_cfgs if a["id"] in wanted]

    if not attacker_cfgs:
        raise ValueError("No attackers selected")
    if not cases:
        raise ValueError("No benchmark cases selected")

    # ---- Paths ----
    output_root = Path(cfg["artifacts"]["output_root"])
    output_root.mkdir(parents=True, exist_ok=True)
    results_path = Path(cfg["reporting"]["detailed_results_path"])
    summary_path = Path(cfg["reporting"]["summary_table_path"])

    # ---- System prompt ----
    system_prompt_path = cfg["target"]["system_prompt_path"]
    system_prompt_template = Path(system_prompt_path).read_text(encoding="utf-8")

    # ---- Clients ----
    defender_client = OpenAICompatibleChatClient(cfg["target"])

    judge_cfg = cfg.get("judging", {}).get("llm_judge", {})
    judge_client = None
    if judge_cfg.get("enabled"):
        judge_client = OpenAICompatibleChatClient(judge_cfg)

    # ---- Instantiate attackers ----
    attackers = [
        instantiate_attacker(acfg, metric_tokenizer) for acfg in attacker_cfgs
    ]

    random_seed = experiment_cfg.get("random_seed", 42)
    attack_seeds = experiment_cfg.get("attack_seeds", [random_seed])
    if not attack_seeds:
        attack_seeds = [random_seed]

    hard_oracle_cfg = cfg.get("judging", {}).get("hard_oracle", {})
    refusal_cfg = cfg.get("judging", {}).get("refusal_tracker", {})

    # ---- Main evaluation loop ----
    try:
        from tqdm import tqdm
        has_tqdm = True
    except ImportError:
        has_tqdm = False

    results: list[dict[str, Any]] = []
    error_count = 0
    success_count = 0
    total_steps = len(attackers) * len(cases)

    pbar = tqdm(total=total_steps, desc="Evaluating", unit="case") if has_tqdm else None

    for attacker in attackers:
        for case_index, case in enumerate(cases):
            run_seed = attack_seeds[case_index % len(attack_seeds)]
            case_id = case.get("case_id", f"index-{case_index}")
            if pbar is not None:
                pbar.set_description(
                    f"[{attacker.id}] {case_id} | ok={success_count} err={error_count}"
                )
            try:
                result = _evaluate_single_case(
                    attacker=attacker,
                    case=case,
                    run_seed=run_seed,
                    system_prompt_template=system_prompt_template,
                    defender_client=defender_client,
                    judge_client=judge_client,
                    hard_oracle_cfg=hard_oracle_cfg,
                    refusal_cfg=refusal_cfg,
                    judge_cfg=judge_cfg,
                    metric_tokenizer=metric_tokenizer,
                )
                results.append(result)
                if result.get("final_label") == "success":
                    success_count += 1
            except Exception:
                error_count += 1
                logger.exception(
                    "Error evaluating case %s with attacker %s",
                    case_id,
                    attacker.id,
                )
            if pbar is not None:
                pbar.update(1)

    if pbar is not None:
        pbar.close()

    # ---- Write outputs ----
    dump_jsonl(results_path, results)
    write_summary_csv(summary_path, aggregate_results(results))

    logger.info("Saved %d results to %s", len(results), results_path)
    logger.info("Saved summary to %s", summary_path)
    if error_count:
        logger.warning("%d case(s) failed with errors", error_count)
