"""
Metrics and instrumentation for LLM-TeamGym experiments.

Tracks LLM API calls (tokens, cost, latency), match outcomes, and
per-topology statistics.  All data stays in-process until explicitly
exported to CSV / JSON — no external dependencies required.
"""

from __future__ import annotations

import csv
import json
import os
import time
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

from llm_team_gym.envs.logger import MatchRecord


# ---------------------------------------------------------------------------
# Cost estimation constants
# ---------------------------------------------------------------------------

# USD per 1 000 tokens  (prompt, completion)
COST_PER_1K_TOKENS: Dict[str, Dict[str, float]] = {
    # OpenAI
    "gpt-4o":              {"prompt": 0.0025, "completion": 0.0100},
    "gpt-4o-mini":         {"prompt": 0.00015, "completion": 0.0006},
    "gpt-4-turbo":         {"prompt": 0.0100, "completion": 0.0300},
    "gpt-4":               {"prompt": 0.0300, "completion": 0.0600},
    "gpt-3.5-turbo":       {"prompt": 0.0005, "completion": 0.0015},
    "o1":                  {"prompt": 0.0150, "completion": 0.0600},
    "o1-mini":             {"prompt": 0.0030, "completion": 0.0120},
    "o3-mini":             {"prompt": 0.0011, "completion": 0.0044},
    # Anthropic
    "claude-sonnet":       {"prompt": 0.0030, "completion": 0.0150},
    "claude-3-5-sonnet":   {"prompt": 0.0030, "completion": 0.0150},
    "claude-3-5-haiku":    {"prompt": 0.0008, "completion": 0.0040},
    "claude-3-opus":       {"prompt": 0.0150, "completion": 0.0750},
    "claude-opus-4":       {"prompt": 0.0150, "completion": 0.0750},
    "claude-sonnet-4":     {"prompt": 0.0030, "completion": 0.0150},
    # Google
    "gemini-2.0-flash":    {"prompt": 0.0001, "completion": 0.0004},
    "gemini-2.5-pro":      {"prompt": 0.00125, "completion": 0.0100},
    # Meta (hosted pricing varies; representative values)
    "llama-3.1-70b":       {"prompt": 0.00088, "completion": 0.00088},
    "llama-3.1-8b":        {"prompt": 0.00018, "completion": 0.00018},
}


def estimate_cost(
    model: str,
    prompt_tokens: int,
    completion_tokens: int,
) -> float:
    """Return estimated USD cost for an LLM call.

    Falls back to zero if the model is not in the lookup table.
    """
    pricing = COST_PER_1K_TOKENS.get(model)
    if pricing is None:
        return 0.0
    return (
        prompt_tokens * pricing["prompt"] / 1000.0
        + completion_tokens * pricing["completion"] / 1000.0
    )


# ---------------------------------------------------------------------------
# LLM call record
# ---------------------------------------------------------------------------

@dataclass
class LLMCallRecord:
    """Single LLM API invocation record."""

    agent_id: str
    model: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    cost_usd: float
    latency_s: float
    timestamp: float = field(default_factory=time.time)
    success: bool = True
    retry_count: int = 0


# ---------------------------------------------------------------------------
# Match-level metrics (kept alongside the MatchRecord)
# ---------------------------------------------------------------------------

@dataclass
class _MatchMetrics:
    """Internal bookkeeping for one match inside the collector."""

    match_record: MatchRecord
    topology_id: Optional[str] = None
    llm_calls: List[LLMCallRecord] = field(default_factory=list)


# ---------------------------------------------------------------------------
# MetricsCollector
# ---------------------------------------------------------------------------

class MetricsCollector:
    """Aggregates LLM-call and match-level metrics for an experiment.

    Can be shared across threads / coroutines (no locking — assume
    single-writer for now; add a ``threading.Lock`` wrapper if needed).
    """

    def __init__(self) -> None:
        self._calls: List[LLMCallRecord] = []
        self._matches: List[_MatchMetrics] = []

    # -- recording ----------------------------------------------------------

    def record_llm_call(self, record: LLMCallRecord) -> None:
        """Append an LLM call record."""
        self._calls.append(record)

    def record_match(
        self,
        match_record: MatchRecord,
        topology_id: Optional[str] = None,
    ) -> None:
        """Register a completed match."""
        self._matches.append(
            _MatchMetrics(match_record=match_record, topology_id=topology_id)
        )

    # -- scalar properties ---------------------------------------------------

    @property
    def total_llm_calls(self) -> int:
        return len(self._calls)

    @property
    def total_tokens(self) -> int:
        return sum(c.total_tokens for c in self._calls)

    @property
    def total_cost(self) -> float:
        return sum(c.cost_usd for c in self._calls)

    # -- per-agent helpers ---------------------------------------------------

    def calls_per_agent(self, agent_id: str) -> int:
        """Number of LLM calls made by *agent_id*."""
        return sum(1 for c in self._calls if c.agent_id == agent_id)

    def tokens_per_agent(self, agent_id: str) -> int:
        """Total tokens consumed by *agent_id*."""
        return sum(c.total_tokens for c in self._calls if c.agent_id == agent_id)

    # -- aggregate helpers ---------------------------------------------------

    def avg_latency(self) -> float:
        """Mean latency across all recorded LLM calls (seconds)."""
        if not self._calls:
            return 0.0
        return sum(c.latency_s for c in self._calls) / len(self._calls)

    # -- summary -------------------------------------------------------------

    def summary(self) -> Dict[str, Any]:
        """Return a dict with all aggregate statistics."""
        agent_ids = sorted({c.agent_id for c in self._calls})
        per_agent = {
            aid: {
                "calls": self.calls_per_agent(aid),
                "tokens": self.tokens_per_agent(aid),
            }
            for aid in agent_ids
        }

        successful = sum(1 for c in self._calls if c.success)
        failed = sum(1 for c in self._calls if not c.success)
        total_retries = sum(c.retry_count for c in self._calls)

        return {
            "total_llm_calls": self.total_llm_calls,
            "successful_calls": successful,
            "failed_calls": failed,
            "total_retries": total_retries,
            "total_tokens": self.total_tokens,
            "total_prompt_tokens": sum(c.prompt_tokens for c in self._calls),
            "total_completion_tokens": sum(c.completion_tokens for c in self._calls),
            "total_cost_usd": self.total_cost,
            "avg_latency_s": self.avg_latency(),
            "per_agent": per_agent,
            "total_matches": len(self._matches),
        }

    # -- export --------------------------------------------------------------

    def export_csv(self, path: str) -> None:
        """Write every LLM call record as one CSV row."""
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        fieldnames = [
            "agent_id", "model", "prompt_tokens", "completion_tokens",
            "total_tokens", "cost_usd", "latency_s", "timestamp",
            "success", "retry_count",
        ]
        with open(path, "w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=fieldnames)
            writer.writeheader()
            for c in self._calls:
                writer.writerow(asdict(c))

    def export_summary_csv(self, path: str) -> None:
        """Write one row per match with aggregated LLM metrics."""
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        fieldnames = [
            "match_id", "game_name", "topology_id", "winner",
            "total_steps", "duration_s",
            "llm_calls", "total_tokens", "total_cost_usd", "avg_latency_s",
        ]
        with open(path, "w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=fieldnames)
            writer.writeheader()
            for mm in self._matches:
                mr = mm.match_record
                # Find calls that fall within this match's time window.
                match_calls = [
                    c for c in self._calls
                    if mr.start_time <= c.timestamp <= (mr.end_time or float("inf"))
                ]
                avg_lat = (
                    sum(c.latency_s for c in match_calls) / len(match_calls)
                    if match_calls else 0.0
                )
                writer.writerow({
                    "match_id": mr.match_id,
                    "game_name": mr.game_name,
                    "topology_id": mm.topology_id or "",
                    "winner": mr.winner or "",
                    "total_steps": mr.total_steps,
                    "duration_s": f"{mr.duration():.4f}",
                    "llm_calls": len(match_calls),
                    "total_tokens": sum(c.total_tokens for c in match_calls),
                    "total_cost_usd": f"{sum(c.cost_usd for c in match_calls):.6f}",
                    "avg_latency_s": f"{avg_lat:.4f}",
                })


# ---------------------------------------------------------------------------
# ExperimentTracker
# ---------------------------------------------------------------------------

class ExperimentTracker:
    """High-level wrapper that drives a full experiment run.

    Manages a :class:`MetricsCollector`, accumulates per-topology stats,
    and persists results as JSON + CSV.

    Usage
    -----
    >>> tracker = ExperimentTracker("ablation_v1", output_dir="results/")
    >>> with tracker.start_match("flat", "TeamFish") as ctx:
    ...     # run the match, record LLM calls via tracker.collector …
    ...     ctx["match_record"] = record
    >>> tracker.save_results()
    """

    def __init__(
        self,
        experiment_name: str,
        output_dir: str = "results",
    ) -> None:
        self.experiment_name = experiment_name
        self.output_dir = output_dir
        self.collector = MetricsCollector()
        self._start_time: float = time.time()

        # Per-topology accumulators
        self._topology_matches: Dict[str, List[MatchRecord]] = {}
        self._topology_calls: Dict[str, List[LLMCallRecord]] = {}

        # Track the currently running match for the context manager
        self._active_topology: Optional[str] = None
        self._active_game_name: Optional[str] = None
        self._active_match_start: Optional[float] = None

    # -- context-manager for a single match ---------------------------------

    @contextmanager
    def start_match(self, topology_id: str, game_name: str):
        """Context manager that timestamps the match window.

        Yields a mutable dict; callers **must** set ``ctx["match_record"]``
        to the :class:`MatchRecord` before the block exits.

        Example::

            with tracker.start_match("hierarchy", "TeamFish") as ctx:
                record = runner.run()
                ctx["match_record"] = record
        """
        ctx: Dict[str, Any] = {}
        self._active_topology = topology_id
        self._active_game_name = game_name
        self._active_match_start = time.time()
        try:
            yield ctx
        finally:
            match_record: Optional[MatchRecord] = ctx.get("match_record")
            if match_record is not None:
                self.end_match(match_record, topology_id)
            self._active_topology = None
            self._active_game_name = None
            self._active_match_start = None

    def end_match(
        self,
        match_record: MatchRecord,
        topology_id: Optional[str] = None,
    ) -> None:
        """Register a completed match and bucket it by topology."""
        topo = topology_id or self._active_topology or "__default__"
        self.collector.record_match(match_record, topology_id=topo)

        self._topology_matches.setdefault(topo, []).append(match_record)

        # Snapshot calls that occurred during this match window.
        match_calls = [
            c for c in self.collector._calls
            if match_record.start_time <= c.timestamp <= (match_record.end_time or float("inf"))
        ]
        self._topology_calls.setdefault(topo, []).extend(match_calls)

    # -- per-topology statistics --------------------------------------------

    def topology_stats(self) -> Dict[str, Dict[str, Any]]:
        """Compute per-topology aggregate statistics.

        Returns a dict keyed by topology_id, each value containing:
        win_rate, avg_score, avg_cost, avg_latency, avg_llm_calls.
        """
        stats: Dict[str, Dict[str, Any]] = {}
        for topo, matches in self._topology_matches.items():
            n = len(matches)
            if n == 0:
                continue

            wins = sum(1 for m in matches if m.winner is not None)
            total_score = sum(
                sum(m.final_team_scores.values()) for m in matches
            )

            calls = self._topology_calls.get(topo, [])
            total_cost = sum(c.cost_usd for c in calls)
            total_lat = sum(c.latency_s for c in calls)
            n_calls = len(calls)

            stats[topo] = {
                "matches": n,
                "win_rate": wins / n,
                "avg_score": total_score / n,
                "total_tokens": sum(c.total_tokens for c in calls),
                "total_cost": total_cost,
                "avg_cost": total_cost / n,
                "avg_latency": total_lat / n_calls if n_calls else 0.0,
                "avg_llm_calls": n_calls / n,
            }
        return stats

    def comparative_table(self) -> List[Dict[str, Any]]:
        """Return a list-of-dicts table suitable for printing or CSV export.

        Columns: topology_id | win_rate | avg_score | total_tokens |
                 total_cost | avg_latency
        """
        rows: List[Dict[str, Any]] = []
        for topo, s in self.topology_stats().items():
            rows.append({
                "topology_id": topo,
                "win_rate": round(s["win_rate"], 4),
                "avg_score": round(s["avg_score"], 4),
                "total_tokens": s["total_tokens"],
                "total_cost": round(s["total_cost"], 6),
                "avg_latency": round(s["avg_latency"], 4),
            })
        return rows

    # -- persistence ---------------------------------------------------------

    def save_results(self) -> Dict[str, str]:
        """Write JSON summary + CSV files to *output_dir*.

        Returns a dict mapping output type to file path.
        """
        base = os.path.join(self.output_dir, self.experiment_name)
        os.makedirs(base, exist_ok=True)

        paths: Dict[str, str] = {}

        # 1. Full summary JSON
        json_path = os.path.join(base, "summary.json")
        payload = {
            "experiment_name": self.experiment_name,
            "start_time": self._start_time,
            "end_time": time.time(),
            "collector_summary": self.collector.summary(),
            "topology_stats": self.topology_stats(),
            "comparative_table": self.comparative_table(),
        }
        with open(json_path, "w") as fh:
            json.dump(payload, fh, indent=2, default=str)
        paths["summary_json"] = json_path

        # 2. LLM calls CSV
        calls_csv = os.path.join(base, "llm_calls.csv")
        self.collector.export_csv(calls_csv)
        paths["llm_calls_csv"] = calls_csv

        # 3. Per-match summary CSV
        match_csv = os.path.join(base, "match_summary.csv")
        self.collector.export_summary_csv(match_csv)
        paths["match_summary_csv"] = match_csv

        # 4. Comparative topology CSV
        comp_csv = os.path.join(base, "topology_comparison.csv")
        table = self.comparative_table()
        if table:
            fieldnames = list(table[0].keys())
            with open(comp_csv, "w", newline="") as fh:
                writer = csv.DictWriter(fh, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(table)
        paths["topology_comparison_csv"] = comp_csv

        return paths
