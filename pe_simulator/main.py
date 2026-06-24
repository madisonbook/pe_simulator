"""
main.py — Pursuit-Evasion Simulation (Parallel Edition)
Implements the multi-agent pursuit scenario from:
  "Robust Multi-Agent Pursuit with Preemptive Communication via
   Reachability-Based Control Barrier Functions"

Parallelism model
-----------------
Three nested levels of work are dispatched to a shared ProcessPoolExecutor:

  Level 1  — Strategies  (outermost loop, e.g. none/full/preemptive)
  Level 2  — Scenarios   (e.g. 8p8e / 12p6e / 16p4e)
  Level 3  — Trials      (innermost, embarrassingly parallel per seed)

All futures are submitted up-front and collected with as_completed so the
progress bar reflects wall-clock throughput rather than sequential batches.
"""

import sys
import time
import argparse
import itertools
import numpy as np
from typing import Dict, List, Set, Tuple
from dataclasses import dataclass, field
from concurrent.futures import ProcessPoolExecutor, as_completed

from logic.assets.config import SimConfig
from logic.assets.data_structures import AgentState, ReachableSet
from logic.assets.reachability import Reachability
from logic.communication.comm_manager import CommunicationManager
from logic.controllers.pursuer_controller import CBFController
from logic.controllers.evader_controller import EvaderController
from logic.assets.logger import setup_worker_logging, setup_master_logging


def make_state(p: np.ndarray, v: np.ndarray = None, u: np.ndarray = None,
               t: float = 0.0) -> AgentState:
    if v is None:
        v = np.zeros(3)
    if u is None:
        u = np.zeros(3)
    return AgentState(p=np.array(p, dtype=float),
                      v=np.array(v, dtype=float),
                      u=np.array(u, dtype=float),
                      t=t)


def integrate(state: AgentState, u: np.ndarray, dt: float,
              v_max: float, t_new: float) -> AgentState:
    """Euler integration with velocity clipping."""
    v_new = state.v + u * dt
    v_mag = np.linalg.norm(v_new)
    if v_mag > v_max:
        v_new = v_new / v_mag * v_max
    p_new = state.p + v_new * dt
    return AgentState(p=p_new, v=v_new, u=u.copy(), t=t_new)


def check_collisions(pursuer_states: Dict[int, AgentState], d_safe: float) -> int:
    """Count pursuer pairs currently violating the safety distance."""
    ids = sorted(pursuer_states.keys())
    count = 0
    for idx_a, i in enumerate(ids):
        for j in ids[idx_a + 1:]:
            dist = np.linalg.norm(pursuer_states[i].p - pursuer_states[j].p)
            if dist < (d_safe - 1e-3):
                count += 1
    return count


def _seed_comm_manager(comm_manager: CommunicationManager,
                        pursuer_states: Dict[int, AgentState],
                        pursuer_ids: List[int]) -> None:
    for sender_id in pursuer_ids:
        for receiver_id in pursuer_ids:
            if sender_id == receiver_id:
                continue
            comm_manager.deliver_message(receiver_id, sender_id,
                                         pursuer_states[sender_id])
            comm_manager.last_transmitted_state[(sender_id, receiver_id)] = \
                pursuer_states[sender_id].copy()
            
def _init_worker(q):
    setup_worker_logging(q)

@dataclass
class TrialResult:
    total_messages: int
    msgs_per_agent_per_sec: float
    collisions: int
    num_captured: int
    avg_capture_time: float
    sim_time: float

def run_trial(config: SimConfig, seed: int, verbose: bool = False) -> TrialResult:
    np.random.seed(seed)

    bounds = config.world_bounds * 0.5

    # --- Initialise pursuer states ---
    pursuer_states: Dict[int, AgentState] = {}
    for i in range(config.num_pursuers):
        p = np.random.uniform(-bounds, bounds, 3)
        pursuer_states[i] = make_state(p, t=0.0)

    # --- Initialise evader states ---
    evader_states: Dict[int, AgentState] = {}
    for e in range(config.num_evaders):
        p = np.random.uniform(-bounds, bounds, 3)
        evader_states[e] = make_state(p, t=0.0)

    # --- Build components ---
    reachability   = Reachability(config.pursuer_u_max, config.pursuer_v_max, config.d_safe, config.dt)
    comm_manager   = CommunicationManager(config, reachability)
    cbf_controller = CBFController(config)
    evader_ctrl    = EvaderController(config)

    pursuer_ids = list(range(config.num_pursuers))
    comm_manager.update_states([pursuer_states[i] for i in pursuer_ids])
    cbf_controller.update_assignments(pursuer_states, evader_states)
    _seed_comm_manager(comm_manager, pursuer_states, pursuer_ids)

    captured:      Set[int]         = set()
    capture_times: Dict[int, float] = {}
    total_collisions                 = 0
    sim_time                         = 0.0
    active_collisions = set()
    collision_events = []
    qp_infeasible_count = 0
    qp_failure_log = []

    n_steps = int(config.sim_duration / config.dt)

    for step in range(n_steps):
        t = step * config.dt

        active_evaders = {e: s for e, s in evader_states.items() if e not in captured}
        if not active_evaders:
            sim_time = t
            break

        cbf_controller.update_assignments(pursuer_states, active_evaders)
        comm_manager.update_states([pursuer_states[i] for i in pursuer_ids])
        for i in pursuer_ids:
            neighbours = [j for j in pursuer_ids if j != i]
            comm_manager.process_outgoing_messages(
                i, pursuer_states[i], neighbours, t
            )

        ids = sorted(pursuer_states.keys())

        for idx_a, i in enumerate(ids):
            for j in ids[idx_a + 1:]:

                dist = np.linalg.norm(
                    pursuer_states[i].p -
                    pursuer_states[j].p
                )

                pair = (i, j)

                if dist < config.d_safe:

                    if pair not in active_collisions:

                        active_collisions.add(pair)
                        total_collisions += 1

                        collision_events.append({
                            "time": t_new,
                            "pair": pair,
                            "distance": dist,
                        })

                        
                        '''print(
                            f"\nCOLLISION EVENT "
                            f"t={t_new:.2f}s "
                            f"pair={pair} "
                            f"d={dist:.3f}"
                        )'''

                else:
                    active_collisions.discard(pair)


        # --- Pursuer control phase ---
        pursuer_controls: Dict[int, np.ndarray] = {}
        for i in pursuer_ids:
            evader_id, local_idx, local_count = cbf_controller.assignment_manager.get(i)

            if evader_id is None or evader_id not in active_evaders:
                # No assigned evader — damp velocity to hover in place.
                pursuer_controls[i] = np.clip(-2.0 * pursuer_states[i].v,
                                              -config.pursuer_u_max,
                                               config.pursuer_u_max)
                continue

            evader_state = active_evaders[evader_id]

            neighbour_reach_sets: Dict[int, "ReachableSet"] = {}
            for j in pursuer_ids:
                if j == i:
                    continue
                last_known_j = comm_manager.get_inbox(i).get(j)
                if last_known_j is not None:
                    elapsed = t - last_known_j.t
                    reach_set = reachability.compute_sphere_set(last_known_j, elapsed)
                    neighbour_reach_sets[j] = reach_set

            u_i, feasible, metrics = cbf_controller.solve_cbf_qp(
                pursuer_id          = i,
                pursuer_state       = pursuer_states[i],
                evader_state        = evader_state,
                neighbor_reach_sets = neighbour_reach_sets,
                curr_time           = t,
                local_idx           = local_idx,
                local_count         = local_count,
            )
            pursuer_controls[i] = u_i

            if not feasible:
                qp_infeasible_count += 1

                qp_failure_log.append({
                    "time": t,
                    "agent": i
                })

        # --- Evader control phase ---
        evader_controls: Dict[int, np.ndarray] = {}
        pursuer_positions = [pursuer_states[i].p for i in pursuer_ids]
        for e, ev_state in active_evaders.items():
            evader_controls[e] = evader_ctrl.compute_control(ev_state, pursuer_positions)

        # --- Integrate dynamics ---
        t_new = t + config.dt
        for i in pursuer_ids:
            pursuer_states[i] = integrate(
                pursuer_states[i], pursuer_controls[i],
                config.dt, config.pursuer_v_max, t_new
            )
            pursuer_states[i].u = pursuer_controls[i].copy()

        for e in list(active_evaders.keys()):
            evader_states[e] = integrate(
                evader_states[e], evader_controls[e],
                config.dt, config.evader_v_max, t_new
            )

        # --- Capture check ---
        newly_captured: Set[int] = set()
        for e, ev_state in active_evaders.items():
            for i in pursuer_ids:
                dist = np.linalg.norm(pursuer_states[i].p - ev_state.p)
                if dist <= config.d_capture:
                    if e not in captured:
                        captured.add(e)
                        newly_captured.add(e)
                        capture_times[e] = t_new
                        if verbose:
                            print(f"    t={t_new:.2f}s  Evader {e} captured by pursuer {i}")
                    break

        if newly_captured:
            remaining = {e: s for e, s in evader_states.items() if e not in captured}
            if remaining:
                cbf_controller.assignment_manager.partial_update(
                    pursuer_states, remaining, newly_captured
                )

        sim_time = t_new

    total_msgs = comm_manager.get_total_messages()
    msgs_rate  = (total_msgs / (config.num_pursuers * sim_time)
                  if sim_time > 0 else 0.0)
    avg_cap    = (float(np.mean(list(capture_times.values())))
                  if capture_times else float('nan'))

    return TrialResult(
        total_messages         = total_msgs,
        msgs_per_agent_per_sec = msgs_rate,
        collisions             = total_collisions,
        num_captured           = len(captured),
        avg_capture_time       = avg_cap,
        sim_time               = sim_time,
    )

# Top-level wrapper so the tuple argument survives pickling across processes.
def _trial_worker(args: Tuple) -> Tuple:
    """
    Unpack (strategy, scenario_key, seed, config) → run trial → return tagged result.

    Returning the tags alongside the result lets the collector sort futures
    without needing a shared dict or manager.
    """
    strategy, scenario_key, seed, config = args
    result = run_trial(config, seed=seed, verbose=False)
    return strategy, scenario_key, seed, result


# ---------------------------------------------------------------------------
# Aggregation helpers
# ---------------------------------------------------------------------------

@dataclass
class ScenarioSummary:
    strategy:       str
    scenario_key:   str
    scenario_name:  str
    n_trials:       int
    avg_messages:   float
    avg_rate:       float
    total_collisions: int
    avg_sim_time:   float
    avg_cap_time:   float
    results:        List[TrialResult] = field(default_factory=list)


def aggregate(results: List[TrialResult], strategy: str,
              scenario_key: str, scenario_name: str) -> ScenarioSummary:
    cap_times = [r.avg_capture_time for r in results if not np.isnan(r.avg_capture_time)]
    return ScenarioSummary(
        strategy        = strategy,
        scenario_key    = scenario_key,
        scenario_name   = scenario_name,
        n_trials        = len(results),
        avg_messages    = float(np.mean([r.total_messages         for r in results])),
        avg_rate        = float(np.mean([r.msgs_per_agent_per_sec for r in results])),
        total_collisions= sum  (r.collisions                      for r in results),
        avg_sim_time    = float(np.mean([r.sim_time               for r in results])),
        avg_cap_time    = float(np.mean(cap_times)) if cap_times else float('nan'),
        results         = results,
    )


# ---------------------------------------------------------------------------
# Parallel runner
# ---------------------------------------------------------------------------

def run_comparison(
    strategies:    List[str],
    scenario_defs: Dict[str, Tuple[str, SimConfig]],
    n_trials:      int,
    max_workers:   int,
) -> Dict[Tuple[str, str], ScenarioSummary]:
    """
    Submit every (strategy × scenario × trial) combination to the pool
    and collect results as they complete.

    Returns a dict keyed by (strategy, scenario_key).
    """
    listener, q = setup_master_logging()

    # Build the flat work list up-front so all futures are in-flight together.
    work_items: List[Tuple] = []
    for strategy, (scenario_key, (scenario_name, base_config)) in itertools.product(
            strategies, scenario_defs.items()):
        config = SimConfig(
            num_pursuers  = base_config.num_pursuers,
            num_evaders   = base_config.num_evaders,
            comm_strategy = strategy,
        )
        for seed in range(n_trials):
            work_items.append((strategy, scenario_key, seed, config))

    total = len(work_items)
    # Raw results bucket: (strategy, scenario_key) → [TrialResult, ...]
    buckets: Dict[Tuple[str, str], List[TrialResult]] = {
        (s, k): [] for s in strategies for k in scenario_defs
    }

    print(f"\nDispatching {total} trials "
          f"({len(strategies)} strategies × {len(scenario_defs)} scenarios "
          f"× {n_trials} trials) across {max_workers} worker processes …\n")

    wall_start = time.time()
    completed  = 0

    with ProcessPoolExecutor(max_workers=max_workers, initializer=_init_worker, initargs=(q,)) as pool:
        futures = {pool.submit(_trial_worker, item): item for item in work_items}

        for future in as_completed(futures):
            strategy, scenario_key, seed, result = future.result()
            buckets[(strategy, scenario_key)].append(result)
            completed += 1

            # Lightweight progress ticker 
            '''
            if completed % max(1, total // 40) == 0 or completed == total:
                elapsed  = time.time() - wall_start
                rate     = completed / elapsed
                eta      = (total - completed) / rate if rate > 0 else float('inf')
                pct      = 100 * completed / total
                bar_len  = 30
                filled   = int(bar_len * completed / total)
                bar      = "█" * filled + "░" * (bar_len - filled)
                print(f"\r  [{bar}] {pct:5.1f}%  {completed}/{total}  "
                      f"{rate:.1f} trials/s  ETA {eta:5.1f}s",
                      end="", flush=True)


    print()  # newline after progress bar

    '''

    # Aggregate
    summaries: Dict[Tuple[str, str], ScenarioSummary] = {}
    for (strategy, scenario_key), results in buckets.items():
        scenario_name, _ = scenario_defs[scenario_key]
        summaries[(strategy, scenario_key)] = aggregate(
            results, strategy, scenario_key, scenario_name
        )
    
    listener.stop()

    return summaries


# ---------------------------------------------------------------------------
# Report printer
# ---------------------------------------------------------------------------

COLUMN_ORDER = ["none", "periodic", "event", "preemptive", "full"]

_SEP  = "─"
_DSEP = "═"


def _col(s: str, w: int, align: str = "<") -> str:
    return format(str(s), f"{align}{w}")


def print_report(
    summaries:     Dict[Tuple[str, str], ScenarioSummary],
    strategies:    List[str],
    scenario_defs: Dict[str, Tuple[str, SimConfig]],
    wall_elapsed:  float,
) -> None:
    """
    Print a compact comparison table for each metric, grouped by scenario.
    Strategy columns are sorted in COLUMN_ORDER where present.

    Layout example (msgs/agent/s):

      msgs/agent/s
      ┌──────────┬──────────┬──────────┬──────────┐
      │ Scenario │  none    │  event   │preemptive│
      ├──────────┼──────────┼──────────┼──────────┤
      │ 8P, 8E   │  0.123   │  0.087   │  0.061   │
      │ 12P, 6E  │  0.145   │  0.094   │  0.067   │
      └──────────┴──────────┴──────────┴──────────┘
    """

    # Respect any user-specified order; put unknown strategies at the end.
    strat_order = [s for s in COLUMN_ORDER if s in strategies] + \
                  [s for s in strategies   if s not in COLUMN_ORDER]

    scenario_keys  = list(scenario_defs.keys())
    scenario_names = {k: v[0] for k, v in scenario_defs.items()}

    # Column widths
    w_scenario = max(len("Scenario"), max(len(scenario_names[k]) for k in scenario_keys)) + 2
    w_strat    = max(12, max(len(s) for s in strat_order) + 2)

    def hline(left, mid, right, sep):
        row = left + (sep * w_scenario) + mid
        row += mid.join(sep * w_strat for _ in strat_order) + right
        return row

    def header_row():
        row = "│" + _col(" Scenario", w_scenario) + "│"
        row += "│".join(_col(f" {s}", w_strat) for s in strat_order) + "│"
        return row

    metrics = [
        ("avg_messages",     "Avg total messages",    lambda s: f"{s.avg_messages:>10.1f}"),
        ("avg_rate",         "Msgs / agent / s",      lambda s: f"{s.avg_rate:>10.3f}"),
        ("total_collisions", "Total collisions",      lambda s: f"{s.total_collisions:>10d}"),
        ("avg_cap_time",     "Avg capture time (s)",  lambda s: (f"{s.avg_cap_time:>10.2f}"
                                                                  if not np.isnan(s.avg_cap_time)
                                                                  else f"{'N/A':>10}")),
        ("avg_sim_time",     "Avg sim time (s)",      lambda s: f"{s.avg_sim_time:>10.2f}"),
    ]

    print(f"\n{'═'*60}")
    print("  COMPARISON REPORT")
    print(f"{'═'*60}")
    print(f"  Wall-clock time : {wall_elapsed:.1f}s")
    print(f"  Strategies      : {', '.join(strat_order)}")
    print(f"  Scenarios       : {', '.join(scenario_names[k] for k in scenario_keys)}")

    for metric_key, metric_label, fmt_fn in metrics:
        print(f"\n  ── {metric_label} ──")
        print("  " + hline("┌", "┬", "┐", _SEP))
        print("  " + header_row())
        print("  " + hline("├", "┼", "┤", _SEP))
        for sk in scenario_keys:
            row = "│" + _col(f" {scenario_names[sk]}", w_scenario) + "│"
            for strat in strat_order:
                summary = summaries.get((strat, sk))
                cell = fmt_fn(summary) if summary else f"{'—':>10}"
                row += _col(f" {cell.strip()}", w_strat) + "│"
            print("  " + row)
        print("  " + hline("└", "┴", "┘", _SEP))


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

ALL_STRATEGIES = ["none", "periodic", "event", "preemptive", "full"]
#ALL_STRATEGIES = ["periodic"]

SCENARIO_DEFS: Dict[str, Tuple[str, SimConfig]] = {
    "8p4e":  ("8P, 4E",  SimConfig(num_pursuers=8,  num_evaders=4)),
    "8p8e":  ("8P, 8E",  SimConfig(num_pursuers=8,  num_evaders=8)),   
    "12p4e":  ("12P, 4E",  SimConfig(num_pursuers=12,  num_evaders=4)),
    "12p6e":  ("12P, 6E",  SimConfig(num_pursuers=12,  num_evaders=6)),
    "12p12e": ("12P, 8E", SimConfig(num_pursuers=12, num_evaders=12)),
    "16p4e":  ("16P, 4E",  SimConfig(num_pursuers=16,  num_evaders=4)),
    "16p8e":  ("16P, 8E",  SimConfig(num_pursuers=16,  num_evaders=8)),
    "16p16e": ("16P, 16E", SimConfig(num_pursuers=16, num_evaders=16)),
}

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Pursuit-evasion simulation — parallel comparison study",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples
--------
  # Full comparison study, all strategies × all scenarios, 20 trials each
  python main.py --trials 20

  # Compare only two strategies across all scenarios
  python main.py --strategies preemptive none --trials 10

  # Single scenario, three strategies, 30 trials, 8 workers
  python main.py --scenario 8p8e --strategies none event preemptive --trials 30 --workers 8

  # Quick single-trial debug run (sequential, verbose)
  python main.py --single --strategy preemptive --scenario 8p8e
""",
    )

    p.add_argument("--strategies", nargs="+",
                   choices=ALL_STRATEGIES, default=ALL_STRATEGIES,
                   metavar="STRATEGY",
                   help=f"Strategies to compare. Choices: {ALL_STRATEGIES}. "
                        f"Default: all.")
    p.add_argument("--strategy",
                   choices=ALL_STRATEGIES,
                   help="Shortcut: single strategy (overrides --strategies).")
    p.add_argument("--scenario", default="all",
                   choices=list(SCENARIO_DEFS.keys()) + ["all"],
                   help="Scenario(s) to run. Default: all.")
    p.add_argument("--trials",  type=int, default=20,
                   help="Number of trials per (strategy, scenario) cell. Default: 50.")
    p.add_argument("--workers", type=int, default=None,
                   help="Worker processes. Default: os.cpu_count().")
    p.add_argument("--single",  action="store_true",
                   help="Run one verbose trial for debugging (sequential, no comparison).")
    p.add_argument("--verbose", action="store_true",
                   help="Print per-capture events (only effective with --single).")

    return p.parse_args()


def main() -> None:
    args = _parse_args()

    # ---- Resolve strategies ----
    if args.strategy:
        strategies = [args.strategy]
    else:
        strategies = args.strategies

    # ---- Resolve scenarios ----
    if args.scenario == "all":
        scenario_defs = SCENARIO_DEFS
    else:
        scenario_defs = {args.scenario: SCENARIO_DEFS[args.scenario]}

    # ---- Single debug run ----
    if args.single:
        if len(strategies) > 1:
            print("--single accepts one strategy at a time; using first:", strategies[0])
        strategy = strategies[0]
        sk       = next(iter(scenario_defs))
        name, base_cfg = scenario_defs[sk]
        config = SimConfig(
            num_pursuers  = base_cfg.num_pursuers,
            num_evaders   = base_cfg.num_evaders,
            comm_strategy = strategy,
        )
        print(f"\nSingle debug trial: {name}  strategy={strategy}")
        r = run_trial(config, seed=0, verbose=args.verbose or True)
        print(f"\n  Messages     : {r.total_messages}")
        print(f"  Msgs/agent/s : {r.msgs_per_agent_per_sec:.3f}")
        print(f"  Collisions   : {r.collisions}")
        print(f"  Captured     : {r.num_captured}/{config.num_evaders}")
        print(f"  Sim time     : {r.sim_time:.2f}s")
        sys.exit(0)

    # ---- Parallel comparison study ----
    import os
    max_workers = args.workers or os.cpu_count()

    wall_start = time.time()

    summaries = run_comparison(
        strategies    = strategies,
        scenario_defs = scenario_defs,
        n_trials      = args.trials,
        max_workers   = max_workers,
    )

    wall_elapsed = time.time() - wall_start

    print_report(summaries, strategies, scenario_defs, wall_elapsed)
    print("\nDone.")


if __name__ == "__main__":
    main()