import numpy as np
from scipy.optimize import linear_sum_assignment
from typing import Dict, List, Optional, Tuple, Set
from logic.assets.data_structures import AgentState

class PursuerAssignmentManager:
    """
    Assigns each pursuer to exactly one evader using the Hungarian Algorithm.
    """

    def __init__(self):
        self.assignments: Dict[int, int]       = {}   # pursuer_id -> evader_id
        self._teams:      Dict[int, List[int]] = {}   # evader_id  -> [pursuer_ids]

    def update(
        self,
        pursuer_states: Dict[int, AgentState],
        evader_states:  Dict[int, AgentState],
        hysteresis:     float = 1.0,
    ) -> None:
        if not evader_states or not pursuer_states:
            return

        pursuer_ids = sorted(pursuer_states.keys())
        evader_ids  = sorted(evader_states.keys())
        n_p = len(pursuer_ids)
        n_e = len(evader_ids)

        # Distance matrix D[i, j] = ||p_pursuer_i - p_evader_j||
        P = np.stack([pursuer_states[pid].p for pid in pursuer_ids])  # (n_p, 3)
        E = np.stack([evader_states[eid].p  for eid in evader_ids])   # (n_e, 3)
        D = np.linalg.norm(P[:, None, :] - E[None, :, :], axis=2)     # (n_p, n_e)

        # assignment via Hungarian algorithm 
        cap = max(1, int(np.ceil(n_p / n_e)))
        D_tiled = np.tile(D, cap)[:, :n_p] 
        row_ind, col_ind = linear_sum_assignment(D_tiled)

        optimal_assign: Dict[int, int] = {}
        for r, c in zip(row_ind, col_ind):
            pid = pursuer_ids[r]
            eid = evader_ids[c % n_e]   
            optimal_assign[pid] = eid

        new_assign: Dict[int, int] = {}
        evader_col = {eid: j for j, eid in enumerate(evader_ids)}
        for r, pid in enumerate(pursuer_ids):
            proposed_eid = optimal_assign[pid]
            current_eid  = self.assignments.get(pid)
            if (
                current_eid is not None
                and current_eid in evader_col        
                and proposed_eid != current_eid
                and D[r, evader_col[current_eid]] - D[r, evader_col[proposed_eid]] < hysteresis
            ):
                new_assign[pid] = current_eid         
            else:
                new_assign[pid] = proposed_eid

        new_teams: Dict[int, List[int]] = {eid: [] for eid in evader_ids}
        for pid in pursuer_ids:
            new_teams[new_assign[pid]].append(pid)

        self.assignments = new_assign
        self._teams      = new_teams

    def partial_update(
        self,
        pursuer_states: Dict[int, AgentState],
        evader_states:  Dict[int, AgentState],
        newly_captured: Set[int],
    ) -> None:
        """
        Reassign pursuers whose evaders just got captured
        """
        if not newly_captured:
            return  

        pursuer_ids = sorted(pursuer_states.keys())
        evader_ids  = sorted(evader_states.keys()) 

        if not evader_ids:
            return

        freed_pursuers = [
            pid for pid in pursuer_ids
            if self.assignments.get(pid) in newly_captured
        ]

        if not freed_pursuers:
            return

        slot_count = {eid: 0 for eid in evader_ids}
        for pid in pursuer_ids:
            if pid not in freed_pursuers:
                eid = self.assignments.get(pid)
                if eid in slot_count:
                    slot_count[eid] += 1

        cap = max(1, int(np.ceil(len(pursuer_ids) / len(evader_ids))))
        evader_col = {eid: j for j, eid in enumerate(evader_ids)}

        P = np.stack([pursuer_states[pid].p for pid in freed_pursuers])
        E = np.stack([evader_states[eid].p  for eid in evader_ids])
        D = np.linalg.norm(P[:, None, :] - E[None, :, :], axis=2)

        order = np.argsort(D.min(axis=1))
        for row, pi in enumerate(order):
            pid = freed_pursuers[pi]
            chosen_eid = min(
                evader_ids,
                key=lambda eid, r=pi: ( 
                    slot_count[eid] >= cap,
                    D[r, evader_col[eid]],
                    eid,
                ),
            )
            self.assignments[pid] = chosen_eid
            slot_count[chosen_eid] += 1

        new_teams: Dict[int, List[int]] = {eid: [] for eid in evader_ids}
        for pid in pursuer_ids:
            eid = self.assignments.get(pid)
            if eid in new_teams:
                new_teams[eid].append(pid)
        self._teams = new_teams

    def get(self, pursuer_id: int) -> Tuple[Optional[int], int, int]:

        evader_id = self.assignments.get(pursuer_id)
        if evader_id is None:
            return None, 0, 1

        team        = self._teams.get(evader_id, [pursuer_id])
        local_idx   = team.index(pursuer_id) if pursuer_id in team else 0
        local_count = len(team)
        return evader_id, local_idx, local_count

    def get_team(self, evader_id: int) -> List[int]:
        """Return sorted list of pursuer IDs assigned to evader_id."""
        return list(self._teams.get(evader_id, []))

    def summary(self) -> str:
        lines = [f"  evader {eid}: pursuers {team}"
                 for eid, team in sorted(self._teams.items())]
        return "\n".join(lines) if lines else "  (no assignments)"
