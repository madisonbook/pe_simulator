import numpy as np
from typing import List, Dict, Optional, Tuple
from logic.assets.data_structures import AgentState, ReachableSet, CommStrategy
from logic.assets.config import SimConfig
from logic.assets.reachability import Reachability


def compute_kinematic(config: SimConfig, sender_state: AgentState, receiver_state: AgentState, curr_time: float) -> float:

    # agent i -> agent j
    i = sender_state
    j = receiver_state

    if i is None or j is None:
        return 1.0   # no info, communicate

    tau_ji = curr_time - j.t

    predicted_pj = (j.p + j.v * tau_ji)

    p_ij = i.p - predicted_pj
    v_ij = i.v - j.v
    u_ij = i.u - j.u

    d_ij = float(np.linalg.norm(p_ij))
    if d_ij < 1e-3:
        return 1.0 # protect against divide by 0

    p_hat = p_ij / d_ij

    ratio = (d_ij - config.d_safe) / d_ij
    R_d = 1.0 - (ratio ** 2)

    v_closing = -float(np.dot(p_hat, v_ij))
    R_v = np.clip(v_closing / (2 * (config.pursuer_v_max)), 0.0, 1.0)

    u_closing = -float(np.dot(p_hat, u_ij))
    R_u = np.clip(u_closing / (2 * (config.pursuer_u_max)), 0.0, 1.0)

    k = 1 - ((1 - config.w_d * R_d) * (1.0 - config.w_v * R_v) * (1.0 - config.w_u * R_u))

    return float(k)

def compute_uncertainty(reachability: Reachability, prev_sender_state: AgentState, curr_sender_state: AgentState, receiver_state: AgentState, curr_time: float) -> float:

        # last time i -> j
        prev_i = prev_sender_state
        curr_i = curr_sender_state
        j = receiver_state

        if prev_i is None or j is None:
            return 1.0  # no info, communicate

        tau_ij = curr_time - prev_i.t
        r = reachability.compute_radius(tau_ij)

        tau_ji = curr_time - j.t
        pred_pj = reachability.compute_sphere_set(j, tau_ji).center

        d_ij = float(np.linalg.norm(curr_i.p - pred_pj))
        d_ij = max(d_ij, 1e-3)

        u = r / (r + d_ij)

        return float(u)