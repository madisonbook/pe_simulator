import numpy as np
import osqp
import scipy.sparse as sp
from typing import Dict, List, Tuple
from logic.assets.data_structures import AgentState, ReachableSet, CommStrategy
from logic.assets.config import SimConfig
from logic.assets.pe_assignment import PursuerAssignmentManager
from logic.controllers.hocbf import hocbf
import logging

class CBFController:

    def __init__(self, config: SimConfig):
        self.config = config
        self.v_max = self.config.pursuer_v_max
        self.u_max = self.config.pursuer_u_max
        self.d_safe = self.config.d_safe
        self.d_sense = self.config.d_sense

        self.alpha_0 = self.config.alpha_0
        self.alpha_1 = self.config.alpha_1
        self.p_soft = self.config.p_soft

        self.assignment_manager = PursuerAssignmentManager()
        self.solver = osqp.OSQP()
        self.hocbf = hocbf(self.u_max, self.v_max, self.d_safe, self.d_sense, self.alpha_0, self.alpha_1, self.config.dt)

    def update_assignments(self, pursuer_states: Dict[int, AgentState], evader_states: Dict[int, AgentState] ) -> None:
        self.assignment_manager.update(pursuer_states, evader_states)

    def compute_nominal_control(self, pursuer_state: AgentState, evader_state: AgentState, local_idx: int = 0, local_count: int = 1) -> np.ndarray:

        n        = max(local_count, 1)
        ev_vel   = evader_state.v if evader_state.v is not None else np.zeros(3)
        rel      = pursuer_state.p - evader_state.p
        dist_to_evader = float(np.linalg.norm(rel))

        # -----------------------------------------------------------------
        # Fibonacci-sphere angular slot — shared across all phases so
        # formation directions stay consistent from approach to capture.
        # -----------------------------------------------------------------
        golden = np.pi * (3.0 - np.sqrt(5.0))
        idx    = local_idx % n
        if n == 1:
            ev_speed = np.linalg.norm(ev_vel)
            # Single pursuer: slot is behind the evader's travel direction
            orbit_dir = (-ev_vel / ev_speed) if ev_speed > 1e-3 else np.array([1.0, 0.0, 0.0])
        else:
            y         = 1.0 - (idx / (n - 1)) * 2.0
            r_ring    = np.sqrt(max(1.0 - y * y, 0.0))
            theta     = golden * idx
            orbit_dir = np.array([r_ring * np.cos(theta),
                                   r_ring * np.sin(theta),
                                   y])

        # -----------------------------------------------------------------
        # COLLAPSE PHASE  (dist ≤ 1.5 · d_capture)
        #
        # Instead of diving straight at the evader centre, each pursuer
        # targets its own angular slot at r_orbit (< d_capture) from the
        # evader.  A tangential swirl term makes the approach curve in
        # rather than collide head-on with other pursuers.
        # -----------------------------------------------------------------
        collapse_radius = self.config.d_capture * 1.5

        if dist_to_evader <= collapse_radius:
            # Orbit point: just inside capture radius at this pursuer's slot
            r_orbit = self.config.d_capture * 0.75
            slot     = evader_state.p + orbit_dir * r_orbit
            to_slot  = slot - pursuer_state.p
            dist_to_slot = float(np.linalg.norm(to_slot))

            if dist_to_slot < 1e-3:
                # Already at orbit slot — circulate tangentially so the
                # pursuer doesn't just sit stationary inside capture radius.
                tangent = np.cross(orbit_dir, np.array([0.0, 0.0, 1.0]))
                if np.linalg.norm(tangent) < 1e-3:
                    tangent = np.cross(orbit_dir, np.array([1.0, 0.0, 0.0]))
                tangent  /= np.linalg.norm(tangent)
                desired_v = tangent * self.v_max * 0.5 + ev_vel
            else:
                slot_hat = to_slot / dist_to_slot

                # Tangential swirl: perpendicular to both orbit_dir and the
                # current approach direction.  Gives a curving, non-head-on
                # trajectory into the capture zone.
                tangent  = np.cross(orbit_dir, slot_hat)
                tang_mag = np.linalg.norm(tangent)
                tangent  = (tangent / tang_mag) if tang_mag > 1e-3 else np.zeros(3)

                # approach_frac → 1 when far, → 0 when close to slot.
                # More swirl as the pursuer nears its slot (avoids overshoot).
                approach_frac = float(np.clip(dist_to_slot / r_orbit, 0.0, 1.0))
                swirl_frac    = 0.45 * (1.0 - 0.5 * approach_frac)

                desired_v = (
                    slot_hat * self.v_max * approach_frac
                    + tangent * self.v_max * swirl_frac
                    + ev_vel
                )

            d_mag = np.linalg.norm(desired_v)
            if d_mag > self.v_max:
                desired_v = desired_v / d_mag * self.v_max

            u_nom = 2.0 * (desired_v - pursuer_state.v)
            mag   = np.linalg.norm(u_nom)
            if mag > self.u_max:
                u_nom = u_nom / mag * self.u_max
            return u_nom

        # -----------------------------------------------------------------
        # SURROUND / APPROACH PHASES  (dist > 1.5 · d_capture)
        #
        # Formation slot uses the same orbit_dir so the position the
        # pursuer holds in the surround ring is consistent with the slot
        # it will occupy during collapse.
        # -----------------------------------------------------------------
        if n == 1:
            r_form = max(self.config.d_capture * 0.3, 1.0)
        else:
            r_min_safe = self.d_safe / (2.0 * np.sin(np.pi / n))
            r_form     = max(r_min_safe, 1.0)
            r_form     = min(r_form, self.config.d_capture * 0.85)

        offset  = orbit_dir * r_form
        slot    = evader_state.p + offset
        to_slot = slot - pursuer_state.p
        dist    = float(np.linalg.norm(to_slot))
        surround_radius = 2.0 * r_form

        if dist_to_evader > surround_radius:
            # Far approach: fly straight to formation slot at full speed
            if dist < 1e-3:
                u_nom = -2.0 * pursuer_state.v
            else:
                desired_vel = to_slot / dist * self.v_max
                u_nom       = 2.0 * (desired_vel - pursuer_state.v)
        else:
            # Surround orbit: hold formation ring, match evader velocity
            rel_xy = np.array([rel[0], rel[1], 0.0])
            r_xy   = float(np.linalg.norm(rel_xy))

            if r_xy < 1e-3:
                rel_xy = np.array([offset[0], offset[1], 0.0])
                r_xy   = float(np.linalg.norm(rel_xy)) or 1e-3

            radial_hat  = rel_xy / r_xy
            tangent_hat = np.array([-radial_hat[1], radial_hat[0], 0.0])

            radial_err  = r_form - r_xy
            z_err       = offset[2] - rel[2]

            v_orbit = self.v_max * 0.6
            desired_vel = (
                tangent_hat  * v_orbit
                + radial_hat * (4.0 * radial_err)
                + np.array([0.0, 0.0, 3.0 * z_err])
                + ev_vel
            )
            dv_mag = np.linalg.norm(desired_vel)
            if dv_mag > self.v_max:
                desired_vel = desired_vel / dv_mag * self.v_max

            u_nom = 2.0 * (desired_vel - pursuer_state.v)

        mag = np.linalg.norm(u_nom)
        if mag > self.u_max:
            u_nom = u_nom / mag * self.u_max
        return u_nom
    
    def _ball_constraint_rows(self, radius: float):
        """
        Polyhedral approximation of ||u|| <= radius.

        Returns rows A_ball, l_ball, u_ball such that

            n_i^T u <= radius

        for many directions n_i on the unit sphere.
        """

        directions = []
        N_DIR = 100

        golden = np.pi * (3.0 - np.sqrt(5.0))

        for k in range(N_DIR):
            z = 1.0 - 2.0 * (k + 0.5) / N_DIR
            r = np.sqrt(max(0.0, 1.0 - z * z))
            theta = golden * k

            directions.append([
                r * np.cos(theta),
                r * np.sin(theta),
                z
            ])

        A_ball = np.zeros((N_DIR, 4))

        for i, n in enumerate(directions):
            A_ball[i, :3] = n

        l_ball = np.full(N_DIR, -np.inf)
        u_ball = np.full(N_DIR, radius)

        return A_ball, l_ball, u_ball

    def solve_cbf_qp(
        self,
        pursuer_id:          int,
        pursuer_state:       AgentState,
        evader_state:        AgentState,
        neighbor_reach_sets: Dict[int, ReachableSet],
        curr_time:        float,
        local_idx:           int = 0,
        local_count:         int = 1,
    ) -> Tuple[np.ndarray, bool, Dict]:

        u_nom = self.compute_nominal_control(
            pursuer_state, evader_state, local_idx, local_count
        )

        metrics: Dict = {
            'min_h': float('inf'),
            'active_constraints': 0,
            'cbf_infos': [],
            'cbf_type': 'HOCBF',
        }

        comm = CommStrategy(self.config.comm_strategy)
        if comm == CommStrategy.NONE or (hasattr(comm, 'value') and comm.value == 'none'):
            u = _clip_to_ball(u_nom, self.u_max)
            return u, True, metrics

        hard_A: List[np.ndarray] = []
        hard_b: List[float]      = []

        for neighbor_id, reach_set in neighbor_reach_sets.items():
            a, b, info = self.hocbf.sphere_hocbf(pursuer_state, reach_set)

            phi = self.u_max * np.linalg.norm(a) - b

            
            if phi < 0:
                '''print(
                    f"INFEASIBLE CBF "
                    f"t={curr_time:.2f} "
                    f"agent={pursuer_id} "
                    f"neighbor={neighbor_id} "
                    f"phi={phi:.3f} "
                    f"h={info['h']:.3f}"
                )'''

            metrics['cbf_infos'].append({'neighbor_id': neighbor_id, **info})
            metrics['min_h'] = min(metrics['min_h'], info['h'])

            if np.linalg.norm(a) < 1e-9:
                continue

            hard_A.append(a)
            hard_b.append(b)

        a_sense, b_sense = self.hocbf.sensing_hocbf(pursuer_state, evader_state)

        #ub = self.u_max / np.sqrt(3)
        N  = 4

        if hard_A:
            A_hard_mat = np.stack(hard_A)
            hard_rows  = np.hstack([A_hard_mat, np.zeros((len(hard_A), 1))])
            b_hard_vec = np.array(hard_b)
        else:
            hard_rows  = np.empty((0, N))
            b_hard_vec = np.empty(0)

        soft_row   = np.append(a_sense, 1.0).reshape(1, N)
        b_soft_vec = np.array([b_sense])

        #A_full = np.vstack([hard_rows, soft_row])
        #b_full = np.concatenate([b_hard_vec, b_soft_vec])

        A_full = hard_rows
        b_full = b_hard_vec

        P = sp.diags([1.0, 1.0, 1.0, self.p_soft], format='csc')
        q = np.array([-u_nom[0], -u_nom[1], -u_nom[2], 0.0])

        if A_full.shape[0] > 0:
            A_cbf = A_full
            l_cbf = b_full
            u_cbf = np.full_like(b_full, np.inf)
        else:
            A_cbf = np.zeros((0, N))
            l_cbf = np.zeros(0)
            u_cbf = np.zeros(0)

        #A_bounds = np.eye(N)
        #l_bounds = np.array([-ub, -ub, -ub, 0.0])
        #u_bounds = np.array([ub, ub, ub, np.inf])

        #A = np.vstack([A_cbf, A_bounds])
        #l = np.concatenate([l_cbf, l_bounds])
        #u = np.concatenate([u_cbf, u_bounds])

        A_ball, l_ball, u_ball = self._ball_constraint_rows(radius=self.u_max)

        A = np.vstack([A_cbf, A_ball])
        l = np.concatenate([l_cbf, l_ball])
        u = np.concatenate([u_cbf, u_ball])

        A_sparse = sp.csc_matrix(A)

        '''
        print(
            f"agent={pursuer_id}",
            f"a_sense={a_sense}",
            f"b_sense={b_sense}"
        )

        assert np.all(np.isfinite(P.data))
        assert np.all(np.isfinite(q))
        assert np.all(np.isfinite(A))
        assert np.all(np.isfinite(l))
        assert not np.any(np.isnan(u))

        print("\nA shape =", A.shape)
        print("l =", l)
        print("u =", u)

        for i in range(len(l)):
            if l[i] > u[i]:
                print("BAD ROW", i, l[i], u[i])
        '''

        self.solver.setup(
            P=P,
            q=q,
            A=A_sparse,
            l=l,
            u=u,
            verbose=False,
            polish=False
        )

        res = self.solver.solve()

        if res.info.status in ['solved', 'solved inaccurate']:
            x = res.x
            u = x[:3]
            delta = float(x[3])
            feasible = True

            '''
            if A_full.shape[0] > 0:
                slack = A_full @ x - b_full
                feasible = np.all(slack >= -1e-6)
                metrics['active_constraints'] = int(np.sum(slack < 1e-4))
            else:
                feasible = True
            '''

            metrics['sensing_slack'] = delta
        else:
            feasible = False
            u = _clip_to_ball(-pursuer_state.v * 2.0, self.u_max)

        if not feasible:
            '''logging.error(
                f"QP failed: "
                f"status={res.info.status}, "
                f"status_val={res.info.status_val}, "
                f"time={curr_time:.2f}, "
                f"agent={pursuer_id}"
            )'''
        return u, feasible, metrics

def _clip_to_ball(v: np.ndarray, r: float) -> np.ndarray:
    mag = np.linalg.norm(v)
    return v if mag <= r else v / mag * r