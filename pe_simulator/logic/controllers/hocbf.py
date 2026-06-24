from typing import Tuple, Dict
import numpy as np
from logic.assets.data_structures import AgentState, ReachableSet
from logic.assets.reachability import Reachability

class hocbf:

    def __init__(self, u_max: float, v_max: float, d_safe: float, d_sense: float, alpha_0: float, alpha_1: float, dt: float):

        self.u_max = u_max
        self.v_max = v_max
        self.d_safe = d_safe
        self.d_sense = d_sense
        self.alpha_0 = alpha_0
        self.alpha_1 = alpha_1
        self.dt = dt
        self.reachability = Reachability(u_max, v_max, d_safe, dt)
    
    def sphere_derivatives(self,
        i: AgentState,
        j: ReachableSet
    ) -> Tuple[float, float, np.ndarray, float]:

        tau = j.tau
        p_hat_j = j.center
        p_rel = i.p - p_hat_j
        v_rel = i.v - j.velocity

        r = self.reachability.compute_radius_inf(tau)
        r2    = r * r
        r_dot = self.u_max * tau
        r_dotdot = self.u_max

        h       = float(p_rel @ p_rel) - r2
        h_dot   = 2.0 * float(p_rel @ v_rel) - 2.0 * r * r_dot
        Lf2_h   = 2.0 * float(v_rel @ v_rel) - 2.0 * r_dot ** 2 - 2.0 * r * r_dotdot
        Lg_Lf_h = 2.0 * p_rel

        '''r_k  = self.reachability.compute_radius_inf(tau)
        r_k1 = self.reachability.compute_radius_inf(tau + self.dt)
        r_k2 = self.reachability.compute_radius_inf(tau + 2.0 * self.dt)
 
        # Relative positions (u = 0)
        p_rel_k1      = p_rel + self.dt * v_rel
        p_rel_k2_free = p_rel + 2.0 * self.dt * v_rel
 
        # Barrier values
        h_k       = float(p_rel @ p_rel) - r_k  ** 2
        h_k1      = float(p_rel_k1 @ p_rel_k1) - r_k1 ** 2
        h_k2_free = float(p_rel_k2_free @ p_rel_k2_free) - r_k2 ** 2

        #Lg_h_k2 = 2.0 * self.dt ** 2 * p_rel_k2_free
        dt2 = self.dt ** 2
        Lg_h_k2 = dt2 * (3.0 * p_rel_k2_free + (self.alpha_0 - 1.0) * p_rel_k1)'''

        #return h_k, h_k1, h_k2_free, Lg_h_k2
        return h, h_dot, Lg_Lf_h, Lf2_h
    
    def sphere_hocbf(
        self,
        state_i:    AgentState,
        sphere_set: ReachableSet
    ) -> Tuple[np.ndarray, float, Dict]:
        
        #r_inflated = sphere_set.radius_inf

        h, h_dot, Lg_Lf_h, Lf2_h = self.sphere_derivatives(state_i, sphere_set)

        psi_1 = h_dot + self.alpha_0 * h
        a     = Lg_Lf_h
        b     = -Lf2_h - (self.alpha_0 + self.alpha_1) * h_dot - (self.alpha_0 * self.alpha_1 * h) 

        #h_k, h_k1, h_k2_free, Lg_h_k2 = self.sphere_derivatives(state_i, sphere_set)
 
         #a_0, a_1 = self.alpha_0, self.alpha_1

        #psi_1 = h_k1 + (a_0 - 1.0) * h_k
 
        # Affine CBF constraint coefficients
        '''a = Lg_h_k2
        b = -(
            h_k2_free
            + (a_0 + a_1 - 2.0) * h_k1
            + (a_0 - 1.0) * (a_1 - 1.0) * h_k
        )'''
        #b += (self.alpha_0 + self.alpha_1) * self.u_max * self.dt 

        #b_max = self.max_accel * np.sum(np.abs(a))
        #b     = min(b, b_max)

        #lambda_max = 1.0 / (r_inflated ** 2)
        #b += 2.0 * self.max_accel * lambda_max * self.config.dt

        '''
        if b > 0 and h > 0:
            print(
                f"h={h:.3f}",
                f"hdot={h_dot:.3f}",
                f"alpha0*h={self.alpha_0*h:.3f}",
                f"psi1={psi_1:.3f}",
                f"Lf2h={Lf2_h:.3f}",
                f"rhs={b:.3f}",
                f"phi={self.u_max*np.linalg.norm(a)-b:.3f}"
            )'''

        return a, b, {
            'h': h, 'h_dot': h_dot, 'psi_1': psi_1,
            'Lf2_h': Lf2_h, 'Lg_Lf_h': Lg_Lf_h, 'a': a, 'b': b
        }
    
    def sensing_hocbf(
        self,
        pursuer_state: AgentState,
        evader_state:  AgentState,
    ) -> Tuple[np.ndarray, float]:
        
        R     = self.d_sense
        p_rel = pursuer_state.p - evader_state.p
        v_e   = evader_state.v if evader_state.v is not None else np.zeros(3)
        v_rel = pursuer_state.v - v_e

        h  = R ** 2 - float(p_rel @ p_rel)
        h_dot = -2.0 * float(p_rel @ v_rel)
        Lf2_h = -2.0 * float(v_rel @ v_rel)
        Lg_Lf_h = -2.0 * p_rel

        a     = Lg_Lf_h
        b     = -Lf2_h - (self.alpha_0 + self.alpha_1) * h_dot - (self.alpha_0 * self.alpha_1 * h)

        return a, b

    