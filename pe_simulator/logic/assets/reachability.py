
import numpy as np
from logic.assets.data_structures import AgentState, ReachableSet


class Reachability:
    def __init__(self, u_max: float, v_max: float, d_safe: float, dt: float):

        self.u_max = u_max
        self.v_max = v_max
        self.d_safe = d_safe
        self.dt = dt

    def compute_sphere_set(self, state: 'AgentState', tau: float) -> 'ReachableSet':      
        """
        Computes spherical overapproximation of reachable set.
        """
        pos = state.p
        vel = state.v

        center = pos + vel * tau

        radius = .5 * self.u_max * (tau ** 2)
        #radius = max(radius, self.v_max * self.dt)
        radius_inf = radius + self.d_safe
        volume = (4/3) * np.pi * (radius ** 3)
        
        return ReachableSet(
            center=center, 
            velocity=vel,     
            radius=radius, 
            radius_inf = radius_inf,
            tau = tau,
            volume=volume,
        )
    
    def compute_radius(self, tau: float) -> 'float':
        radius = .5 * self.u_max * (tau ** 2)
        #radius = max(radius, self.v_max * self.dt)
        return radius
    
    def compute_radius_inf(self, tau: float) -> 'float':
        radius = .5 * self.u_max * (tau ** 2) 
        #radius = max(radius, self.v_max * self.dt)
        return radius + self.d_safe

    # ------------------------------------------------------------------
    # Visualization
    # ------------------------------------------------------------------
    
    @staticmethod
    def get_sphere_surface_points(center: np.ndarray, radius: float,
                                n_points: int = 20) -> np.ndarray:
        """
        Surface points for 3-D plotting in visualization
        """
        u = np.linspace(0, 2 * np.pi, n_points)
        v = np.linspace(0,     np.pi, n_points)
        sx = np.outer(np.cos(u), np.sin(v))
        sy = np.outer(np.sin(u), np.sin(v))
        sz = np.outer(np.ones_like(u), np.cos(v))
        return center[:, None, None] + radius * np.stack([sx, sy, sz], axis=0)