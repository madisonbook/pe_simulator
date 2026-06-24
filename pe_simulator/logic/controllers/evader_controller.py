import numpy as np
from typing import List
from logic.assets.data_structures import AgentState
from logic.assets.config import SimConfig


class EvaderController:
    
    def __init__(self, config: SimConfig):
        self.config = config
        self.v_max = config.evader_v_max
        self.u_max= config.evader_u_max
        
    def compute_control(self, evader_state: AgentState, 
                       pursuer_positions: List[np.ndarray]) -> np.ndarray:
        """Evade by moving away from nearest pursuer"""
        if len(pursuer_positions) == 0:
            return np.zeros(3)
        
        # Find nearest pursuer
        min_distance = float('inf')
        nearest_pursuer = None
        for p_pos in pursuer_positions:
            distance = np.linalg.norm(evader_state.p - p_pos)
            if distance < min_distance:
                min_distance = distance
                nearest_pursuer = p_pos
        
        # Move away from nearest pursuer
        away_direction = evader_state.p - nearest_pursuer
        distance = np.linalg.norm(away_direction)
        
        if distance < 0.01:
            # Random direction to avoid divide by 0
            theta = np.random.uniform(0, 2*np.pi)
            phi = np.random.uniform(0, np.pi)
            away_unit = np.array([
                np.sin(phi) * np.cos(theta),
                np.sin(phi) * np.sin(theta),
                np.cos(phi)
            ])
        else:
            away_unit = away_direction / distance # unit vector

        desired_vel = away_unit * self.v_max
        control = desired_vel - evader_state.v
        
        # Clip to u_max
        control_mag = np.linalg.norm(control)
        if control_mag > self.u_max:
            control = control / control_mag * self.u_max
            
        return control
