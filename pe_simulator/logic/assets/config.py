"""
Configuration parameters for 3D pursuit-evasion simulation
"""

from dataclasses import dataclass

@dataclass
class SimConfig:
    
    # Agent parameters
    num_pursuers: int = 3 # number of pursuers
    num_evaders: int = 2 # number of evaders
    dt: float = 0.02 # time step 
    
    # Dynamics - Pursuers
    pursuer_v_max: float = 10 # pursuer max velocity
    pursuer_u_max: float = 5 # pursuer max control input
    
    # Dynamics - Evaders 
    evader_v_max: float = 7.5 # evader max velocity
    evader_u_max: float = 3 # evader max control input
    
    # Constraints
    d_safe: float = 2 # safety constraint d_safe
    d_capture: float = 5.0  # capture constraint
    d_sense: float = 50.0 # sensing range constraint
    alpha_0: float = .75 # controller gains
    alpha_1: float = 1 # controller gains
    p_soft: float = 1e3 # sensing range slack
    w_d: float = .6 # preemptive distance weight
    w_v: float = .25 # preemptive velocity weight
    w_u: float = .15 # preemptive control weight
    
    # Communication (broadcast-based)
    comm_strategy: str = "preemptive"  # "periodic", "none", "full", "preemptive", "event"
    periodic_comm_interval: float = 5 # in Hz
    broadcast_range: float = 200  # broadcast radius (inf = unlimited)
    risk_threshold: float = .25 # preemptive risk threshold
    max_comm_interval: float = 5 # max T between comms

    # Simulation
    sim_duration: float = 100.0 # max sim duration
    random_initial_positions: bool = True  # random starting positions
    
    # Visualization
    show_reachable_sets: bool = True
    random_seed: int = 119 # seed for visualization
    reachable_set_visualization_distance: float = 12.0  # only show ellipsoids within this distance
    reachable_set_alpha: float = 0.2  # transparency of ellipsoids (0-1)
    reachable_set_resolution: int = 10  # points per dimension for ellipsoid mesh
    world_bounds: float = 100.0  # +/- bounds for visualization and simulation
    boundary_margin: float = 5.0  # distance from boundary to start repulsion
    boundary_strength: float = 0.3  # fraction of max accel for boundary repulsion
    track_pursuer_risk: int = 1
