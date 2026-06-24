"""
Data structures for agent states and communication messages
"""

import numpy as np
from dataclasses import dataclass, field
from enum import Enum

@dataclass
class AgentState:
    p: np.ndarray  # [x, y, z]
    v: np.ndarray  # [vx, vy, vz]
    u: np.ndarray = field(default_factory=lambda: np.zeros(3))  # [ux, uy, uz]
    t: float = 0.0
    
    def copy(self):
        return AgentState(
            p=self.p.copy(),
            v=self.v.copy(),
            u=self.u.copy(),
            t=self.t
        )

@dataclass
class ReachableSet:
    center: np.ndarray  # center of reachable set
    velocity: np.ndarray # predicted velocity of agent
    radius: float  # radius of sphere
    radius_inf: float # safe distance inflated radius
    tau: float  # time since last comm
    volume: float = 0.0 # volume of the reachable set

@dataclass
class AgentMessage:
    sender_id: int
    p: np.ndarray
    v: np.ndarray
    u: np.ndarray          
    t: float


@dataclass
class ConstraintDiagnostic:
    neighbor_id: int
    distance: float
    safe_radius: float
    tau: float
    reach_radius: float
    h: float
    h_dot: float
    psi1: float
    A: np.ndarray
    b: float
    max_lhs: float
    margin: float

class CommStrategy(Enum):
    NONE = "none"
    PERIODIC = "periodic"
    FULL = "full"
    PREEMPTIVE = "preemptive"
    EVENT = "event"
