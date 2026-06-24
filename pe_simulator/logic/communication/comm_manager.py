import numpy as np
from typing import List, Dict, Tuple
from logic.assets.data_structures import AgentState, CommStrategy
from logic.assets.config import SimConfig
from logic.communication.message_inbox import MessageInbox
from logic.communication.preempt_risk import compute_kinematic, compute_uncertainty

class CommunicationManager:

    def __init__(self, config: SimConfig, reachability):
        self.config = config
        self.strategy = CommStrategy(config.comm_strategy)
        self.reachability = reachability

        # (sender_id, receiver_id) -> AgentState that was transmitted
        self.last_transmitted_state: Dict[Tuple[int, int], AgentState] = {}

        self.message_events: List[Tuple[float, int, int]] = []
        self.inboxes: Dict[int, MessageInbox] = {}
        self.current_states: Dict[int, AgentState] = {}

    def update_states(self, states: List[AgentState]) -> None:
        for agent_id, state in enumerate(states):
            self.current_states[agent_id] = state

    def get_inbox(self, agent_id: int) -> MessageInbox:
        if agent_id not in self.inboxes:
            self.inboxes[agent_id] = MessageInbox()
        return self.inboxes[agent_id]

    def deliver_message(self, receiver_id: int, sender_id: int, state: AgentState) -> None:
        self.get_inbox(receiver_id).store(sender_id, state)

    def should_send_to(self, sender_id: int, receiver_id: int, curr_time: float, k: float, u: float) -> bool:
        if self.strategy == CommStrategy.NONE:
            return False

        if self.strategy == CommStrategy.FULL:
            return True

        if self.strategy == CommStrategy.PERIODIC:
            return self._should_send_periodic(sender_id, receiver_id, curr_time)
        
        if self.strategy == CommStrategy.EVENT:
            return self._should_send_event(sender_id, receiver_id)

        if self.strategy == CommStrategy.PREEMPTIVE:

            score = 1 - ((1 - k) * (1 - u))
            if score >= self.config.risk_threshold:
                #print(f"(1 - risk)(1 - unc) {sender_id}\u2192{receiver_id}: risk={risk:.3f}, unc = {uncertainty:.3f}")
                return True

            # 4. Heartbeat
            last_sender_state = self.last_transmitted_state.get((sender_id, receiver_id), None)
            if last_sender_state is None:
                return True

            tau = curr_time - last_sender_state.t

            if tau >= self.config.max_comm_interval:
                return True

            return False

        return False 

    def _should_send_periodic(self, sender_id: int, receiver_id: int, curr_time: float) -> bool:
        last_sender_state = self.last_transmitted_state.get((sender_id, receiver_id), None)
        if last_sender_state is None:
            return True

        tau = curr_time - last_sender_state.t

        period = 1.0 / self.config.periodic_comm_interval
        return tau >= period
    
    def _should_send_event(self, sender_id, receiver_id):
        
        last_tx = self.last_transmitted_state.get((sender_id, receiver_id))
        if last_tx is None:
            return True  
        
        current = self.current_states.get(sender_id)
        delta_p = 4 * self.config.d_safe
        delta_v = self.config.pursuer_v_max * .25
        delta_u = self.config.pursuer_u_max * .25

        if np.linalg.norm(current.p - last_tx.p) > delta_p:
            return True
        if np.linalg.norm(current.v - last_tx.v) > delta_v:
            return True
        if np.linalg.norm(current.u - last_tx.u) > delta_u:
            return True

        return False
    
    def record_send(self, sender_id: int, receiver_id: int, curr_time: float, sent_state: AgentState) -> None:

        self.last_transmitted_state[(sender_id, receiver_id)] = sent_state.copy()
        self.message_events.append((curr_time, sender_id, receiver_id))

    def process_outgoing_messages(self,
                                   sender_id: int,
                                   sender_state: AgentState,
                                   receiver_ids: List[int],
                                   curr_time: float,
                                   ) -> Tuple[List[int], Dict[int, float]]:

        recipients:  List[int]        = []
        risk_scores: Dict[int, float] = {}

        for receiver_id in receiver_ids:
            if receiver_id == sender_id:
                continue

            receiver_state = self.get_inbox(sender_id).get(receiver_id)
            last_sender_state = self.last_transmitted_state.get((sender_id, receiver_id), None)

            k = compute_kinematic(self.config, sender_state, receiver_state, curr_time)
            u = compute_uncertainty(self.reachability, last_sender_state, sender_state, receiver_state, curr_time)

            send        = self.should_send_to(
                sender_id, receiver_id, curr_time, k, u
            )

            risk_scores[receiver_id] = 1 - ( 1 - k ) * ( 1 - u ), k, u

            if send:
                self.deliver_message(receiver_id, sender_id, sender_state)
                self.record_send(sender_id, receiver_id, curr_time, sender_state)
                recipients.append(receiver_id)

        return recipients, risk_scores

    def get_total_messages(self) -> int:
        return len(self.message_events)