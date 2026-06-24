
from typing import List, Dict, Optional
from logic.assets.data_structures import AgentState

class MessageInbox:
    """
    Stores the most recent state received from each other agent.

    Keyed by sender_id so callers can easily retrieve a specific agent's
    last known (p, v, u, t) for risk-score computation.
    """

    def __init__(self):
        self._messages: Dict[int, AgentState] = {}

    def store(self, sender_id: int, state: AgentState) -> None:
        self._messages[sender_id] = state.copy()

    def get(self, sender_id: int) -> Optional[AgentState]:
        return self._messages.get(sender_id)

    def all(self) -> List[AgentState]:
        return list(self._messages.values())

    def known_senders(self) -> List[int]:
        return list(self._messages.keys())

