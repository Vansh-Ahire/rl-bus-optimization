import uuid
from typing import Dict, Optional
from environment import BusRoutingEnv
from tasks import get_task

class SessionStore:
    """Manages environment instances for multiple concurrent episodes."""
    def __init__(self):
        self.sessions: Dict[str, BusRoutingEnv] = {}

    def create_session(self, task_id: str = "task_2") -> str:
        """Create a new environment session and return its ID."""
        session_id = str(uuid.uuid4())
        task = get_task(task_id)
        self.sessions[session_id] = task.build_env()
        return session_id

    def get_env(self, session_id: str) -> Optional[BusRoutingEnv]:
        """Retrieve the environment for a given session ID."""
        return self.sessions.get(session_id)

    def close_session(self, session_id: str):
        """Remove a session from the store."""
        if session_id in self.sessions:
            del self.sessions[session_id]

# Singleton instance
store = SessionStore()
