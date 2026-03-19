"""
Pipeline trace system for Analytics Copilot.

Records structured traces of each agent step during query execution,
capturing timing, status, and detail for monitoring and debugging.
"""

import time
import uuid
from datetime import datetime, timezone
from typing import Optional


class PipelineTrace:
    """Records structured per-agent step traces during query execution."""

    def __init__(self, question: str):
        self._query_id = str(uuid.uuid4())
        self._question = question
        self._timestamp = datetime.now(timezone.utc).isoformat()
        self._steps: list[dict] = []
        self._start_time = time.perf_counter()
        self._step_start: Optional[float] = None
        self._step_agent: Optional[str] = None

    def start_step(self, agent: str) -> None:
        """Begin timing a pipeline step."""
        self._step_agent = agent
        self._step_start = time.perf_counter()

    def end_step(self, status: str, detail: str) -> None:
        """End the current step and record it."""
        duration_ms = 0
        if self._step_start is not None:
            duration_ms = int((time.perf_counter() - self._step_start) * 1000)
        self._steps.append({
            "agent": self._step_agent or "Unknown",
            "status": status,
            "duration_ms": duration_ms,
            "detail": detail,
        })
        self._step_start = None
        self._step_agent = None

    def finish(
        self,
        success: bool = True,
        error: Optional[str] = None,
        final_sql: Optional[str] = None,
        row_count: Optional[int] = None,
    ) -> dict:
        """Finalize the trace and return the complete trace dict."""
        total_duration_ms = int((time.perf_counter() - self._start_time) * 1000)
        return {
            "query_id": self._query_id,
            "question": self._question,
            "timestamp": self._timestamp,
            "steps": self._steps,
            "total_duration_ms": total_duration_ms,
            "final_sql": final_sql,
            "row_count": row_count,
            "success": success,
            "error": error,
        }
