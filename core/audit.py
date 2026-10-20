import json
from datetime import datetime

from core.config import AUDIT_DIR


class AuditLogger:
    def __init__(self, run_id: str):
        self.run_id = run_id
        self.log_path = AUDIT_DIR / f"{run_id}.jsonl"

    def log(self, agent: str, action: str, **kwargs) -> None:
        entry = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "run_id": self.run_id,
            "agent": agent,
            "action": action,
            **kwargs,
        }
        with self.log_path.open("a", encoding="utf-8") as log_file:
            log_file.write(json.dumps(entry) + "\n")

    def get_logs(self) -> list[dict]:
        if not self.log_path.exists():
            return []

        with self.log_path.open("r", encoding="utf-8") as log_file:
            return [json.loads(line) for line in log_file if line.strip()]