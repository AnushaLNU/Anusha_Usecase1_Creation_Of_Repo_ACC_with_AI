import os
from pathlib import Path

from dotenv import load_dotenv

BASE_DIR = Path(__file__).parent.parent

load_dotenv(BASE_DIR / ".env")

PROFILES_DIR = BASE_DIR / "data" / "profiles"
STTM_DIR = BASE_DIR / "data" / "sttm"
BRONZE_DIR = BASE_DIR / "data" / "bronze_layer"
SILVER_DIR = BASE_DIR / "data" / "silver_layer"
GOLD_DIR = BASE_DIR / "data" / "gold_layer"
REPORTS_DIR = BASE_DIR / "data" / "reports"
AUDIT_DIR = BASE_DIR / "data" / "audit_logs"

LLM_PROVIDER = os.getenv("LLM_PROVIDER", "anthropic").strip().lower()
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", os.environ.get("API_KEY", ""))
ANTHROPIC_MODEL = os.getenv(
    "ANTHROPIC_MODEL", os.getenv("MODEL", "claude-sonnet-4-6")
)
GITHUB_TOKEN = os.environ.get("GITHUB_TOKEN", "")
GITHUB_MODEL = os.environ.get("GITHUB_MODEL", "openai/gpt-4.1-mini")


def ensure_dirs():
    for directory in (
        PROFILES_DIR,
        STTM_DIR,
        BRONZE_DIR,
        SILVER_DIR,
        GOLD_DIR,
        REPORTS_DIR,
        AUDIT_DIR,
    ):
        directory.mkdir(parents=True, exist_ok=True)