from pathlib import Path
from dotenv import load_dotenv
import os

PROJECT_ROOT = Path(__file__).resolve().parents[2]
ENV_PATH = PROJECT_ROOT / ".env"

def load_env():
    """프로젝트 공통 .env 파일을 로드한다."""
    load_dotenv(ENV_PATH, override=True)

def get_env(key: str, default=None):
    """환경 변수를 읽기 쉽게 가져오는 함수."""
    return os.getenv(key, default)
