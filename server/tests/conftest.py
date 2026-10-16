import json
from pathlib import Path

import pytest

FIXTURES = Path(__file__).parent / "fixtures"


def fixture_json(name: str):
    return json.loads((FIXTURES / name).read_text())


@pytest.fixture(autouse=True)
def data_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("FH_SKIP_ENCODERS", "1")
    monkeypatch.delenv("REDIS_URL", raising=False)
    monkeypatch.delenv("FH_ALLOW_PRIVATE_URLS", raising=False)
    from framehound import config, db

    config.reset_settings()
    db.init()
    yield tmp_path / "data"
    config.reset_settings()
