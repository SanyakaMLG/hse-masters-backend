from typing import Generator
import pytest
from fastapi.testclient import TestClient
from main import app


@pytest.fixture
def app_client() -> Generator[TestClient, None, None]:
    return TestClient(app)
