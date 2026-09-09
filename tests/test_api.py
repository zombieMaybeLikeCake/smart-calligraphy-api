"""最基本的 API 測試。TestClient 用 `with` 進入，觸發 app/main.py 的
lifespan——這代表跑這個檔案會真的把 word/stroke 兩個模型載入 GPU（或
DEVICE=cpu 時降級），不是純粹的 mock，跑起來要幾秒鐘是預期行為。"""
import pytest
from fastapi.testclient import TestClient

from app.main import app, settings


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["models"]["word"]["loaded"] is True
    assert body["models"]["stroke"]["loaded"] is True


def test_predict_word_rejects_non_image(client):
    r = client.post("/predict/word", files={"file": ("a.txt", b"hello", "text/plain")})
    assert r.status_code == 400


def test_predict_stroke_rejects_non_image(client):
    r = client.post("/predict/stroke", files={"file": ("a.txt", b"hello", "text/plain")})
    assert r.status_code == 400


def test_predict_requires_api_key(client, monkeypatch):
    monkeypatch.setattr(settings, "api_key", "test-only-key-for-this-test")
    r = client.post("/predict/word", files={"file": ("a.txt", b"hello", "text/plain")})
    # 沒帶 X-API-Key：就算檔案本身也不合法，也要先卡在驗證回 401，不是 400
    # ——驗證這個順序本身就是在測 dependencies=[Depends(verify_api_key)]
    # 真的比 UploadFile 的 body 解析先執行。
    assert r.status_code == 401
