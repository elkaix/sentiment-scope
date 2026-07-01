import io


def _upload(client, content: bytes, name="data.csv"):
    return client.post("/api/analyze/csv", files={"file": (name, io.BytesIO(content), "text/csv")})


def test_csv_happy_path(client_with_model):
    csv_bytes = b"text\nI love this\nawful experience\n"
    resp = _upload(client_with_model, csv_bytes)
    assert resp.status_code == 200
    body = resp.json()
    assert len(body["results"]) == 2
    assert body["results"][1]["text"] == "awful experience"


def test_csv_missing_text_column_is_400(client_with_model):
    resp = _upload(client_with_model, b"comment\nhello\n")
    assert resp.status_code == 400
    assert "text" in resp.json()["detail"]


def test_csv_over_500_rows_is_400(client_with_model):
    csv_bytes = b"text\n" + b"row\n" * 501
    resp = _upload(client_with_model, csv_bytes)
    assert resp.status_code == 400


def test_csv_skips_blank_rows(client_with_model):
    csv_bytes = b"text\nhello\n\n   \nworld\n"
    resp = _upload(client_with_model, csv_bytes)
    assert resp.status_code == 200
    assert len(resp.json()["results"]) == 2


def test_csv_non_utf8_is_400(client_with_model):
    resp = _upload(client_with_model, b"text\n\xff\xfe broken \xff\n")
    assert resp.status_code == 400
