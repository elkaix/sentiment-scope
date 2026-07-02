import io

from app.routes import MAX_CSV_BYTES


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


def test_csv_blank_rows_dont_count_toward_row_limit(client_with_model):
    # 300 non-empty rows interleaved with 250 whitespace-only rows: >500 raw
    # CSV data rows but only 300 non-empty texts, so this must be accepted.
    # (A whitespace char, not a truly empty line, is required so the row
    # survives csv.DictReader's own blank-line skipping and still reaches
    # the row-limit check.)
    rows = "row\n \n" * 250 + "row\n" * 50
    csv_bytes = b"text\n" + rows.encode()
    resp = _upload(client_with_model, csv_bytes)
    assert resp.status_code == 200
    assert len(resp.json()["results"]) == 300


def test_csv_exactly_500_non_empty_rows_with_blanks_is_200(client_with_model):
    # Exactly 500 non-empty texts plus extra whitespace-only rows: still at
    # the limit, not over it, so this must be accepted.
    rows = "row\n \n" * 500
    csv_bytes = b"text\n" + rows.encode()
    resp = _upload(client_with_model, csv_bytes)
    assert resp.status_code == 200
    assert len(resp.json()["results"]) == 500


def test_csv_non_utf8_is_400(client_with_model):
    resp = _upload(client_with_model, b"text\n\xff\xfe broken \xff\n")
    assert resp.status_code == 400


def test_csv_over_max_bytes_is_413(client_with_model):
    # One oversized field, not millions of rows: cheap to build, still
    # exercises the raw byte-size guard before any CSV parsing happens.
    csv_bytes = b"text\n" + b"x" * (MAX_CSV_BYTES + 1)
    resp = _upload(client_with_model, csv_bytes)
    assert resp.status_code == 413
    assert "5MB" in resp.json()["detail"]


def test_csv_far_over_max_bytes_still_413(client_with_model):
    # A body many times the limit still 413s: the endpoint uses a BOUNDED read
    # (file.read(MAX_CSV_BYTES + 1)), so the guard fires on the first MAX+1 bytes
    # and can't be defeated by a body larger than RAM. Behavior is all we can
    # assert here; the bounded read is what keeps that behavior cheap.
    csv_bytes = b"text\n" + b"x" * (MAX_CSV_BYTES * 3)
    resp = _upload(client_with_model, csv_bytes)
    assert resp.status_code == 413
