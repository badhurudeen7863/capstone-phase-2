"""DB -> API roundtrip: create, read, update, delete + CSV bulk upload."""
from __future__ import annotations

import io


def test_crud_roundtrip(client, headers):
    cats = client.get("/api/v1/categories", headers=headers).json()
    cat_id = cats[0]["category_id"]

    created = client.post(
        "/api/v1/expenses",
        headers=headers,
        json={"amount": 999.5, "transaction_date": "2026-08-02", "category_id": cat_id, "notes": "roundtrip"},
    )
    assert created.status_code == 201
    txn = created.json()
    assert txn["amount"] == 999.5 and txn["notes"] == "roundtrip"

    listed = client.get("/api/v1/expenses", headers=headers, params={"search": "roundtrip"}).json()
    assert any(t["id"] == txn["id"] for t in listed)

    updated = client.put(
        f"/api/v1/expenses/{txn['id']}", headers=headers, json={"amount": 1234.0}
    )
    assert updated.status_code == 200 and updated.json()["amount"] == 1234.0

    deleted = client.delete(f"/api/v1/expenses/{txn['id']}", headers=headers)
    assert deleted.status_code == 204
    listed = client.get("/api/v1/expenses", headers=headers, params={"search": "roundtrip"}).json()
    assert not any(t["id"] == txn["id"] for t in listed)


def test_month_filter(client, headers, seeded_user):
    rows = client.get("/api/v1/expenses", headers=headers, params={"month": "2026-08"}).json()
    assert rows and all(str(t["transaction_date"]).startswith("2026-08") for t in rows)
    bad = client.get("/api/v1/expenses", headers=headers, params={"month": "2026-13"})
    assert bad.status_code == 422


def test_csv_bulk_upload(client, headers):
    csv_bytes = io.BytesIO(
        b"Txn Date,Narration,Debit\n"
        b"01/08/2026,UBER TRIP,340.50\n"
        b"2026-08-03,SWIGGY PAYMENT,512.00\n"
        b"05 Aug 2026,TNEB ELECTRICITY,1899.00\n"
        b"\"Aug 7, 2026\",APOLLO PHARMACY,260.00\n"
        b"bad-date,UNKNOWN,10.00\n"
    )
    r = client.post(
        "/api/v1/expenses/upload-csv",
        headers=headers,
        files={"file": ("upload.csv", csv_bytes, "text/csv")},
    )
    assert r.status_code == 200
    rep = r.json()
    assert rep["inserted"] == 4 and rep["skipped"] == 1
    assert rep["category_mapping"]["UBER TRIP"] == "Transport"
    assert rep["category_mapping"]["TNEB ELECTRICITY"] == "Utilities"
    assert len(rep["date_formats_seen"]) == 4


def test_ownership_enforced(client, headers):
    created = client.post(
        "/api/v1/expenses",
        headers=headers,
        json={"amount": 100, "transaction_date": "2026-08-05", "category_id": 1},
    ).json()
    other = client.post(
        "/api/v1/auth/register",
        json={"name": "Other", "email": "other@example.com", "password": "secret123"},
    ).json()
    other_h = {"Authorization": f"Bearer {other['access_token']}"}
    assert client.get(f"/api/v1/expenses/{created['id']}", headers=other_h).status_code in (404, 405)
    assert client.put(f"/api/v1/expenses/{created['id']}", headers=other_h, json={"amount": 1}).status_code == 404
    assert client.delete(f"/api/v1/expenses/{created['id']}", headers=other_h).status_code == 404
