
import httpx
import pytest

from olist_client import OlistClient, OlistClientError


SALES_TOKEN = "test-sales-token-not-a-real-credential"
ANALYST_TOKEN = "test-analyst-token-not-a-real-credential"


def success_response(request):
    assert request.method == "POST"
    assert request.url.path == "/api/v1/sql-rag/query"
    assert request.headers["Authorization"] == f"Bearer {SALES_TOKEN}"

    return httpx.Response(
        200,
        json={
            "request_id": "test-request-id",
            "answer": "There are 100 orders.",
            "evidence": {
                "sql": "SELECT COUNT(*) FROM orders",
                "source_views": ["orders"],
                "columns": ["count"],
                "rows": [{"count": 100}],
                "row_count": 1,
                "row_limit": 100,
                "truncated": False,
            },
            "metadata": {
                "strategy": "placeholder",
                "intent": "order_count",
                "elapsed_ms": 10,
                "warnings": [],
            },
        },
    )


def test_sales_query_uses_sales_token():
    transport = httpx.MockTransport(success_response)

    with OlistClient(
        base_url="http://testserver",
        sales_token=SALES_TOKEN,
        transport=transport,
    ) as client:
        result = client.query("How many orders are there?")

    assert result["answer"] == "There are 100 orders."
    assert result["evidence"]["source_views"] == ["orders"]


def test_analyst_role_uses_analyst_token():
    def handler(request):
        assert request.headers["Authorization"] == f"Bearer {ANALYST_TOKEN}"
        return httpx.Response(200, json={"schema": "analytics", "tables": {}})

    transport = httpx.MockTransport(handler)

    with OlistClient(
        base_url="http://testserver",
        sales_token=SALES_TOKEN,
        analyst_token=ANALYST_TOKEN,
        transport=transport,
    ) as client:
        result = client.get_schema(role="analyst")

    assert result["schema"] == "analytics"


def test_analyst_does_not_fall_back_to_sales():
    with OlistClient(
        base_url="http://testserver",
        sales_token=SALES_TOKEN,
    ) as client:
        with pytest.raises(OlistClientError) as error:
            client.query(
                "What is the average review score?",
                role="analyst",
            )

    assert error.value.status_code == 503


def test_forbidden_response_is_not_retried_with_analyst():
    calls = []

    def handler(request):
        calls.append(request.headers["Authorization"])
        return httpx.Response(403, json={"error": "forbidden"})

    transport = httpx.MockTransport(handler)

    with OlistClient(
        base_url="http://testserver",
        sales_token=SALES_TOKEN,
        analyst_token=ANALYST_TOKEN,
        transport=transport,
    ) as client:
        with pytest.raises(OlistClientError) as error:
            client.query("What is the average review score?")

    assert error.value.status_code == 403
    assert calls == [f"Bearer {SALES_TOKEN}"]


def test_invalid_max_rows_is_rejected_before_http_call():
    transport = httpx.MockTransport(
        lambda request: pytest.fail("HTTP request should not be made")
    )

    with OlistClient(
        base_url="http://testserver",
        sales_token=SALES_TOKEN,
        transport=transport,
    ) as client:
        with pytest.raises(ValueError):
            client.query("How many orders are there?", max_rows=5000)


def test_incomplete_response_is_rejected():
    transport = httpx.MockTransport(
        lambda request: httpx.Response(200, json={"answer": "Incomplete"})
    )

    with OlistClient(
        base_url="http://testserver",
        sales_token=SALES_TOKEN,
        transport=transport,
    ) as client:
        with pytest.raises(OlistClientError) as error:
            client.query("How many orders are there?")

    assert error.value.status_code == 502

