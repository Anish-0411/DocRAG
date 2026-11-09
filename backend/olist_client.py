
from typing import Literal

import httpx


OlistRole = Literal["sales", "analyst"]


class OlistClientError(Exception):
    """Safe error raised when the Olist API cannot complete a request."""

    def __init__(self, message: str, status_code: int):
        super().__init__(message)
        self.message = message
        self.status_code = status_code


class OlistClient:
    QUERY_PATH = "/api/v1/sql-rag/query"
    SCHEMA_PATH = "/api/v1/sql-rag/schema"
    MAX_RESPONSE_BYTES = 1_000_000

    def __init__(
        self,
        base_url: str,
        sales_token: str,
        analyst_token: str | None = None,
        timeout_seconds: float = 20.0,
        transport: httpx.BaseTransport | None = None,
    ):
        if not base_url.strip():
            raise ValueError("Olist base URL is required.")

        if not sales_token.strip():
            raise ValueError("Olist sales token is required.")

        if timeout_seconds <= 0:
            raise ValueError("Timeout must be positive.")

        self.base_url = base_url.rstrip("/")
        self.tokens = {
            "sales": sales_token,
            "analyst": analyst_token,
        }

        self.timeout = httpx.Timeout(
            timeout_seconds,
            connect=5.0,
        )

        self._client = httpx.Client(
            base_url=self.base_url,
            timeout=self.timeout,
            follow_redirects=False,
            trust_env=False,
            transport=transport,
            headers={"Accept": "application/json"},
        )

    def _get_token(self, role: OlistRole) -> str:
        if role not in ("sales", "analyst"):
            raise ValueError("Unsupported Olist role.")

        token = self.tokens.get(role)

        if not token:
            raise OlistClientError(
                f"The Olist {role} credential is not configured.",
                503,
            )

        return token

    def _request(
        self,
        method: str,
        path: str,
        role: OlistRole,
        payload: dict | None = None,
    ) -> dict:
        token = self._get_token(role)

        try:
            with self._client.stream(
                method,
                path,
                json=payload,
                headers={"Authorization": f"Bearer {token}"},
            ) as response:

                if response.status_code == 401:
                    raise OlistClientError(
                        "Olist authentication failed.",
                        502,
                    )

                if response.status_code == 403:
                    raise OlistClientError(
                        "The configured Olist role is not authorized for this request.",
                        403,
                    )

                if response.status_code == 429:
                    raise OlistClientError(
                        "Olist is temporarily busy. Try again later.",
                        503,
                    )

                if response.status_code >= 500:
                    raise OlistClientError(
                        "The Olist service is temporarily unavailable.",
                        503,
                    )

                if response.status_code >= 400:
                    raise OlistClientError(
                        "Olist rejected the request.",
                        502,
                    )

                body = bytearray()

                for chunk in response.iter_bytes():
                    body.extend(chunk)

                    if len(body) > self.MAX_RESPONSE_BYTES:
                        raise OlistClientError(
                            "The Olist response exceeded the allowed size.",
                            502,
                        )

                try:
                    data = httpx.Response(
                        status_code=response.status_code,
                        headers=response.headers,
                        content=bytes(body),
                    ).json()
                except ValueError:
                    raise OlistClientError(
                        "Olist returned an invalid response.",
                        502,
                    ) from None

                if not isinstance(data, dict):
                    raise OlistClientError(
                        "Olist returned an unexpected response format.",
                        502,
                    )

                return data

        except OlistClientError:
            raise

        except httpx.TimeoutException:
            raise OlistClientError(
                "The Olist request timed out.",
                504,
            ) from None

        except httpx.RequestError:
            raise OlistClientError(
                "Could not connect to the Olist service.",
                503,
            ) from None

    def query(
        self,
        question: str,
        max_rows: int = 100,
        role: OlistRole = "sales",
    ) -> dict:
        if not isinstance(question, str) or not 3 <= len(question.strip()) <= 1000:
            raise ValueError("Question must contain 3–1000 characters.")

        if not isinstance(max_rows, int) or isinstance(max_rows, bool):
            raise ValueError("max_rows must be an integer.")

        if not 1 <= max_rows <= 1000:
            raise ValueError("max_rows must be between 1 and 1000.")

        data = self._request(
            "POST",
            self.QUERY_PATH,
            role,
            {
                "question": question.strip(),
                "max_rows": max_rows,
            },
        )

        required = {"request_id", "answer", "evidence", "metadata"}

        if not required.issubset(data):
            raise OlistClientError(
                "Olist returned an incomplete query response.",
                502,
            )

        if not isinstance(data["answer"], str):
            raise OlistClientError(
                "Olist returned an invalid answer.",
                502,
            )

        if not isinstance(data["evidence"], dict):
            raise OlistClientError(
                "Olist returned invalid evidence.",
                502,
            )

        if not isinstance(data["metadata"], dict):
            raise OlistClientError(
                "Olist returned invalid metadata.",
                502,
            )

        return data

    def get_schema(self, role: OlistRole = "sales") -> dict:
        return self._request(
            "GET",
            self.SCHEMA_PATH,
            role,
        )

    def close(self):
        self._client.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

