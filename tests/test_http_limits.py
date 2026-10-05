import asyncio

from medical_rag.http_limits import RequestBodyLimit


def test_chunked_body_limit_is_enforced_before_json_parsing():
    messages = iter(
        [
            {"type": "http.request", "body": b"a" * 8192, "more_body": True},
            {"type": "http.request", "body": b"b" * 8193, "more_body": False},
        ]
    )
    responses = []

    async def receive():
        return next(messages)

    async def send(message):
        responses.append(message)

    async def downstream(scope, receive, send):
        raise AssertionError("Oversized input must not reach JSON parsing")

    asyncio.run(
        RequestBodyLimit(downstream)(
            {"type": "http", "method": "POST"},
            receive,
            send,
        )
    )
    assert responses[0]["status"] == 413


def test_chunked_body_is_replayed_without_loss():
    messages = iter(
        [
            {"type": "http.request", "body": b'{"query":', "more_body": True},
            {"type": "http.request", "body": b'"heart"}', "more_body": False},
        ]
    )
    bodies = []

    async def receive():
        return next(messages)

    async def send(message):
        pass

    async def downstream(scope, receive, send):
        bodies.append((await receive())["body"])

    asyncio.run(
        RequestBodyLimit(downstream)(
            {"type": "http", "method": "POST"},
            receive,
            send,
        )
    )
    assert bodies == [b'{"query":"heart"}']
