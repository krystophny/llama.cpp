import json
from pathlib import Path

import pytest

from utils import *

server: ServerProcess


@pytest.fixture(autouse=True)
def create_server(tmp_path: Path):
    global server
    server = ServerPreset.tinyllama2()
    server.http_trace_dir = str(tmp_path / "trace")
    server.http_trace_max_bytes = 4096


def read_trace_records() -> list[dict]:
    trace_path = Path(server.http_trace_dir) / "http-trace.jsonl"
    assert trace_path.exists(), f"missing trace file at {trace_path}"
    return [json.loads(line) for line in trace_path.read_text().splitlines() if line.strip()]


def test_http_trace_chat_completion_non_stream():
    global server
    server.start()
    res = server.make_request("POST", "/v1/chat/completions", data={
        "max_tokens": 8,
        "messages": [
            {"role": "system", "content": "Book"},
            {"role": "user", "content": "What is the best book"},
        ],
    })
    assert res.status_code == 200
    server.stop()

    records = read_trace_records()
    assert [record["type"] for record in records] == ["request_start", "response_finish"]
    assert records[0]["path"] == "/v1/chat/completions"
    assert records[0]["trace_seq"] == 1
    assert records[1]["trace_seq"] == 2
    assert records[0]["body_json"]["messages"][1]["content"] == "What is the best book"
    assert records[1]["status"] == 200
    assert records[1]["stream"] is False
    assert records[1]["body_json"]["choices"][0]["message"]["role"] == "assistant"
    assert records[0]["request_id"] == records[1]["request_id"]


def test_http_trace_chat_completion_stream():
    global server
    server.start()
    stream = server.make_stream_request("POST", "/v1/chat/completions", data={
        "max_tokens": 8,
        "messages": [
            {"role": "system", "content": "Book"},
            {"role": "user", "content": "What is the best book"},
        ],
        "stream": True,
    })
    chunks = list(stream)
    assert chunks
    server.stop()

    records = read_trace_records()
    assert records[0]["type"] == "request_start"
    assert records[-1]["type"] == "response_finish"
    assert records[-1]["stream"] is True
    stream_events = [record for record in records if record["type"] == "stream_event"]
    assert stream_events
    assert stream_events[0]["sequence"] == 1
    assert [record["trace_seq"] for record in records] == list(range(1, len(records) + 1))
    assert all(record["path"] == "/v1/chat/completions" for record in records if record["type"] == "request_start")
    assert all(record["request_id"] == records[0]["request_id"] for record in records)


def test_http_trace_responses_non_stream():
    global server
    server.start()
    res = server.make_request("POST", "/v1/responses", data={
        "model": "gpt-4.1",
        "input": [
            {"role": "system", "content": "Book"},
            {"role": "user", "content": "What is the best book"},
        ],
        "max_output_tokens": 8,
        "temperature": 0.8,
    })
    assert res.status_code == 200
    server.stop()

    records = read_trace_records()
    assert [record["type"] for record in records] == ["request_start", "response_finish"]
    assert records[0]["path"] == "/v1/responses"
    assert records[0]["trace_seq"] == 1
    assert records[1]["trace_seq"] == 2
    assert records[0]["body_json"]["input"][0]["role"] == "system"
    assert records[1]["status"] == 200
    assert records[1]["stream"] is False
    assert records[1]["body_json"]["output"][0]["type"] == "message"


def test_http_trace_responses_stream():
    global server
    server.start()
    stream = server.make_stream_request("POST", "/v1/responses", data={
        "model": "gpt-4.1",
        "input": [
            {"role": "system", "content": "Book"},
            {"role": "user", "content": "What is the best book"},
        ],
        "max_output_tokens": 8,
        "temperature": 0.8,
        "stream": True,
    })
    chunks = list(stream)
    assert chunks
    server.stop()

    records = read_trace_records()
    assert records[0]["type"] == "request_start"
    assert records[-1]["type"] == "response_finish"
    assert records[-1]["stream"] is True
    stream_events = [record for record in records if record["type"] == "stream_event"]
    assert stream_events
    assert [record["trace_seq"] for record in records] == list(range(1, len(records) + 1))
    assert any("response.output_text.delta" in record.get("body_text", "") or "response.output_text.delta" in record.get("body_text_excerpt", "") for record in stream_events)
    assert all(record["request_id"] == records[0]["request_id"] for record in records)


def test_http_trace_appends_multiple_requests_with_monotonic_trace_seq():
    global server
    server.start()
    first = server.make_request("POST", "/v1/chat/completions", data={
        "max_tokens": 8,
        "messages": [
            {"role": "system", "content": "Book"},
            {"role": "user", "content": "What is the best book"},
        ],
    })
    second = server.make_request("POST", "/v1/responses", data={
        "model": "gpt-4.1",
        "input": [
            {"role": "system", "content": "Book"},
            {"role": "user", "content": "What is the best book"},
        ],
        "max_output_tokens": 8,
    })
    assert first.status_code == 200
    assert second.status_code == 200
    server.stop()

    records = read_trace_records()
    assert len({record["request_id"] for record in records if record["type"] == "request_start"}) == 2
    assert [record["trace_seq"] for record in records] == list(range(1, len(records) + 1))
