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


def read_trace_request_files() -> list[Path]:
    trace_dir = Path(server.http_trace_dir)
    assert trace_dir.exists(), f"missing trace dir at {trace_dir}"
    return sorted(trace_dir.glob("*.jsonl"))


def read_trace_records(path: Path) -> list[dict]:
    assert path.exists(), f"missing trace file at {path}"
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


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

    files = read_trace_request_files()
    assert len(files) == 1
    records = read_trace_records(files[0])
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

    files = read_trace_request_files()
    assert len(files) == 1
    records = read_trace_records(files[0])
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

    files = read_trace_request_files()
    assert len(files) == 1
    records = read_trace_records(files[0])
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

    files = read_trace_request_files()
    assert len(files) == 1
    records = read_trace_records(files[0])
    assert records[0]["type"] == "request_start"
    assert records[-1]["type"] == "response_finish"
    assert records[-1]["stream"] is True
    stream_events = [record for record in records if record["type"] == "stream_event"]
    assert stream_events
    assert [record["trace_seq"] for record in records] == list(range(1, len(records) + 1))
    assert any("response.output_text.delta" in record.get("body_text", "") or "response.output_text.delta" in record.get("body_text_excerpt", "") for record in stream_events)
    assert all(record["request_id"] == records[0]["request_id"] for record in records)


def test_http_trace_writes_one_file_per_request():
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

    files = read_trace_request_files()
    assert len(files) == 2
    all_records = [read_trace_records(path) for path in files]
    assert [records[0]["type"] for records in all_records] == ["request_start", "request_start"]
    assert [records[-1]["type"] for records in all_records] == ["response_finish", "response_finish"]
    assert files[0].name < files[1].name


def test_http_trace_uses_tmp_file_while_stream_active():
    global server
    server.start()
    stream = server.make_stream_request("POST", "/v1/responses", data={
        "model": "gpt-4.1",
        "input": [
            {"role": "system", "content": "Book"},
            {"role": "user", "content": "What is the best book"},
        ],
        "max_output_tokens": 16,
        "temperature": 0.8,
        "stream": True,
    })
    first_chunk = next(stream)
    assert first_chunk
    trace_dir = Path(server.http_trace_dir)
    tmp_files = sorted(trace_dir.glob("*.jsonl.tmp"))
    assert len(tmp_files) == 1
    assert tmp_files[0].read_text()

    chunks = list(stream)
    assert chunks
    server.stop()

    assert not list(trace_dir.glob("*.jsonl.tmp"))
    assert len(read_trace_request_files()) == 1


def test_http_trace_finalizes_active_request_on_shutdown():
    global server
    server.start()
    stream = server.make_stream_request("POST", "/v1/responses", data={
        "model": "gpt-4.1",
        "input": [
            {"role": "system", "content": "Book"},
            {"role": "user", "content": "What is the best book"},
        ],
        "max_output_tokens": 64,
        "temperature": 0.8,
        "stream": True,
    })
    first_chunk = next(stream)
    assert first_chunk
    trace_dir = Path(server.http_trace_dir)
    assert len(list(trace_dir.glob("*.jsonl.tmp"))) == 1

    server.stop()

    assert not list(trace_dir.glob("*.jsonl.tmp"))
    files = read_trace_request_files()
    assert len(files) == 1
    records = read_trace_records(files[0])
    assert records[-1]["type"] == "response_finish"
    assert records[-1]["status"] in (200, 499)
    if records[-1]["status"] == 499:
        assert records[-1]["aborted"] is True
