import json

import requests

from utils import ServerPreset


def test_completion_stream_keepalive_while_queued(monkeypatch):
    server = ServerPreset.tinyllama2()
    monkeypatch.setenv("LLAMA_ARG_STREAM_KEEPALIVE", "1")
    server.n_slots = 1
    server.n_threads = 1
    server.n_predict = -1
    server.enable_ctx_shift = True
    server.start()
    url = f"http://{server.server_host}:{server.server_port}/completion"

    # Hold the only slot so the second request waits independently of model speed.
    with requests.post(url, json={
        "prompt": "Once upon a time",
        "stream": True,
        "n_predict": -1,
        "ignore_eos": True,
    }, stream=True, timeout=5) as busy:
        assert busy.status_code == 200
        busy_lines = busy.iter_lines(chunk_size=1)
        assert next(busy_lines).startswith(b"data: ")
        with requests.post(url, json={
            "prompt": "A different story",
            "stream": True,
            "n_predict": 1,
            "sse_ping_interval": -1,
        }, stream=True, timeout=3) as queued:
            assert queued.status_code == 200
            lines = queued.iter_lines(chunk_size=1)
            for _ in range(3):
                assert next(lines) == b": keepalive"
                assert next(lines) == b""
            busy.close()
            chunks = [json.loads(line[6:]) for line in lines if line.startswith(b"data: ")]
            assert chunks[-1]["stop"] is True
            assert chunks[-1]["tokens_predicted"] == 1
            assert any(chunk["content"] for chunk in chunks[:-1])
