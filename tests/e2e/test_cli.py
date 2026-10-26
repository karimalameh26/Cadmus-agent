from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest
from typer.testing import CliRunner

from cadmus.cli import app
from cadmus.config import load_config

runner = CliRunner()


def test_initialize_real_project(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    args = ["init", "--label", "invoice", "--label", "receipt", "--model", "my-model",
            "--reviewer", "reviewer-1"]
    result = runner.invoke(app, args)
    assert result.exit_code == 0, result.output
    cfg = load_config("cadmus.toml")
    assert cfg.provider.kind == "openai_compatible"
    assert cfg.provider.model == "my-model"
    assert cfg.task.labels == ["invoice", "receipt"]
    assert cfg.input.json_records.label_field == "ground_truth"
    assert not cfg.policy.approved_for_live
    assert list(tmp_path.iterdir()) == [tmp_path / "cadmus.toml"]
    original = (tmp_path / "cadmus.toml").read_bytes()
    duplicate = runner.invoke(app, args)
    assert duplicate.exit_code == 2
    assert json.loads(duplicate.stdout)["error"]["code"] == "config_exists"
    assert (tmp_path / "cadmus.toml").read_bytes() == original


def test_initialize_direct_files_and_invalid_settings(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    args = ["init", "--label", "invoice", "--model", "my-model", "--reviewer", "reviewer-1"]
    invalid = runner.invoke(app, [*args, "--mode", "unknown"])
    assert invalid.exit_code == 2
    assert not (tmp_path / "cadmus.toml").exists()
    result = runner.invoke(app, [*args, "--mode", "direct_files"])
    assert result.exit_code == 0, result.output
    cfg = load_config("cadmus.toml")
    assert cfg.input.roots == [tmp_path / "documents"]
    assert cfg.input.json_records is None


@pytest.mark.parametrize("mode", ["json_records", "direct_files"])
def test_live_http_classification(tmp_path: Path, monkeypatch, mode: str) -> None:
    monkeypatch.chdir(tmp_path)
    requests = []

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, format, *args) -> None:
            pass

        def do_POST(self) -> None:
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            requests.append((self.path, self.headers.get("Authorization"), body))
            answer = {"status": "success", "labels": ["invoice"], "evidence": ["Invoice"],
                      "explanation": "Invoice document", "confidence": 0.9}
            payload = json.dumps({"choices": [{"message": {"content": json.dumps(answer)}}],
                                  "usage": {"prompt_tokens": 20, "completion_tokens": 10}}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.01}, daemon=True)
    thread.start()
    try:
        args = ["init", "--label", "invoice", "--label", "receipt", "--model", "real-model",
                "--reviewer", "reviewer-1", "--mode", mode,
                "--base-url", f"http://127.0.0.1:{server.server_port}/v1"]
        assert runner.invoke(app, args).exit_code == 0
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        missing = runner.invoke(app, ["doctor"])
        assert missing.exit_code == 1
        assert not json.loads(missing.stdout)["ok"]
        assert requests == []
        monkeypatch.setenv("OPENAI_API_KEY", "test-credential")
        healthy = runner.invoke(app, ["doctor", "--check-connectivity"])
        assert healthy.exit_code == 0, healthy.output
        assert json.loads(healthy.stdout)["ok"]
        if mode == "json_records":
            source = tmp_path / "records.json"
            source.write_text(json.dumps([{"id": "doc-1", "text": "Invoice for consulting"}]), encoding="utf-8")
        else:
            source = tmp_path / "documents"
            source.mkdir()
            (source / "invoice.txt").write_text("Invoice for consulting", encoding="utf-8")
        result = runner.invoke(app, ["-q", "classify", str(source), "--output", "results.json"])
        assert result.exit_code == 0, result.output
        exported = json.loads((tmp_path / "results.json").read_text(encoding="utf-8"))
        assert exported["counts"] == {"success": 1}
        assert exported["results"][0]["labels"] == ["invoice"]
        path, authorization, body = requests[-1]
        assert path == "/v1/chat/completions"
        assert authorization == "Bearer test-credential"
        assert body["model"] == "real-model"
        assert body["response_format"]["json_schema"]["strict"] is True
        assert "Invoice for consulting" in json.dumps(body["messages"])
        assert "test-credential" not in result.output
    finally:
        server.shutdown()
        thread.join(timeout=5)
        server.server_close()


def test_cli_end_to_end_json_stdout_and_exit_codes(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    assert runner.invoke(app, ["demo", "init", "."]).exit_code == 0

    bad = runner.invoke(app, ["-c", "missing.toml", "config", "validate"])
    assert bad.exit_code == 2 and json.loads(bad.stdout)["error"]["code"] == "config_not_found"

    imp = runner.invoke(app, ["dataset", "import", "records_batch1.json", "--verified-by", "reviewer-1"])
    assert imp.exit_code == 0 and json.loads(imp.stdout)["accepted"] == 90
    assert "Import finished" in imp.stderr

    snap = json.loads(runner.invoke(app, ["-q", "dataset", "snapshot"]).stdout)["snapshot_id"]
    opt = runner.invoke(app, ["-q", "optimize", snap])
    assert opt.exit_code == 0
    result = json.loads(opt.stdout)
    assert result["decision"]["outcome"] == "promoted"

    logs = runner.invoke(app, ["runs", "logs", result["run_id"], "--stage", "final_gate", "--format", "jsonl"])
    lines = [json.loads(line) for line in logs.stdout.splitlines()]
    assert lines and all(e["stage"] == "final_gate" for e in lines)
    assert any(e["event_type"] == "gate.consumed" for e in lines)

    report = json.loads(runner.invoke(app, ["runs", "report", result["run_id"]]).stdout)
    assert "decision.json" in report["artifacts"]

    cls = runner.invoke(app, ["-q", "classify", "records_batch1.json", "--summary-only"])
    assert cls.exit_code == 0 and json.loads(cls.stdout)["counts"]

    for fmt in ("json", "jsonl"):
        output = tmp_path / f"predictions.{fmt}"
        args = ["-q", "classify", "records_batch1.json", "--summary-only", "--output", str(output),
                "--output-format", fmt]
        exported = runner.invoke(app, args)
        assert exported.exit_code == 0, exported.output
        assert "results" not in json.loads(exported.stdout)
        text = output.read_text(encoding="utf-8")
        records = json.loads(text)["results"] if fmt == "json" else [json.loads(line) for line in text.splitlines()]
        assert len(records) == 90
        duplicate = runner.invoke(app, args)
        assert duplicate.exit_code == 2
        assert output.read_text(encoding="utf-8") == text

    bad_snap = runner.invoke(app, ["-q", "optimize", "snap-nope"])
    assert bad_snap.exit_code == 2 and json.loads(bad_snap.stdout)["error"]["code"] == "snapshot_not_found"
