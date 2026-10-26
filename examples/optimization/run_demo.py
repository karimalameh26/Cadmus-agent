"""SDK walkthrough of the full lifecycle with the credential-free fake provider.

Run from the repository root:  python examples/optimization/run_demo.py
The fake provider's improvement is scripted; it demonstrates orchestration, not real-world accuracy.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

from cadmus import Framework, load_config
from cadmus.demo import write_demo
from cadmus.observability import ConsoleSubscriber


def main() -> None:
    work = Path(tempfile.mkdtemp(prefix="cadmus-demo-"))
    write_demo(work, batch=1)
    write_demo(work, batch=2)
    cfg = load_config(work / "cadmus.toml")
    with Framework(cfg, subscribers=[ConsoleSubscriber(level="INFO")]) as fw:
        fw.import_dataset([work / "records_batch1.json"], verified_by="reviewer-1")
        snap = fw.snapshot()["snapshot_id"]
        first = fw.optimize(snap)
        print("cycle 1:", first["decision"]["outcome"], first["decision"]["reason_codes"])

        fw.import_dataset([work / "records_batch2.json"], verified_by="reviewer-1")
        second = fw.optimize(fw.snapshot()["snapshot_id"])
        print("cycle 2:", second["decision"]["outcome"], second["decision"]["reason_codes"])

        result = fw.classify([work / "records_batch2.json"])
        print("classified:", result["counts"], "with", result["bundle_id"])
        print("trace a run with: python -m cadmus -c", work / "cadmus.toml", "runs logs", second["run_id"])


if __name__ == "__main__":
    main()
