"""Command-line interface. Results are JSON on stdout; progress and diagnostics go to stderr.

Exit codes: 0 completed (including a completed cycle that did not promote), 1 execution failure,
2 configuration/usage/access error, 3 input validation error, 4 audit persistence failure, 130 cancelled.
"""

from __future__ import annotations

import json
import sys
from collections.abc import Callable
from functools import wraps
from pathlib import Path
from typing import Any, TypeVar

import typer

from cadmus.config import initialize_config, load_config
from cadmus.errors import CadmusError, ConfigError
from cadmus.observability import ConsoleSubscriber
from cadmus.runtime import Framework

app = typer.Typer(no_args_is_help=True, add_completion=False, help="Ground-truth-gated self-improving classifier.")
config_app = typer.Typer(no_args_is_help=True, help="Configuration checks.")
inputs_app = typer.Typer(no_args_is_help=True, help="Validate and preview inputs.")
dataset_app = typer.Typer(no_args_is_help=True, help="Import evidence/labels and freeze snapshots.")
corrections_app = typer.Typer(no_args_is_help=True, help="Submit and review label corrections.")
prompts_app = typer.Typer(no_args_is_help=True, help="Prompt bundles, champions and rollback.")
runs_app = typer.Typer(no_args_is_help=True, help="Inspect, follow and resume runs.")
monitor_app = typer.Typer(no_args_is_help=True, help="Production monitoring and rollback.")
demo_app = typer.Typer(no_args_is_help=True, help="Synthetic demo data.")
for sub, name in ((config_app, "config"), (inputs_app, "inputs"), (dataset_app, "dataset"),
                  (corrections_app, "corrections"), (prompts_app, "prompts"), (runs_app, "runs"),
                  (monitor_app, "monitor"), (demo_app, "demo")):
    app.add_typer(sub, name=name)

F = TypeVar("F", bound=Callable[..., Any])


@app.callback()
def _root(ctx: typer.Context,
          config: Path = typer.Option(Path("cadmus.toml"), "--config", "-c", envvar="CADMUS_CONFIG",
                                      help="Path to the TOML configuration."),
          verbose: bool = typer.Option(False, "--verbose", "-v", help="Show per-document events on stderr."),
          quiet: bool = typer.Option(False, "--quiet", "-q", help="Only warnings and errors on stderr."),
          no_color: bool = typer.Option(False, "--no-color", help="Disable ANSI colors.")) -> None:
    ctx.obj = {"config": config, "verbose": verbose, "quiet": quiet, "no_color": no_color}


def _fw(ctx: typer.Context) -> Framework:
    o = ctx.find_root().obj
    cfg = load_config(o["config"])
    level = "WARNING" if o["quiet"] else ("DEBUG" if o["verbose"] else cfg.logging.console_level)
    console = ConsoleSubscriber(level=level, verbose=o["verbose"], color=False if o["no_color"] else None)
    return Framework(cfg, subscribers=[console])


def _out(obj: Any) -> None:
    sys.stdout.write(json.dumps(obj, indent=2, ensure_ascii=False, default=str) + "\n")
    sys.stdout.flush()


def handled(fn: F) -> F:
    @wraps(fn)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        try:
            return fn(*args, **kwargs)
        except CadmusError as exc:
            sys.stderr.write(f"error [{exc.code}]: {exc.message}\n")
            _out({"ok": False, "error": exc.to_dict()})
            raise typer.Exit(exc.exit_code) from None
        except KeyboardInterrupt:
            sys.stderr.write("cancelled\n")
            raise typer.Exit(130) from None
    return wrapper  # type: ignore[return-value]


# --- config / inputs ---------------------------------------------------------------------------------------------
@app.command("init")
@handled
def project_init(ctx: typer.Context,
             labels: list[str] = typer.Option(..., "--label", help="Repeat for each classification label."),
             model: str = typer.Option(..., help="Your provider's model or deployment name."),
             reviewer: str = typer.Option(..., help="Identity allowed to verify labels."),
             client_id: str = typer.Option("default"),
             mode: str = typer.Option("json_records", help="json_records | direct_files"),
             base_url: str = typer.Option("https://api.openai.com/v1"),
             api_key_env: str = typer.Option("OPENAI_API_KEY")) -> None:
    """Create a live-provider configuration without synthetic data. Never overwrites an existing config."""
    path = initialize_config(ctx.find_root().obj["config"], labels=labels, model=model, reviewer=reviewer,
                     client_id=client_id, mode=mode, base_url=base_url, api_key_env=api_key_env)
    _out({"config": str(path), "provider": "openai_compatible", "approved_for_live": False,
        "next_steps": [f"Export {api_key_env} in your shell (do not put secrets in the config).",
                 "Review input mappings, label definitions and policy thresholds in the config.",
                 "Run doctor --check-connectivity, then inputs preview and classify.",
                 "Approve the policy explicitly before running live optimization."]})


@config_app.command("validate")
@handled
def config_validate(ctx: typer.Context) -> None:
    """Validate configuration and print identities and warnings."""
    with _fw(ctx) as fw:
        _out(fw.validate_config())


@inputs_app.command("validate")
@handled
def inputs_validate(ctx: typer.Context, sources: list[Path]) -> None:
    """Validate inputs (JSON files or local files/directories) without classifying."""
    with _fw(ctx) as fw:
        _out(fw.validate_inputs(sources))


@inputs_app.command("preview")
@handled
def inputs_preview(ctx: typer.Context, sources: list[Path], limit: int = typer.Option(3, min=1)) -> None:
    """Show exactly what the model would see for the first N inputs (explicit local output, not logged)."""
    with _fw(ctx) as fw:
        _out(fw.preview(sources, limit))


# --- dataset -----------------------------------------------------------------------------------------------------
@dataset_app.command("import")
@handled
def dataset_import(ctx: typer.Context, sources: list[Path],
                   verified_by: str = typer.Option(None, help="Reviewer attesting source labels are trusted.")) -> None:
    """Normalize and store evidence; record source labels (pending unless --verified-by)."""
    with _fw(ctx) as fw:
        _out(fw.import_dataset(sources, verified_by))


@dataset_app.command("import-labels")
@handled
def dataset_import_labels(ctx: typer.Context, path: Path, verified_by: str = typer.Option(None)) -> None:
    """Import labels from a JSON array of {"name"|"document_id", "labels"}."""
    with _fw(ctx) as fw:
        _out(fw.import_labels(path, verified_by))


@dataset_app.command("snapshot")
@handled
def dataset_snapshot(ctx: typer.Context, seed: int = typer.Option(None)) -> None:
    """Freeze verified labels and evidence into grouped development/selection/gate splits."""
    with _fw(ctx) as fw:
        _out(fw.snapshot(seed))


@dataset_app.command("inspect")
@handled
def dataset_inspect(ctx: typer.Context, snapshot_id: str) -> None:
    with _fw(ctx) as fw:
        _out(fw.inspect_snapshot(snapshot_id))


@dataset_app.command("list")
@handled
def dataset_list(ctx: typer.Context) -> None:
    with _fw(ctx) as fw:
        _out(fw.datasets.list_snapshots())


# --- classify / evaluate / optimize ------------------------------------------------------------------------------
@app.command("classify")
@handled
def classify(ctx: typer.Context, sources: list[Path], bundle: str = typer.Option(None, help="Bundle id override."),
             summary_only: bool = typer.Option(False, help="Omit per-document results from stdout."),
             output: Path | None = typer.Option(None, "--output", "-o", help="Export to a new JSON or JSONL file."),
             output_format: str = typer.Option("json", help="json (full report) | jsonl (one result per line).")
             ) -> None:
    """Classify inputs with the champion bundle (or the baseline if no champion exists)."""
    if output_format not in ("json", "jsonl"):
        raise ConfigError("Output format must be json or jsonl", "output_format_invalid")
    if output is not None and (output.exists() or not output.parent.is_dir()):
        raise ConfigError("Output must be a new file in an existing directory", "output_unavailable")
    with _fw(ctx) as fw:
        result = fw.classify(sources, bundle)
        if output is not None:
            try:
                with output.open("x", encoding="utf-8") as stream:
                    if output_format == "jsonl":
                        for record in result["results"]:
                            stream.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
                    else:
                        stream.write(json.dumps(result, indent=2, ensure_ascii=False, default=str) + "\n")
            except OSError:
                raise ConfigError("Cannot write output; results remain available in run artifacts",
                                  "output_write_failed") from None
            result["output"] = str(output)
        if summary_only:
            result.pop("results", None)
        _out(result)


@app.command("evaluate")
@handled
def evaluate(ctx: typer.Context, snapshot_id: str,
             split: str = typer.Option("development", help="development | selection | gate (gate is consumed)."),
             bundle: str = typer.Option(None)) -> None:
    """Evaluate a bundle against verified labels in a snapshot split."""
    with _fw(ctx) as fw:
        _out(fw.evaluate(snapshot_id, split, bundle))


@app.command("optimize")
@handled
def optimize(ctx: typer.Context, snapshot_id: str) -> None:
    """Run one bounded optimization cycle. A non-promotion outcome is a successful completion (exit 0)."""
    with _fw(ctx) as fw:
        _out(fw.optimize(snapshot_id))


# --- corrections -------------------------------------------------------------------------------------------------
@corrections_app.command("add")
@handled
def corrections_add(ctx: typer.Context, document: str, labels: list[str],
                    submitted_by: str = typer.Option(..., "--by")) -> None:
    """Submit a correction (pending until a configured reviewer verifies it)."""
    with _fw(ctx) as fw:
        _out(fw.add_correction(document, labels, submitted_by))


@corrections_app.command("verify")
@handled
def corrections_verify(ctx: typer.Context, label_id: int, reviewer: str = typer.Option(...)) -> None:
    with _fw(ctx) as fw:
        _out(fw.verify_correction(label_id, reviewer))


@corrections_app.command("reject")
@handled
def corrections_reject(ctx: typer.Context, label_id: int, reviewer: str = typer.Option(...),
                       reason: str = typer.Option(...)) -> None:
    with _fw(ctx) as fw:
        _out(fw.reject_correction(label_id, reviewer, reason))


@corrections_app.command("list")
@handled
def corrections_list(ctx: typer.Context, status: str = typer.Option(None)) -> None:
    with _fw(ctx) as fw:
        _out(fw.list_corrections(status))


# --- prompts -----------------------------------------------------------------------------------------------------
@prompts_app.command("list")
@handled
def prompts_list(ctx: typer.Context) -> None:
    with _fw(ctx) as fw:
        _out(fw.list_prompts())


@prompts_app.command("show")
@handled
def prompts_show(ctx: typer.Context, bundle_id: str = typer.Argument(None)) -> None:
    with _fw(ctx) as fw:
        _out(fw.show_prompt(bundle_id))


@prompts_app.command("diff")
@handled
def prompts_diff(ctx: typer.Context, a: str, b: str) -> None:
    with _fw(ctx) as fw:
        sys.stdout.write(fw.diff_prompts(a, b) + "\n")


@prompts_app.command("baseline")
@handled
def prompts_baseline(ctx: typer.Context, snapshot_id: str) -> None:
    """Register the configured baseline as first champion if it passes absolute gates."""
    with _fw(ctx) as fw:
        _out(fw.register_baseline(snapshot_id))


@prompts_app.command("rollback")
@handled
def prompts_rollback(ctx: typer.Context, to: str = typer.Option(None, help="Target bundle (default: previous)."),
                     reason: str = typer.Option("manual")) -> None:
    with _fw(ctx) as fw:
        _out(fw.rollback(to, reason))


# --- runs --------------------------------------------------------------------------------------------------------
@runs_app.command("list")
@handled
def runs_list(ctx: typer.Context, limit: int = typer.Option(50)) -> None:
    with _fw(ctx) as fw:
        _out(fw.list_runs(limit))


@runs_app.command("show")
@handled
def runs_show(ctx: typer.Context, run_id: str) -> None:
    with _fw(ctx) as fw:
        _out(fw.show_run(run_id))


@runs_app.command("logs")
@handled
def runs_logs(ctx: typer.Context, run_id: str, follow: bool = typer.Option(False, "--follow", "-f"),
              stage: str = typer.Option(None), level: str = typer.Option("INFO"),
              event_type: str = typer.Option(None, "--type", help="Event type prefix, e.g. provider."),
              document: str = typer.Option(None), candidate: str = typer.Option(None),
              fmt: str = typer.Option("text", "--format", help="text | jsonl")) -> None:
    """Print (or follow) a run's ordered events. Works for running and finished runs."""
    with _fw(ctx) as fw:
        for e in fw.run_logs(run_id, stage=stage, level=level.upper(), event_type=event_type, document_id=document,
                             candidate_id=candidate, follow=follow):
            if fmt == "jsonl":
                sys.stdout.write(json.dumps(e, ensure_ascii=False, sort_keys=True) + "\n")
            else:
                extras = {k: v for k, v in e.items() if k not in {
                    "schema_version", "event_id", "timestamp", "sequence", "level", "event_type", "run_id", "stage",
                    "status", "message", "client_id"}}
                sys.stdout.write(f"{e['sequence']:>5} {e['timestamp']} {e['level']:<7} {e['stage']:<22} "
                                 f"{e['event_type']:<30} {e['status']:<10} {e['message']}"
                                 f"{'  ' + json.dumps(extras, default=str) if extras else ''}\n")
            sys.stdout.flush()


@runs_app.command("report")
@handled
def runs_report(ctx: typer.Context, run_id: str) -> None:
    with _fw(ctx) as fw:
        _out(fw.run_report(run_id))


@runs_app.command("resume")
@handled
def runs_resume(ctx: typer.Context, run_id: str) -> None:
    """Resume an interrupted optimize/classify/evaluate run, reusing completed work."""
    with _fw(ctx) as fw:
        _out(fw.resume(run_id))


@runs_app.command("prune")
@handled
def runs_prune(ctx: typer.Context) -> None:
    """Delete JSONL exports and run artifacts older than logging.retention_days (audit DB is kept)."""
    with _fw(ctx) as fw:
        _out(fw.prune())


# --- monitor / doctor / demo -------------------------------------------------------------------------------------
@monitor_app.command("evaluate")
@handled
def monitor_evaluate(ctx: typer.Context, apply: bool = typer.Option(None, "--apply/--no-apply",
                                                                     help="Override monitor.auto_rollback.")) -> None:
    """Compare recent production predictions with delayed verified labels; roll back on regression."""
    with _fw(ctx) as fw:
        _out(fw.monitor_evaluate(apply))


@app.command("doctor")
@handled
def doctor(ctx: typer.Context, check_connectivity: bool = typer.Option(False, help="Send one tiny live request.")
           ) -> None:
    """Check configuration, storage, optional dependencies and credential presence (never prints secrets)."""
    with _fw(ctx) as fw:
        checks = fw.doctor(check_connectivity)
        _out(checks)
        if not checks["ok"]:
            raise typer.Exit(1)


@demo_app.command("init")
@handled
def demo_init(directory: Path = typer.Argument(Path("demo")), batch: int = typer.Option(1, min=1)) -> None:
    """Write a demo config and synthetic JSON records (batch 2+ provides fresh data for later cycles)."""
    from cadmus.demo import write_demo
    _out(write_demo(directory, batch))


def main() -> None:
    app()


if __name__ == "__main__":
    main()
