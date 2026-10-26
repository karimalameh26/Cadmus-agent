# Security and data handling

- **Deployment model:** local, single-host SDK/CLI. Client scoping isolates data within one installation; it is not
  a hardened multi-tenant service. Protect `data_dir` with OS access controls.
- **Secrets:** credentials are read only from the environment variable named in `provider.api_key_env`. Never put
  keys in TOML, logs or committed files.
- **What is stored:** `data_dir/state.sqlite3` (labels, events, predictions, decisions), `artifacts/` (normalized
  evidence and images; may contain customer content), `runs/` (protected results: predictions, candidate diffs,
  reports), `logs/` (JSONL events with identifiers, counts and reason codes only).
- **Logs:** events are field-allowlisted. OCR/document text, image bytes, prompts, provider bodies, file paths and
  secrets are never logged. Source names appear only in explicitly requested outputs (`classify`, `inputs preview`).
- **Untrusted content:** document content is treated as data. Structural tags are neutralized and the classifier has
  no tools. This reduces, but does not eliminate, prompt-injection risk.
- **Reviewer authorization:** a configured identity list under OS access controls, not an adversarial security
  boundary. A reviewer cannot verify their own correction.
- **Retention:** set `logging.retention_days` and run `runs prune` to delete JSONL exports and run artifacts.
  Database audit records and decision ancestry are kept; once evidence artifacts are deleted, exact replay of affected
  runs is no longer possible.
- **Live providers:** use `https` endpoints (plain `http` is allowed only for localhost). Confirm the provider's data
  retention terms before sending client documents.
