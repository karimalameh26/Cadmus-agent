"""Immutable prompt bundles and compatibility-scoped champions with compare-and-swap promotion and rollback."""

from __future__ import annotations

import difflib
import json
from typing import Any

from ag_agent.contracts import PromptBundle, utc_now
from ag_agent.errors import InputError, NotFound
from ag_agent.storage import Store


class Registry:
    def __init__(self, store: Store, client_id: str) -> None:
        self.store = store
        self.client_id = client_id

    def save(self, bundle: PromptBundle) -> None:
        with self.store.tx() as conn:
            conn.execute("INSERT OR IGNORE INTO bundles (bundle_id, client_id, compat_key, payload, created_at) "
                         "VALUES (?,?,?,?,?)", (bundle.bundle_id, self.client_id, bundle.compatibility_key,
                                                bundle.model_dump_json(), bundle.created_at))

    def get(self, bundle_id: str) -> PromptBundle:
        row = self.store.one("SELECT payload FROM bundles WHERE bundle_id=? AND client_id=?",
                             (bundle_id, self.client_id))
        if row is None:
            raise NotFound("Prompt bundle not found for this client", "bundle_not_found")
        return PromptBundle.model_validate_json(row["payload"])

    def list_bundles(self, compat_key: str | None = None) -> list[dict[str, Any]]:
        sql = "SELECT payload FROM bundles WHERE client_id=?"
        params: tuple[Any, ...] = (self.client_id,)
        if compat_key:
            sql += " AND compat_key=?"
            params = (self.client_id, compat_key)
        champions = {r["compat_key"]: r["bundle_id"] for r in
                     self.store.all("SELECT compat_key, bundle_id FROM champions WHERE client_id=?", (self.client_id,))}
        out = []
        for r in self.store.all(sql + " ORDER BY created_at", params):
            b = PromptBundle.model_validate_json(r["payload"])
            out.append({"bundle_id": b.bundle_id, "compat_key": b.compatibility_key, "origin": b.origin,
                        "parent_id": b.parent_id, "examples": len(b.examples), "guidance_chars": len(b.guidance),
                        "created_at": b.created_at, "champion": champions.get(b.compatibility_key) == b.bundle_id})
        return out

    def champion_id(self, compat_key: str) -> str | None:
        row = self.store.one("SELECT bundle_id FROM champions WHERE client_id=? AND compat_key=?",
                             (self.client_id, compat_key))
        return row["bundle_id"] if row else None

    def champion(self, compat_key: str) -> PromptBundle | None:
        cid = self.champion_id(compat_key)
        return self.get(cid) if cid else None

    def promote(self, compat_key: str, expected: str | None, new_bundle: PromptBundle, *, run_id: str, action: str,
                reason: str, reference_score: float | None) -> bool:
        """Atomic compare-and-swap. Returns False (stale incumbent) if the champion changed since `expected`."""
        if new_bundle.compatibility_key != compat_key:
            raise InputError("Bundle is not compatible with this champion slot", "bundle_incompatible")
        self.save(new_bundle)
        now = utc_now()
        with self.store.tx() as conn:
            row = conn.execute("SELECT bundle_id, version FROM champions WHERE client_id=? AND compat_key=?",
                               (self.client_id, compat_key)).fetchone()
            current = row["bundle_id"] if row else None
            if current != expected:
                return False
            if row is None:
                conn.execute("INSERT INTO champions (client_id, compat_key, bundle_id, version, updated_at) "
                             "VALUES (?,?,?,?,?)", (self.client_id, compat_key, new_bundle.bundle_id, 1, now))
            else:
                conn.execute("UPDATE champions SET bundle_id=?, version=?, updated_at=? WHERE client_id=? AND "
                             "compat_key=?",
                             (new_bundle.bundle_id, row["version"] + 1, now, self.client_id, compat_key))
            conn.execute("INSERT INTO champion_history (client_id, compat_key, from_bundle, to_bundle, action, run_id, "
                         "reason, reference_score, created_at) VALUES (?,?,?,?,?,?,?,?,?)",
                         (self.client_id, compat_key, current, new_bundle.bundle_id, action, run_id, reason,
                          reference_score, now))
        return True

    def history(self, compat_key: str) -> list[dict[str, Any]]:
        return [dict(r) for r in self.store.all("SELECT * FROM champion_history WHERE client_id=? AND compat_key=? "
                                                "ORDER BY id", (self.client_id, compat_key))]

    def last_change(self, compat_key: str) -> dict[str, Any] | None:
        hist = self.history(compat_key)
        return hist[-1] if hist else None

    def reference_score(self, compat_key: str, bundle_id: str) -> float | None:
        for h in reversed(self.history(compat_key)):
            if h["to_bundle"] == bundle_id and h["reference_score"] is not None:
                return float(h["reference_score"])
        return None

    def rollback_target(self, compat_key: str, target: str | None) -> PromptBundle:
        current = self.champion_id(compat_key)
        if current is None:
            raise NotFound("No champion to roll back", "no_champion")
        if target is None:
            prior = [h["from_bundle"] for h in reversed(self.history(compat_key))
                     if h["to_bundle"] == current and h["from_bundle"]]
            if not prior:
                raise NotFound("No prior compatible bundle in history", "no_rollback_target")
            target = prior[0]
        bundle = self.get(target)
        if bundle.compatibility_key != compat_key:
            raise InputError("Rollback target is not compatible", "bundle_incompatible")
        if bundle.bundle_id == current:
            raise InputError("Target is already the champion", "already_champion")
        return bundle


def diff_bundles(a: PromptBundle | None, b: PromptBundle) -> str:
    a_lines = (a.guidance if a else "").splitlines()
    out = list(difflib.unified_diff(a_lines, b.guidance.splitlines(), fromfile=a.bundle_id if a else "(none)",
                                    tofile=b.bundle_id, lineterm=""))
    a_ex = [e.document_id for e in a.examples] if a else []
    b_ex = [e.document_id for e in b.examples]
    if a_ex != b_ex:
        out.append(f"examples: {json.dumps(a_ex)} -> {json.dumps(b_ex)}")
    return "\n".join(out)
