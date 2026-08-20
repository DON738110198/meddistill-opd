from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import tarfile
import zipfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

RUN_RE = re.compile(r"^trio://(run_[^/]+)/(?:sampler_weights|weights)/")
PROJECT_CHECKPOINT_PREFIXES = (
    "medical-opd-",
    "opd-ceval-async-qwen35-4b-base-anchor-",
    "sft-medical-qwen35-4b-medical-o1-",
    "t27-medical-sft-",
    "medical-",
    "staged-",
    "base-anchor-",
)


def _utc_now() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _rest_client() -> Any:
    import pytrio as trio

    return trio.ServiceClient().create_rest_client()


def inventory(output: Path) -> None:
    rest = _rest_client()
    rows: list[dict[str, Any]] = []
    offset = 0
    while True:
        page = rest.list_user_checkpoints(limit=100, offset=offset).result()
        rows.extend(checkpoint.model_dump(mode="json") for checkpoint in page.checkpoints)
        offset += len(page.checkpoints)
        if offset >= page.cursor.total_count or not page.checkpoints:
            break
    rows.sort(key=lambda row: (row["time"], row["checkpoint_id"]), reverse=True)
    _write_json(
        output,
        {
            "generated_at": _utc_now(),
            "count": len(rows),
            "checkpoints": rows,
        },
    )
    print(f"inventory: {len(rows)} checkpoints -> {output}")


def _index_inventory(path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    payload = _read_json(path)
    by_path = {row["path"]: row for row in payload["checkpoints"]}
    by_id = {row["checkpoint_id"]: row for row in payload["checkpoints"]}
    return by_path, by_id


def _safe_name(label: str, checkpoint: dict[str, Any]) -> str:
    kind = checkpoint["checkpoint_type"]
    checkpoint_id = checkpoint["checkpoint_id"]
    safe_label = re.sub(r"[^A-Za-z0-9_.-]+", "-", label).strip("-")
    return f"{safe_label}__{kind}__{checkpoint_id}.tar"


def _inspect_archive(path: Path) -> tuple[str, list[str]]:
    if zipfile.is_zipfile(path):
        with zipfile.ZipFile(path) as archive:
            bad_member = archive.testzip()
            members = archive.namelist()
        if bad_member is not None:
            raise RuntimeError(f"corrupt zip member {bad_member!r} in {path}")
        return "zip", members
    if tarfile.is_tarfile(path):
        with tarfile.open(path, mode="r:*") as archive:
            members = [member.name for member in archive.getmembers()]
        return "tar", members
    raise RuntimeError(f"unsupported checkpoint archive format: {path}")


def download(inventory_path: Path, selection_path: Path, destination: Path) -> None:
    by_path, _ = _index_inventory(inventory_path)
    selection = _read_json(selection_path)
    requested = selection["keep_and_download"]
    missing = [item["uri"] for item in requested if item["uri"] not in by_path]
    if missing:
        raise RuntimeError(f"selected checkpoints missing from inventory: {missing}")

    destination.mkdir(parents=True, exist_ok=True)
    rest = _rest_client()
    verified: list[dict[str, Any]] = []
    for index, item in enumerate(requested, start=1):
        checkpoint = by_path[item["uri"]]
        target = destination / _safe_name(item["label"], checkpoint)
        print(
            f"[{index}/{len(requested)}] {item['label']} "
            f"{checkpoint['checkpoint_id']} -> {target.name}",
            flush=True,
        )
        expected_size = int(checkpoint["size_bytes"])
        if not target.exists() or target.stat().st_size != expected_size:
            rest.download_checkpoint(
                checkpoint["checkpoint_id"], target, resume=True
            ).result()
        actual_size = target.stat().st_size
        if actual_size != expected_size:
            raise RuntimeError(
                f"size mismatch for {target}: expected {expected_size}, got {actual_size}"
            )
        archive_format, members = _inspect_archive(target)
        if not members:
            raise RuntimeError(f"empty checkpoint archive: {target}")
        verified.append(
            {
                **item,
                **checkpoint,
                "local_file": str(target),
                "local_size_bytes": actual_size,
                "sha256": _sha256(target),
                "archive_format": archive_format,
                "archive_member_count": len(members),
                "archive_members": members,
                "verified_at": _utc_now(),
            }
        )
        _write_json(destination.parent / "download_manifest.partial.json", verified)

    _write_json(
        destination.parent / "download_manifest.json",
        {
            "generated_at": _utc_now(),
            "selection_file": str(selection_path),
            "inventory_file": str(inventory_path),
            "count": len(verified),
            "total_size_bytes": sum(row["local_size_bytes"] for row in verified),
            "checkpoints": verified,
        },
    )
    print(f"downloaded and verified: {len(verified)} checkpoints")


def plan_delete(inventory_path: Path, selection_path: Path, output: Path) -> None:
    payload = _read_json(inventory_path)
    selection = _read_json(selection_path)
    keep_uris = {item["uri"] for item in selection["keep_and_download"]}
    inventory_uris = {row["path"] for row in payload["checkpoints"]}
    missing = sorted(keep_uris - inventory_uris)
    if missing:
        raise RuntimeError(f"selected checkpoints missing from inventory: {missing}")

    project_rows = []
    for row in payload["checkpoints"]:
        leaf = row["path"].rsplit("/", 1)[-1]
        if leaf.startswith(PROJECT_CHECKPOINT_PREFIXES):
            project_rows.append(row)
    delete_rows = [row for row in project_rows if row["path"] not in keep_uris]
    keep_rows = [row for row in project_rows if row["path"] in keep_uris]
    if len(keep_rows) != len(keep_uris):
        raise RuntimeError(
            f"keep count mismatch: project inventory has {len(keep_rows)}, "
            f"selection has {len(keep_uris)}"
        )

    deletion = [
        {
            "checkpoint_id": row["checkpoint_id"],
            "path": row["path"],
            "checkpoint_type": row["checkpoint_type"],
            "size_bytes": row["size_bytes"],
            "reason": "project_checkpoint_not_in_frozen_keep_selection",
        }
        for row in delete_rows
    ]
    _write_json(
        output,
        {
            "generated_at": _utc_now(),
            "scope": (
                "Only checkpoints whose leaf name begins with an explicit "
                "medical-project prefix"
            ),
            "project_prefixes": list(PROJECT_CHECKPOINT_PREFIXES),
            "inventory_file": str(inventory_path),
            "selection_file": str(selection_path),
            "project_checkpoint_count": len(project_rows),
            "keep_remote_count": len(keep_rows),
            "delete_count": len(deletion),
            "delete_size_bytes": sum(int(row["size_bytes"]) for row in deletion),
            "excluded_nonproject_count": len(payload["checkpoints"]) - len(project_rows),
            "delete": deletion,
        },
    )
    print(
        f"delete plan: project={len(project_rows)}, keep={len(keep_rows)}, "
        f"delete={len(deletion)}, excluded={len(payload['checkpoints']) - len(project_rows)}"
    )


def _verify_download_manifest(path: Path) -> None:
    manifest = _read_json(path)
    checkpoints = manifest["checkpoints"]
    if manifest["count"] != len(checkpoints) or not checkpoints:
        raise RuntimeError("download manifest count is empty or inconsistent")
    for row in checkpoints:
        local_file = Path(row["local_file"])
        if not local_file.is_file():
            raise RuntimeError(f"archived checkpoint missing: {local_file}")
        if local_file.stat().st_size != int(row["local_size_bytes"]):
            raise RuntimeError(f"archived checkpoint size drift: {local_file}")
        if _sha256(local_file) != row["sha256"]:
            raise RuntimeError(f"archived checkpoint hash drift: {local_file}")
        archive_format, members = _inspect_archive(local_file)
        if archive_format != row["archive_format"] or len(members) != int(
            row["archive_member_count"]
        ):
            raise RuntimeError(f"archived checkpoint structure drift: {local_file}")


def normalize_manifest(path: Path) -> None:
    manifest = _read_json(path)
    changed = 0
    for row in manifest["checkpoints"]:
        source = Path(row["local_file"])
        expected_suffix = ".zip" if row["archive_format"] == "zip" else ".tar"
        target = source.with_suffix(expected_suffix)
        if source != target:
            if target.exists():
                raise RuntimeError(f"normalized archive target already exists: {target}")
            source.replace(target)
            row["local_file"] = str(target)
            changed += 1
    _write_json(path, manifest)
    print(f"normalized archive extensions: {changed}")


def delete(
    inventory_path: Path,
    deletion_path: Path,
    download_manifest_path: Path,
    receipt_path: Path,
    execute: bool,
) -> None:
    _, by_id = _index_inventory(inventory_path)
    deletion = _read_json(deletion_path)
    requested = deletion["delete"]
    missing = [item["checkpoint_id"] for item in requested if item["checkpoint_id"] not in by_id]
    if missing:
        raise RuntimeError(f"delete IDs missing from inventory: {missing}")
    if not execute:
        total = sum(int(by_id[item["checkpoint_id"]]["size_bytes"]) for item in requested)
        print(f"dry run: {len(requested)} checkpoints, {total} bytes")
        return

    _verify_download_manifest(download_manifest_path)
    rest = _rest_client()
    partial_path = receipt_path.with_suffix(".partial.json")
    receipts: list[dict[str, Any]] = _read_json(partial_path) if partial_path.exists() else []
    completed_ids = {row["checkpoint_id"] for row in receipts}
    requested_ids = {row["checkpoint_id"] for row in requested}
    if not completed_ids.issubset(requested_ids):
        raise RuntimeError("partial deletion receipt contains IDs outside the frozen plan")
    pending = [item for item in requested if item["checkpoint_id"] not in completed_ids]
    for index, item in enumerate(pending, start=len(receipts) + 1):
        checkpoint = by_id[item["checkpoint_id"]]
        match = RUN_RE.match(checkpoint["path"])
        if match is None:
            raise RuntimeError(f"cannot extract run ID from {checkpoint['path']}")
        run_id = match.group(1)
        print(
            f"[{index}/{len(requested)}] delete {checkpoint['checkpoint_id']} "
            f"{checkpoint['path']}",
            flush=True,
        )
        rest.delete_checkpoint(run_id, checkpoint["checkpoint_id"]).result()
        receipts.append(
            {
                **item,
                **checkpoint,
                "training_run_id": run_id,
                "deleted_at": _utc_now(),
                "status": "deleted",
            }
        )
        _write_json(partial_path, receipts)

    _write_json(
        receipt_path,
        {
            "generated_at": _utc_now(),
            "inventory_file": str(inventory_path),
            "deletion_file": str(deletion_path),
            "count": len(receipts),
            "total_size_bytes": sum(int(row["size_bytes"]) for row in receipts),
            "checkpoints": receipts,
        },
    )
    print(f"deleted: {len(receipts)} checkpoints")


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    inventory_parser = subparsers.add_parser("inventory")
    inventory_parser.add_argument("--output", type=Path, required=True)

    download_parser = subparsers.add_parser("download")
    download_parser.add_argument("--inventory", type=Path, required=True)
    download_parser.add_argument("--selection", type=Path, required=True)
    download_parser.add_argument("--destination", type=Path, required=True)

    plan_delete_parser = subparsers.add_parser("plan-delete")
    plan_delete_parser.add_argument("--inventory", type=Path, required=True)
    plan_delete_parser.add_argument("--selection", type=Path, required=True)
    plan_delete_parser.add_argument("--output", type=Path, required=True)

    normalize_parser = subparsers.add_parser("normalize-manifest")
    normalize_parser.add_argument("--manifest", type=Path, required=True)

    delete_parser = subparsers.add_parser("delete")
    delete_parser.add_argument("--inventory", type=Path, required=True)
    delete_parser.add_argument("--deletion", type=Path, required=True)
    delete_parser.add_argument("--download-manifest", type=Path, required=True)
    delete_parser.add_argument("--receipt", type=Path, required=True)
    delete_parser.add_argument("--execute", action="store_true")

    args = parser.parse_args()
    if args.command == "inventory":
        inventory(args.output)
    elif args.command == "download":
        download(args.inventory, args.selection, args.destination)
    elif args.command == "plan-delete":
        plan_delete(args.inventory, args.selection, args.output)
    elif args.command == "normalize-manifest":
        normalize_manifest(args.manifest)
    elif args.command == "delete":
        delete(
            args.inventory,
            args.deletion,
            args.download_manifest,
            args.receipt,
            args.execute,
        )
    else:
        raise AssertionError(args.command)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise
