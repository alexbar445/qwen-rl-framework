"""Compare two captured Vast template search responses without making API calls."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import sys
from typing import Any, Mapping

_SCHEMA = "qwen-rl-instance-template-compare/v1"
_MAX_SNAPSHOT_BYTES = 2 * 1024 * 1024
_IMAGE_REF = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._/:+-]*@sha256:[0-9a-f]{64}$")
_IMAGE_REPOSITORY = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._/:+-]*$")
_DIGEST = re.compile(r"sha256:([0-9a-f]{64})$")
_DIGEST_HEX = re.compile(r"^[0-9a-f]{64}$")
_HASH_ID = re.compile(r"^[0-9a-f]{32}$")
_DOCKER_TAG = re.compile(r"^[A-Za-z0-9_][A-Za-z0-9_.-]{0,127}$")
_TEMPLATE_ID_TEXT = re.compile(r"^[0-9]{1,20}$")
_FIELD = re.compile(r"^[A-Za-z_][A-Za-z0-9_.-]{0,63}$")

# Vast SDK search-template row fields plus the stable id returned by that endpoint.
_ROW_FIELDS = frozenset("""
    id creator_id args_str command default_tag desc docker_login_repo docker_login_user
    docker_login_pass env extra_filters href image jupyter_dir jup_direct jupyter_tested
    jupyterlab_tested max_cuda min_cuda name onstart python_utf8 lang_utf8 repo runtype
    ssh_direct tag use_jupyter_lab use_ssh hash_id created_from created_from_id count_created
    desc_count recent_create_date created_at deleted_at private readme_visible readme_hash
    recommended_disk_space recommended cached autoscaler vm sort_order volume_info tags
""".split())
_ACTIVITY_FIELDS = frozenset({"count_created", "desc_count", "recent_create_date", "cached"})
_MESSAGES = {
    "snapshot_unreadable": "A snapshot could not be read as a JSON array.",
    "snapshot_incomplete": "A snapshot is not one complete Vast template row for the requested ID.",
    "template_schema_unknown": "The template row contains missing or unknown fields.",
    "template_id_invalid": "A positive numeric template ID is required.",
    "hash_id_invalid": "A snapshot is missing a valid template revision hash.",
    "image_tag_ambiguous": "The image/tag fields use an unverified digest encoding.",
    "expected_image_invalid": "The expected image must be an immutable repository@sha256 reference.",
    "image_mismatch": "The after snapshot does not contain the requested immutable image reference.",
    "settings_changed": "Non-image template settings changed between snapshots.",
    "hash_id_stale": "Image or settings changed without a corresponding template hash change.",
    "new_template_hash_stale": "A newly created template ID did not receive a new revision hash.",
    "template_matches": "The after snapshot matches the requested digest with no non-image settings drift.",
}


class _SnapshotError(Exception):
    def __init__(self, code: str):
        super().__init__(code)
        self.code = code


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON key")
        result[key] = value
    return result


def _reject_constant(_value: str) -> None:
    raise ValueError("non-finite JSON number")


def _read_rows(path: Path, template_id: int) -> dict[str, Any]:
    try:
        if path.stat().st_size > _MAX_SNAPSHOT_BYTES:
            raise _SnapshotError("snapshot_unreadable")
        raw = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=_unique_object,
                         parse_constant=_reject_constant)
    except _SnapshotError:
        raise
    except Exception:
        raise _SnapshotError("snapshot_unreadable") from None
    if not isinstance(raw, list) or len(raw) != 1 or not isinstance(raw[0], dict):
        raise _SnapshotError("snapshot_incomplete")
    row = raw[0]
    if type(row.get("id")) is not int or row["id"] != template_id:
        raise _SnapshotError("snapshot_incomplete")
    if set(row) != _ROW_FIELDS:
        raise _SnapshotError("template_schema_unknown")
    if not isinstance(row.get("hash_id"), str) or _HASH_ID.fullmatch(row["hash_id"]) is None:
        raise _SnapshotError("hash_id_invalid")
    if not isinstance(row.get("image"), str) or not isinstance(row.get("tag"), str):
        raise _SnapshotError("template_schema_unknown")
    if any(not isinstance(key, str) or _FIELD.fullmatch(key) is None for key in row):
        raise _SnapshotError("template_schema_unknown")
    return row


def _image_reference(row: Mapping[str, Any]) -> str:
    image, tag = row["image"], row["tag"]
    if (image.endswith("@sha256") and _IMAGE_REPOSITORY.fullmatch(image[:-7])
            and _DIGEST_HEX.fullmatch(tag)):
        return f"{image}:{tag}"
    if "@sha256" in image:
        if not _IMAGE_REF.fullmatch(image) or tag != "":
            raise _SnapshotError("image_tag_ambiguous")
        return image
    if tag == "":
        return image
    if not _DOCKER_TAG.fullmatch(tag):
        raise _SnapshotError("image_tag_ambiguous")
    # The Vast SDK stores the repository and Docker tag in separate fields.
    return f"{image}:{tag}"


def _changed_fields(
    before: Mapping[str, Any], after: Mapping[str, Any], *, new_template_id: bool = False,
) -> list[str]:
    ignored = _ACTIVITY_FIELDS | {"hash_id", "image", "tag"}
    if new_template_id:
        ignored |= {"id", "created_at", "created_from", "created_from_id"}
    changed = []
    for key in sorted((set(before) | set(after)) - ignored):
        before_value, after_value = before.get(key), after.get(key)
        if key == "extra_filters":
            before_value = _normalize_extra_filters(before_value)
            after_value = _normalize_extra_filters(after_value)
        if before_value != after_value:
            changed.append(key)
    if any(_FIELD.fullmatch(key) is None for key in changed):
        raise _SnapshotError("template_schema_unknown")
    return changed


def _normalize_extra_filters(value: Any) -> Any:
    """Compare Vast's serialized filter object with the equivalent JSON object."""
    if not isinstance(value, str):
        return value
    try:
        parsed = json.loads(value, object_pairs_hook=_unique_object, parse_constant=_reject_constant)
    except Exception:
        return value
    return parsed if isinstance(parsed, dict) else value


def compare_template_snapshots(
    *, template_id: int, before_path: Path, after_path: Path, expected_image: str,
    after_template_id: int | None = None,
) -> dict[str, Any]:
    """Compare captured Vast template rows; an explicit after ID permits version creation."""
    safe_template_id = template_id if type(template_id) is int and template_id > 0 else None
    safe_after_template_id = (
        safe_template_id if after_template_id is None else
        after_template_id if type(after_template_id) is int and after_template_id > 0 else None
    )
    base: dict[str, Any] = {
        "schema": _SCHEMA, "status": "incomplete", "template_id": safe_template_id,
        "after_template_id": safe_after_template_id,
        "before_hash_id": None, "after_hash_id": None, "matched_digest": None,
        "changed_fields": [], "reason_code": None,
    }
    if safe_template_id is None or safe_after_template_id is None:
        base["reason_code"] = "template_id_invalid"
        base["message"] = _MESSAGES["template_id_invalid"]
        return base
    expected_digest = _DIGEST.search(expected_image) if isinstance(expected_image, str) else None
    if not isinstance(expected_image, str) or expected_digest is None or not _IMAGE_REF.fullmatch(expected_image):
        base["reason_code"] = "expected_image_invalid"
        base["message"] = _MESSAGES["expected_image_invalid"]
        return base
    try:
        before = _read_rows(Path(before_path), template_id)
        after = _read_rows(Path(after_path), safe_after_template_id)
        base["before_hash_id"], base["after_hash_id"] = before["hash_id"], after["hash_id"]
        before_image = _image_reference(before)
        after_image = _image_reference(after)
        new_template_id = safe_after_template_id != safe_template_id
        changed = _changed_fields(before, after, new_template_id=new_template_id)
    except _SnapshotError as error:
        base["reason_code"] = error.code
        base["message"] = _MESSAGES[error.code]
        return base

    image_matches = after_image == expected_image
    image_changed = before_image != after_image
    base["changed_fields"] = changed
    if image_matches:
        base["matched_digest"] = "sha256:" + expected_digest.group(1)
    if not image_matches:
        base.update(status="blocked", reason_code="image_mismatch", message=_MESSAGES["image_mismatch"])
    elif changed:
        base.update(status="blocked", reason_code="settings_changed", message=_MESSAGES["settings_changed"])
    elif new_template_id and before["hash_id"] == after["hash_id"]:
        base.update(status="blocked", reason_code="new_template_hash_stale",
                    message=_MESSAGES["new_template_hash_stale"])
    elif image_changed and before["hash_id"] == after["hash_id"]:
        base.update(status="blocked", reason_code="hash_id_stale", message=_MESSAGES["hash_id_stale"])
    else:
        base.update(status="passed", reason_code="template_matches", message=_MESSAGES["template_matches"])
    return base


def format_human(report: Mapping[str, Any]) -> str:
    """Render only hashes, digest, stable reason codes, and field names."""
    status = report.get("status")
    status = status if status in {"passed", "blocked", "incomplete"} else "incomplete"
    lines = [f"Template snapshot check: {status.upper()}"]
    template_id = report.get("template_id")
    if type(template_id) is int and template_id > 0:
        lines.append(f"Template ID: {template_id}")
    after_template_id = report.get("after_template_id")
    if (type(after_template_id) is int and after_template_id > 0
            and after_template_id != template_id):
        lines.append(f"After template ID: {after_template_id}")
    for label, key in (("Before hash", "before_hash_id"), ("After hash", "after_hash_id")):
        value = report.get(key)
        if isinstance(value, str) and _HASH_ID.fullmatch(value):
            lines.append(f"{label}: {value}")
    digest = report.get("matched_digest")
    if isinstance(digest, str) and re.fullmatch(r"sha256:[0-9a-f]{64}", digest):
        lines.append(f"Matched digest: {digest}")
    reason = report.get("reason_code")
    if isinstance(reason, str) and reason in _MESSAGES:
        lines.append(f"Result: {_MESSAGES[reason]}")
    elif report.get("reason_code") == "template_schema_unknown":
        lines.append(str(report.get("message", _MESSAGES["template_schema_unknown"])))
    fields = report.get("changed_fields")
    if isinstance(fields, list) and fields:
        lines.append("Changed settings: " + ", ".join(
            field for field in fields if isinstance(field, str) and field in _ROW_FIELDS and _FIELD.fullmatch(field)))
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    def parse_template_id(value: str) -> int:
        if _TEMPLATE_ID_TEXT.fullmatch(value) is None:
            raise argparse.ArgumentTypeError("a positive numeric template ID is required")
        parsed = int(value)
        if parsed < 1:
            raise argparse.ArgumentTypeError("a positive numeric template ID is required")
        return parsed

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--template-id", type=parse_template_id, required=True)
    parser.add_argument("--after-template-id", type=parse_template_id,
                        help="API-returned ID when Vast creates a new template version; defaults to --template-id")
    parser.add_argument("--before", type=Path, required=True, help="captured JSON array from Vast template search")
    parser.add_argument("--after", type=Path, required=True, help="captured JSON array from Vast template search")
    parser.add_argument("--image", required=True, help="expected repository@sha256 digest reference")
    parser.add_argument("--json", action="store_true", help="write the stable machine-readable report")
    args = parser.parse_args(argv)
    report = compare_template_snapshots(template_id=args.template_id, before_path=args.before,
                                        after_path=args.after, expected_image=args.image,
                                        after_template_id=args.after_template_id)
    if args.json:
        print(json.dumps(report, sort_keys=True))
    else:
        print(format_human(report))
    return {"passed": 0, "blocked": 1, "incomplete": 2}[report["status"]]


if __name__ == "__main__":
    raise SystemExit(main())
