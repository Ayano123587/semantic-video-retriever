"""Hash-only attestation of an externally approved advanced phase2 parent."""

import hashlib
import json
import stat
import sys
import time
import traceback
from pathlib import Path

EXPECTED_ADVANCED_PLAN_SHA256 = None


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def authenticate(package, expected):
    package = Path(package)
    require(
        sha(package / "advanced-plan.json") == expected,
        "Externally reviewed plan required",
    )
    plan = json.loads((package / "advanced-plan.json").read_bytes())
    require(
        plan["schema"] == "advanced_checkpoint_attestation_transport_v1",
        "Hash-only transport required",
    )
    for name, entry in plan["files"].items():
        require(
            Path(name).name == name and "/" not in name and "\\" not in name,
            "Plain safe payload names required",
        )
        path = package / name
        require(
            not path.is_symlink()
            and path.stat().st_size == entry["bytes"]
            and sha(path) == entry["sha256"],
            "Frozen payload differs",
        )
    return plan


def validate_contract(contract, header, proof):
    require(
        contract["schema"] == "advanced_phase2_hash_contract_v1"
        and contract["action"] == "hash_only"
        and contract["max_seconds"] == 1800
        and contract["training_authorized"] is False
        and contract["native_deserialization_authorized"] is False,
        "Actual separate hash-only contract required",
    )
    parent = contract["parent_execution"]
    require(
        parent["canonical"] == proof["canonical"]
        and parent["id"] == proof["id"]
        and parent["version"] == proof["version"] == 1
        and parent["source_sha256"] == proof["source_sha256"]
        and proof["source_exact_before_status"] is True,
        "Exact actual parent execution proof required",
    )
    require(
        header == contract["header"]
        and header["completed_seeds"] == 10
        and header["stage"] == "fit"
        and header["batches"] == 34122,
        "Actual advanced parent header required",
    )
    files = contract["files"]
    names = {
        contract["snapshot_name"],
        contract["header_name"],
        *contract["completed_names"],
    }
    require(
        set(files) == names
        and len(names) == 12
        and len(contract["completed_names"]) == 10,
        "Exact snapshot/header/ten-seed scope required",
    )
    require(
        files[contract["snapshot_name"]]["known_sha256"] == header["snapshot_sha256"]
        and files[contract["header_name"]]["known_sha256"] == contract["header_sha256"],
        "Known native snapshot/header anchors required",
    )
    require(
        sum(row["known_sha256"] is None for row in files.values()) == 4,
        "Exactly four new native hashes remain to be measured",
    )
    for name in files:
        require(
            Path(name).name == name
            and "/" not in name
            and "\\" not in name
            and name not in (".", ".."),
            "Safe native basename required",
        )
    return contract


def hash_inventory(parent, contract):
    require(
        sys.platform == "linux"
        and parent.is_absolute()
        and parent.resolve().is_relative_to(Path("/kaggle/input")),
        "Linux mounted parent required",
    )
    for path in (parent, *parent.parents):
        require(not stat.S_ISLNK(path.lstat().st_mode), "Symlink mount forbidden")
    result = {}
    for name, expected in contract["files"].items():
        path = parent / name
        require(
            path.is_file() and not path.is_symlink(), "Every exact parent file required"
        )
        pin = sha(path)
        require(
            expected["known_sha256"] is None or expected["known_sha256"] == pin,
            "Previously authenticated native bytes differ",
        )
        result[name] = {"sha256": pin, "bytes": path.stat().st_size}
    return result


def run(input_root, output):
    started = time.monotonic()
    output = Path(output)
    output.mkdir(exist_ok=False)
    status = {
        "stage": "bootstrap",
        "native_deserialized": False,
        "training": False,
        "scoring": False,
        "scientific_complete": False,
        "resume_authorized": False,
    }

    def record():
        (output / "attestation-status.json").write_text(
            json.dumps(status, indent=2) + "\n"
        )

    record()
    try:
        matches = [
            path
            for path in Path(input_root).rglob("advanced-plan.json")
            if sha(path) == EXPECTED_ADVANCED_PLAN_SHA256
        ]
        require(len(matches) == 1, "One actual reviewed source package required")
        package = matches[0].parent
        plan = authenticate(package, EXPECTED_ADVANCED_PLAN_SHA256)
        template = (package / "advanced-runner.py").read_bytes()
        token = b"EXPECTED_ADVANCED_PLAN_SHA256" + b" = None"
        require(
            template.count(token) == 1
            and Path(__file__).read_bytes().replace(b"\r\n", b"\n")
            == template.replace(
                token,
                (
                    "EXPECTED_ADVANCED_PLAN_SHA256 = "
                    + repr(EXPECTED_ADVANCED_PLAN_SHA256)
                ).encode(),
            ),
            "Executed hash-only runner differs",
        )
        contract = json.loads((package / "advanced-contract.json").read_bytes())
        require(
            sha(package / "advanced-contract.json") == plan["contract_sha256"],
            "External contract differs",
        )
        header = json.loads((package / "parent-header.json").read_bytes())
        proof = json.loads((package / "parent-source-proof.json").read_bytes())
        validate_contract(contract, header, proof)
        matches = [
            path
            for path in Path(input_root).rglob(contract["header_name"])
            if sha(path) == contract["header_sha256"]
        ]
        require(len(matches) == 1, "Unique actual advanced parent mount required")
        parent = matches[0].parent
        require(
            json.loads(matches[0].read_bytes()) == header,
            "Mounted actual header differs",
        )
        before = hash_inventory(parent, contract)
        authenticate(package, EXPECTED_ADVANCED_PLAN_SHA256)
        require(
            before == hash_inventory(parent, contract),
            "Parent bytes changed during attestation",
        )
        require(time.monotonic() - started < 1800, "Hash-only service window exhausted")
        inventory = {
            "schema": "advanced_phase2_hash_inventory_v1",
            "parent_execution": contract["parent_execution"],
            "contract_sha256": plan["contract_sha256"],
            "header": header,
            "files": before,
            "resolved_mount": str(parent),
            "native_deserialized": False,
            "training": False,
            "scoring": False,
            "resume_authorized": False,
            "scientific_complete": False,
        }
        (output / "advanced-inventory.json").write_text(
            json.dumps(inventory, indent=2) + "\n"
        )
        status.update(
            stage="complete",
            hash_complete=True,
            inventory_sha256=sha(output / "advanced-inventory.json"),
            elapsed_seconds=time.monotonic() - started,
        )
        record()
        return status
    except Exception as error:
        status.update(
            error_type=type(error).__name__,
            message=str(error)[:2000],
            traceback=traceback.format_exc()[-12000:],
            elapsed_seconds=time.monotonic() - started,
        )
        record()
        raise


if __name__ == "__main__":
    run("/kaggle/input", "/kaggle/working/advanced-parent-attestation")
