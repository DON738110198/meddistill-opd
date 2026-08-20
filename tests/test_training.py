from __future__ import annotations

import copy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

import medical_opd.config as config_module
import medical_opd.training as training_module
from medical_opd.backend import UsageLedger
from medical_opd.config import ExperimentConfig, load_config
from medical_opd.io_utils import atomic_write_json, sha256_file, stable_hash
from medical_opd.training import (
    TrainRequest,
    _fit_prompt,
    _local_state,
    _sequence_cache_path,
    run_training,
    training_contract,
    validate_resume_migration,
    validate_resume_state,
)


def test_fit_prompt_keeps_system_prefix_and_generation_suffix() -> None:
    prompt = list(range(200))

    fitted, removed = _fit_prompt(prompt, max_sequence=100, reserved_completion=20)

    assert len(fitted) == 80
    assert removed == 120
    assert fitted[:20] == prompt[:20]
    assert fitted[20:] == prompt[-60:]
    assert fitted[-1] == prompt[-1]


def test_fit_prompt_does_not_modify_prompt_within_p90_sequence_budget() -> None:
    prompt = list(range(80))

    fitted, removed = _fit_prompt(prompt, max_sequence=100, reserved_completion=20)

    assert fitted is prompt
    assert fitted == list(range(80))
    assert removed == 0


@pytest.mark.parametrize(
    ("max_sequence", "reserved_completion"),
    [(16, 16), (1, 0), (8, 7)],
)
def test_fit_prompt_rejects_unusable_prompt_budget(
    max_sequence: int, reserved_completion: int
) -> None:
    with pytest.raises(ValueError, match="no usable prompt budget"):
        _fit_prompt([1, 2, 3], max_sequence=max_sequence, reserved_completion=reserved_completion)


def test_fit_prompt_dynamic_completion_room_never_exceeds_sequence_cap() -> None:
    prompt = list(range(2000))
    fitted, _ = _fit_prompt(prompt, max_sequence=1024, reserved_completion=16)
    completion_cap = min(900, 1024 - len(fitted))

    assert len(fitted) + completion_cap == 1024
    assert completion_cap == 16


def test_sequence_kd_cache_key_includes_dynamic_completion_cap(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _workspace(tmp_path, monkeypatch)
    row = {"id": "medical-1"}

    cap_16 = _sequence_cache_path(config, row, 16)
    cap_32 = _sequence_cache_path(config, row, 32)

    assert cap_16 == _sequence_cache_path(config, row, 16)
    assert cap_16 != cap_32
    assert cap_16.parent == tmp_path / "cache" / "sequence_kd"


def test_m4_one_step_uses_remaining_sequence_room_as_teacher_generation_cap(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _workspace(tmp_path, monkeypatch)
    processed = tmp_path / "data" / "processed"
    medical_rows = [
        {"id": f"medical-{index}", "question": f"question {index}", "completion": "answer"}
        for index in range(4)
    ]
    (processed / "train_medical.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in medical_rows),
        encoding="utf-8",
    )
    (processed / "train_general_replay.jsonl").write_text("", encoding="utf-8")
    atomic_write_json(
        processed / "lengths.json",
        {
            "derived_max_sequence_tokens": 100,
            "derived_max_completion_tokens": 90,
        },
    )

    class _Tokenizer:
        eos_token = None

        def apply_chat_template(self, messages: object, **kwargs: object) -> str:
            return "rendered prompt"

        def encode(self, text: str, **kwargs: object) -> list[int]:
            return list(range(200))

    class _Immediate:
        def __init__(self, value: object) -> None:
            self.value = value

        def __await__(self):
            async def done():
                return self.value

            return done().__await__()

    class _Training:
        def get_tokenizer(self) -> _Tokenizer:
            return _Tokenizer()

        async def forward_backward_async(self, datums: object, loss_fn: str) -> _Immediate:
            return _Immediate(SimpleNamespace(metrics={"loss": 1.0, "token_count": 4}))

        async def optim_step_async(self, adam: object) -> _Immediate:
            return _Immediate(None)

        async def save_state_async(self, name: str) -> _Immediate:
            return _Immediate(SimpleNamespace(path="trio://state-1"))

        async def save_weights_for_sampler_async(self, name: str) -> _Immediate:
            return _Immediate(SimpleNamespace(path="trio://weights-1"))

    training = _Training()

    class _Service:
        async def create_lora_training_client_async(self, **kwargs: object) -> _Training:
            return training

        async def create_sampling_client_async(self, **kwargs: object) -> object:
            return object()

    class _ModelInput:
        @staticmethod
        def from_ints(values: list[int]) -> tuple[int, ...]:
            return tuple(values)

    class _Datum:
        def __init__(self, **kwargs: object) -> None:
            self.kwargs = kwargs

    fake_trio = SimpleNamespace(
        ServiceClient=_Service,
        ModelInput=_ModelInput,
        Datum=_Datum,
        AdamParams=lambda **kwargs: kwargs,
    )
    monkeypatch.setattr(training_module, "_load_trio", lambda: fake_trio)
    observed_caps: list[int] = []

    async def fake_m4_completion(
        trio: object,
        teacher: object,
        tokenizer: object,
        config: ExperimentConfig,
        row: dict[str, object],
        prompt: list[int],
        *,
        max_completion: int,
        seed: int,
    ) -> tuple[list[int], UsageLedger, bool]:
        assert len(prompt) == 84
        observed_caps.append(max_completion)
        return [999], UsageLedger(), False

    monkeypatch.setattr(training_module, "_m4_completion", fake_m4_completion)
    monkeypatch.setattr(
        training_module,
        "fetch_json",
        lambda url: {
            "version": "test",
            "items": [
                {
                    "display_name": config.get("models", role),
                    "prices": {
                        mode: {"unit_price": 100}
                        for mode in ("prefill", "sample", "train")
                    },
                }
                for role in ("student", "teacher")
            ],
        },
    )

    summary = training_module.asyncio.run(
        training_module._train_async(
            config,
            TrainRequest("M4", 1, tmp_path / "m4-run", confirm_paid=True),
        )
    )

    assert observed_caps == [16, 16, 16, 16]
    assert summary["status"] == "completed"
    step = json.loads((tmp_path / "m4-run" / "steps.jsonl").read_text(encoding="utf-8"))
    assert step["prompt_truncated_tokens"] == 4 * (200 - 84)


def _workspace(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> ExperimentConfig:
    loaded = load_config()
    config_path = tmp_path / "experiment.toml"
    config_path.write_bytes(loaded.path.read_bytes())
    config = ExperimentConfig(config_path, copy.deepcopy(loaded.raw))
    monkeypatch.setattr(config_module, "PROJECT_ROOT", tmp_path)
    processed = tmp_path / "data" / "processed"
    processed.mkdir(parents=True)
    (processed / "manifest.json").write_text('{"status":"frozen"}\n', encoding="utf-8")
    (processed / "train_medical.jsonl").write_text('{"id":"medical-0"}\n', encoding="utf-8")
    (processed / "train_general_replay.jsonl").write_text(
        '{"id":"general-0"}\n', encoding="utf-8"
    )
    atomic_write_json(
        processed / "lengths.json",
        {
            "derived_max_sequence_tokens": 672,
            "derived_max_completion_tokens": 576,
        },
    )
    generated = tmp_path / "reports" / "generated"
    generated.mkdir(parents=True)
    atomic_write_json(
        generated / "preflight_latest.json",
        {
            "status": "ready",
            "config_sha256": sha256_file(config.path),
            "configured_models": {
                "student": config.get("models", "student"),
                "teacher": config.get("models", "teacher"),
            },
        },
    )
    return config


def _request(tmp_path: Path, **overrides: object) -> TrainRequest:
    values: dict[str, object] = {
        "method": "M5",
        "target_steps": 10,
        "output_dir": tmp_path / "run",
        "confirm_paid": True,
        "resume_state": "pytrio://checkpoint/state",
    }
    values.update(overrides)
    return TrainRequest(**values)  # type: ignore[arg-type]


def _approved_migration(
    config: ExperimentConfig,
    source_path: Path,
    prior: dict[str, object],
    migration_path: Path,
) -> dict[str, object]:
    checkpoint = next(
        value
        for value in prior["checkpoints"]  # type: ignore[union-attr]
        if value["step"] == prior["completed_steps"]
    )
    contract = training_contract(config, str(prior["method"]))
    record: dict[str, object] = {
        "schema_version": 1,
        "status": "approved",
        "source": {
            "local_state": str(source_path.relative_to(config.root)),
            "sha256": sha256_file(source_path),
            "method": prior["method"],
            "completed_steps": prior["completed_steps"],
            "optimizer_state": checkpoint["state"],
            "sampler_weights": checkpoint["sampler_weights"],
            "artifact_hashes": prior["hashes"],
            "code_hashes": prior["code_hashes"],
        },
        "model_identity": {
            "student": config.get("models", "student"),
            "teacher": config.get("models", "teacher"),
            "verification_status": "verified",
        },
        "approved_training_contract": contract,
        "approved_training_contract_sha256": stable_hash(contract),
        "allowed_differences": ["evaluation-only config and paid-safety guards"],
    }
    atomic_write_json(migration_path, record)
    return record


def test_paid_guard_fires_before_manifest_or_network(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    loaded = load_config()
    config = ExperimentConfig(tmp_path / "missing-config.toml", copy.deepcopy(loaded.raw))
    monkeypatch.setattr(config_module, "PROJECT_ROOT", tmp_path / "missing-root")
    touched_network = False

    def unexpected_train(*args: object, **kwargs: object) -> None:
        nonlocal touched_network
        touched_network = True
        raise AssertionError("paid runner reached async backend")

    monkeypatch.setattr(training_module, "_train_async", unexpected_train)
    request = TrainRequest("M5", 1, tmp_path / "run", confirm_paid=False)

    with pytest.raises(ValueError, match="confirm-paid"):
        run_training(config, request)
    assert touched_network is False


@pytest.mark.parametrize("method", ["M2", "M3", "M4", "M5"])
def test_project_pause_blocks_every_paid_training_method_before_async_runner(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    method: str,
) -> None:
    config = _workspace(tmp_path, monkeypatch)
    atomic_write_json(
        tmp_path / "reports" / "generated" / "PAUSE_PAID_TRAINING.json",
        {"status": "paused"},
    )
    paid_runner_called = False

    async def unexpected_train(*args: object, **kwargs: object) -> dict[str, str]:
        nonlocal paid_runner_called
        paid_runner_called = True
        raise AssertionError("paused project reached paid training")

    monkeypatch.setattr(training_module, "_train_async", unexpected_train)

    with pytest.raises(RuntimeError, match="paid training is paused"):
        run_training(config, _request(tmp_path, method=method, resume_state=None))
    assert paid_runner_called is False


@pytest.mark.parametrize("status", ["blocked", "environment_ready_data_pending"])
def test_blocked_preflight_never_enters_paid_training_runner(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    status: str,
) -> None:
    config = _workspace(tmp_path, monkeypatch)
    atomic_write_json(
        tmp_path / "reports" / "generated" / "preflight_latest.json",
        {
            "status": status,
            "config_sha256": sha256_file(config.path),
            "configured_models": {
                "student": config.get("models", "student"),
                "teacher": config.get("models", "teacher"),
            },
        },
    )
    paid_runner_called = False

    async def unexpected_train(*args: object, **kwargs: object) -> dict[str, str]:
        nonlocal paid_runner_called
        paid_runner_called = True
        raise AssertionError("blocked preflight reached paid training")

    monkeypatch.setattr(training_module, "_train_async", unexpected_train)

    with pytest.raises(RuntimeError, match="preflight is blocked or stale"):
        run_training(config, _request(tmp_path, method="M2", resume_state=None))
    assert paid_runner_called is False


@pytest.mark.parametrize("method", ["M4", "M5"])
def test_teacher_methods_are_blocked_when_superiority_gate_is_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, method: str
) -> None:
    config = _workspace(tmp_path, monkeypatch)
    monkeypatch.setattr(
        training_module,
        "_train_async",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("remote work started")),
    )

    with pytest.raises(RuntimeError, match="superiority gate is missing"):
        run_training(config, _request(tmp_path, method=method, resume_state=None))


@pytest.mark.parametrize("status", ["failed", "blocked", "pending", None])
def test_teacher_methods_are_blocked_unless_superiority_gate_passed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    status: str | None,
) -> None:
    config = _workspace(tmp_path, monkeypatch)
    atomic_write_json(
        tmp_path / "reports" / "generated" / "teacher_gate.json",
        {"status": status},
    )
    monkeypatch.setattr(
        training_module,
        "_train_async",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("remote work started")),
    )

    with pytest.raises(RuntimeError, match="superiority gate failed"):
        run_training(config, _request(tmp_path, method="M5", resume_state=None))


def test_sft_does_not_require_teacher_superiority_gate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _workspace(tmp_path, monkeypatch)

    async def fake_train(*args: object, **kwargs: object) -> dict[str, str]:
        return {"status": "local-test"}

    monkeypatch.setattr(training_module, "_train_async", fake_train)

    result = run_training(config, _request(tmp_path, method="M2", resume_state=None))
    assert result == {"status": "local-test"}


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"method": "M0"}, "method"),
        ({"target_steps": 0}, "positive"),
        ({"target_steps": -1}, "positive"),
        ({"run_name": "???"}, "slug is empty"),
    ],
)
def test_train_request_rejects_invalid_paid_run_inputs(
    tmp_path: Path, kwargs: dict[str, object], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        _request(tmp_path, **kwargs)


def test_local_state_captures_resume_cursor_hashes_usage_and_checkpoints(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _workspace(tmp_path, monkeypatch)
    request = _request(tmp_path, target_steps=50)
    usage = UsageLedger(student_train_tokens=123, optimizer_steps=7, wall_seconds=3.5)
    checkpoints = [{"step": 7, "state": "pytrio://state-7"}]

    state = _local_state(
        request,
        config,
        status="running",
        completed_steps=7,
        usage=usage,
        checkpoints=checkpoints,
    )

    assert state["method"] == "M5"
    assert state["target_steps"] == 50
    assert state["completed_steps"] == 7
    assert state["next_step"] == 8
    assert state["sample_cursor"] == 28
    assert state["seed"] == config.get("experiment", "seed")
    assert state["usage"]["student_train_tokens"] == 123
    assert state["usage"]["optimizer_steps"] == 7
    assert state["checkpoints"] == checkpoints
    assert set(state["hashes"]) == {"config", "manifest", "medical_train", "general_train"}
    assert state["resume_source"] == request.resume_state


def test_valid_resume_accepts_same_method_artifacts_seed_and_cursor(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _workspace(tmp_path, monkeypatch)
    prior_request = _request(tmp_path, target_steps=10)
    prior = _local_state(
        prior_request,
        config,
        status="failed",
        completed_steps=7,
        usage=UsageLedger(optimizer_steps=7),
        checkpoints=[{"step": 7, "state": "pytrio://state-7"}],
    )

    validate_resume_state(prior, _request(tmp_path, target_steps=10), config)
    validate_resume_state(prior, _request(tmp_path, target_steps=50), config)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("method", "M2", "method"),
        ("seed", -1, "seed"),
        ("sample_cursor", 999, "cursor"),
        ("completed_steps", -1, "target"),
    ],
)
def test_resume_rejects_incompatible_local_state(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    field: str,
    value: object,
    message: str,
) -> None:
    config = _workspace(tmp_path, monkeypatch)
    request = _request(tmp_path, target_steps=10)
    prior = _local_state(
        request,
        config,
        status="failed",
        completed_steps=7,
        usage=UsageLedger(optimizer_steps=7),
        checkpoints=[],
    )
    prior[field] = value

    with pytest.raises(RuntimeError, match=message):
        validate_resume_state(prior, request, config)


def test_resume_rejects_target_below_completed_step(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _workspace(tmp_path, monkeypatch)
    prior = _local_state(
        _request(tmp_path, target_steps=10),
        config,
        status="failed",
        completed_steps=7,
        usage=UsageLedger(optimizer_steps=7),
        checkpoints=[],
    )

    with pytest.raises(RuntimeError, match="target"):
        validate_resume_state(prior, _request(tmp_path, target_steps=6), config)


@pytest.mark.parametrize(
    ("path", "payload"),
    [
        ("experiment.toml", b"\n# drift\n"),
        ("data/processed/manifest.json", b'{"status":"changed"}\n'),
        ("data/processed/train_medical.jsonl", b'{"id":"changed"}\n'),
        ("data/processed/train_general_replay.jsonl", b'{"id":"changed"}\n'),
    ],
)
def test_resume_rejects_config_or_frozen_data_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    path: str,
    payload: bytes,
) -> None:
    config = _workspace(tmp_path, monkeypatch)
    request = _request(tmp_path)
    prior = _local_state(
        request,
        config,
        status="failed",
        completed_steps=7,
        usage=UsageLedger(optimizer_steps=7),
        checkpoints=[],
    )
    drifted = tmp_path / path
    drifted.write_bytes(drifted.read_bytes() + payload)

    with pytest.raises(RuntimeError, match="hashes"):
        validate_resume_state(prior, request, config)


def test_run_training_rejects_existing_incomplete_run_without_remote_resume_uri(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _workspace(tmp_path, monkeypatch)
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    request = TrainRequest("M2", 10, run_dir, confirm_paid=True)
    prior = _local_state(
        request,
        config,
        status="failed",
        completed_steps=1,
        usage=UsageLedger(optimizer_steps=1),
        checkpoints=[],
    )
    atomic_write_json(run_dir / "state.json", prior)

    with pytest.raises(RuntimeError, match="requires --resume-state"):
        training_module.asyncio.run(training_module._train_async(config, request))


def test_new_directory_resume_requires_local_state_and_approval_before_pytrio(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _workspace(tmp_path, monkeypatch)
    pytrio_touched = False

    def unexpected_trio() -> object:
        nonlocal pytrio_touched
        pytrio_touched = True
        raise AssertionError("PyTRIO loaded before local resume validation")

    monkeypatch.setattr(training_module, "_load_trio", unexpected_trio)
    request = _request(tmp_path, output_dir=tmp_path / "new-run")

    with pytest.raises(RuntimeError, match="resume-local-state"):
        training_module.asyncio.run(training_module._train_async(config, request))
    assert pytrio_touched is False


def test_approved_new_directory_resume_carries_state_before_pytrio(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _workspace(tmp_path, monkeypatch)
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    source_path = source_dir / "state.json"
    checkpoint = {
        "step": 7,
        "state": "pytrio://checkpoint/state-7",
        "sampler_weights": "pytrio://checkpoint/weights-7",
    }
    prior = _local_state(
        _request(tmp_path, target_steps=7),
        config,
        status="completed",
        completed_steps=7,
        usage=UsageLedger(student_train_tokens=1234, optimizer_steps=7),
        checkpoints=[checkpoint],
    )
    atomic_write_json(source_path, prior)
    migration_path = tmp_path / "migration.json"
    _approved_migration(config, source_path, prior, migration_path)
    request = _request(
        tmp_path,
        target_steps=8,
        output_dir=tmp_path / "extended",
        resume_state=checkpoint["state"],
        resume_local_state=source_path,
        resume_migration=migration_path,
    )

    class RemoteReached(RuntimeError):
        pass

    monkeypatch.setattr(
        training_module,
        "_load_trio",
        lambda: (_ for _ in ()).throw(RemoteReached("validated locally")),
    )

    with pytest.raises(RemoteReached, match="validated locally"):
        training_module.asyncio.run(training_module._train_async(config, request))

    extended = json.loads((tmp_path / "extended" / "state.json").read_text(encoding="utf-8"))
    assert extended["completed_steps"] == 7
    assert extended["usage"]["student_train_tokens"] == 1234
    assert extended["checkpoints"] == [checkpoint]
    assert extended["resume_migration"]["source_state_sha256"] == sha256_file(source_path)
    assert json.loads(source_path.read_text(encoding="utf-8")) == prior


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("source_sha", "SHA256"),
        ("method", "method"),
        ("optimizer", "optimizer URI"),
        ("contract", "training contract"),
        ("identity", "student model identity"),
    ],
)
def test_resume_migration_rejects_unapproved_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutation: str,
    message: str,
) -> None:
    config = _workspace(tmp_path, monkeypatch)
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    source_path = source_dir / "state.json"
    checkpoint = {
        "step": 7,
        "state": "pytrio://checkpoint/state-7",
        "sampler_weights": "pytrio://checkpoint/weights-7",
    }
    prior = _local_state(
        _request(tmp_path, target_steps=7),
        config,
        status="completed",
        completed_steps=7,
        usage=UsageLedger(optimizer_steps=7),
        checkpoints=[checkpoint],
    )
    atomic_write_json(source_path, prior)
    migration_path = tmp_path / "migration.json"
    record = _approved_migration(config, source_path, prior, migration_path)
    request_values: dict[str, object] = {}
    if mutation == "source_sha":
        record["source"]["sha256"] = "0" * 64  # type: ignore[index]
    elif mutation == "method":
        record["source"]["method"] = "M4"  # type: ignore[index]
    elif mutation == "optimizer":
        request_values["resume_state"] = "pytrio://checkpoint/wrong"
    elif mutation == "contract":
        record["approved_training_contract"]["seed"] = -1  # type: ignore[index]
    else:
        record["model_identity"]["student"] = "wrong-model"  # type: ignore[index]
    atomic_write_json(migration_path, record)
    request = _request(
        tmp_path,
        target_steps=8,
        output_dir=tmp_path / "extended",
        resume_state=request_values.get("resume_state", checkpoint["state"]),
        resume_local_state=source_path,
        resume_migration=migration_path,
    )

    with pytest.raises(RuntimeError, match=message):
        validate_resume_migration(prior, request, config)
