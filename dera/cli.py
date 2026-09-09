"""Command-line interface for DERA's three-stage workflow."""

from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.metadata
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any, Sequence

import yaml

from dera.data import (
    ValidationReport,
    available_datasets,
    load_dataset_spec,
    validate_dataset,
)
from dera.engine import (
    checkpoint_sha256,
    config_path,
    prepare_config,
    select_checkpoint,
    write_manifest,
    write_selected_checkpoint,
)

_TRAINING_STAGES = ("foundation", "boundary", "residual")


def _checkpoint_reference(value: str) -> str:
    if "://" not in value and not Path(value).expanduser().is_file():
        raise FileNotFoundError(f"Checkpoint does not exist: {value}")
    return str(Path(value).expanduser().resolve()) if "://" not in value else value


def _cfg_options(values: Sequence[str]) -> dict[str, Any]:
    options: dict[str, Any] = {}
    for item in values:
        if "=" not in item:
            raise ValueError(f"Config override must be KEY=VALUE, got {item!r}")
        key, raw_value = item.split("=", 1)
        if not key:
            raise ValueError("Config override key cannot be empty")
        options[key] = yaml.safe_load(raw_value)
    return options


def _require_valid_data(
    dataset: str,
    data_root: str,
    split: str,
    *,
    masks: bool = False,
) -> ValidationReport:
    report = validate_dataset(dataset, data_root, split=split, require_masks=masks)
    print(report.summary())
    for warning in report.warnings[:10]:
        print(f"warning: {warning}")
    if not report.ok:
        details = "\n".join(f"  - {error}" for error in report.errors[:10])
        raise RuntimeError(f"Dataset validation failed:\n{details}")
    return report


def _manifest_input(
    checkpoint: str | None, *, expected_sha256: str | None = None
) -> dict[str, Any] | None:
    if checkpoint is None:
        return None
    result = {"reference": checkpoint}
    if "://" not in checkpoint:
        result["sha256"] = checkpoint_sha256(checkpoint)
    elif expected_sha256:
        result["expected_sha256"] = expected_sha256
    return result


def _runtime_environment() -> dict[str, Any]:
    packages = {}
    for name in (
        "dera-xray",
        "torch",
        "torchvision",
        "mmcv",
        "mmengine",
        "mmdet",
        "transformers",
    ):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None

    environment: dict[str, Any] = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": packages,
    }
    try:
        import torch

        environment["cuda"] = torch.version.cuda
        environment["gpu"] = (
            torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
        )
    except ImportError:
        environment.update(cuda=None, gpu=None)
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=driver_version",
                "--format=csv,noheader",
            ],
            check=False,
            capture_output=True,
            text=True,
        )
        drivers = sorted(set(result.stdout.split())) if result.returncode == 0 else []
        environment["nvidia_driver"] = drivers or None
    except OSError:
        environment["nvidia_driver"] = None
    return environment


def _source_commit() -> str | None:
    repository = Path(__file__).resolve().parents[1]
    # Do not accidentally report a containing research repository's commit.
    if not (repository / ".git").exists():
        return None
    try:
        result = subprocess.run(
            ["git", "-C", str(repository), "rev-parse", "HEAD"],
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError:
        return None
    return result.stdout.strip() if result.returncode == 0 else None


def _run_training(args: argparse.Namespace) -> Path:
    if args.stage != "foundation" and not (args.init_checkpoint or args.resume):
        raise ValueError(
            f"{args.stage} training requires --init-checkpoint or --resume"
        )
    init_checkpoint = (
        _checkpoint_reference(args.init_checkpoint) if args.init_checkpoint else None
    )
    resume = _checkpoint_reference(args.resume) if args.resume else None
    requested_work_dir = Path(args.work_dir).expanduser().resolve()
    if (
        resume is None
        and requested_work_dir.exists()
        and any(requested_work_dir.iterdir())
    ):
        raise ValueError(
            f"Refusing a fresh run in nonempty {requested_work_dir}. "
            "Choose a new --work-dir or use --resume."
        )

    train_report = _require_valid_data(
        args.dataset, args.data_root, "train", masks=args.stage == "boundary"
    )
    validation_report = None
    if args.stage != "boundary":
        spec = load_dataset_spec(args.dataset)
        validation_report = _require_valid_data(
            args.dataset, args.data_root, "val" if "val" in spec["splits"] else "test"
        )

    cfg = prepare_config(
        args.dataset,
        args.stage,
        args.data_root,
        args.work_dir,
        init_checkpoint=init_checkpoint,
        resume=resume,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        cfg_options=_cfg_options(args.cfg_option),
    )
    work_dir = Path(cfg.work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    resolved_config = work_dir / "config_resolved.py"
    cfg.dump(str(resolved_config))
    effective_init = None if resume else (init_checkpoint or cfg.get("load_from"))
    default_init_hash = (
        cfg.get("grounding_dino_pretrained_sha256")
        if effective_init == cfg.get("grounding_dino_pretrained")
        else None
    )
    spec_digest = hashlib.sha256(
        yaml.safe_dump(load_dataset_spec(args.dataset), sort_keys=True).encode()
    ).hexdigest()
    manifest = dict(
        dataset=args.dataset,
        stage=args.stage,
        data_root=str(Path(args.data_root).expanduser().resolve()),
        command=["dera", *sys.argv[1:]],
        source_commit=_source_commit(),
        environment=_runtime_environment(),
        dataset_spec_sha256=spec_digest,
        config=dict(
            source=str(config_path(args.dataset, args.stage).resolve()),
            resolved=str(resolved_config.resolve()),
            resolved_sha256=checkpoint_sha256(resolved_config),
        ),
        annotations={
            "train": train_report.annotation_sha256,
            "validation": (
                validation_report.annotation_sha256
                if validation_report is not None
                else None
            ),
        },
        initialized_from=_manifest_input(
            effective_init, expected_sha256=default_init_hash
        ),
        resumed_from=_manifest_input(resume),
    )
    write_manifest(work_dir, manifest)

    from mmengine.runner import Runner

    runner = Runner.from_cfg(cfg)
    trainable = [
        (name, parameter.numel())
        for name, parameter in runner.model.named_parameters()
        if parameter.requires_grad
    ]
    print(
        f"{args.stage}: {sum(size for _, size in trainable):,} trainable "
        f"parameters across {len(trainable)} tensors"
    )
    manifest["trainable_parameters"] = {
        "count": sum(size for _, size in trainable),
        "tensors": len(trainable),
        "names": [name for name, _ in trainable],
    }
    write_manifest(work_dir, manifest)
    runner.train()
    # Do not reuse a marker left by an earlier interrupted/resumed run.
    selected = select_checkpoint(work_dir, use_marker=False)
    manifest["selected_checkpoint"] = dict(
        path=str(selected), sha256=checkpoint_sha256(selected)
    )
    write_manifest(work_dir, manifest)
    write_selected_checkpoint(work_dir, selected)
    print(f"Selected checkpoint: {selected}")
    return selected


def _cmd_train(args: argparse.Namespace) -> int:
    _run_training(args)
    return 0


def _cmd_pipeline(args: argparse.Namespace) -> int:
    parent: str | None = args.foundation_init
    for stage in _TRAINING_STAGES:
        stage_args = argparse.Namespace(
            dataset=args.dataset,
            stage=stage,
            data_root=args.data_root,
            work_dir=str(Path(args.work_dir) / stage),
            init_checkpoint=parent,
            resume=None,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            cfg_option=args.cfg_option,
        )
        parent = str(_run_training(stage_args))
        gc.collect()
        try:
            import torch

            torch.cuda.empty_cache()
        except (ImportError, RuntimeError):
            pass
    print(f"DERA pipeline complete. Final checkpoint: {parent}")
    return 0


def _cmd_evaluate(args: argparse.Namespace) -> int:
    checkpoint = _checkpoint_reference(args.checkpoint)
    _require_valid_data(args.dataset, args.data_root, "test")
    cfg = prepare_config(
        args.dataset,
        "inference",
        args.data_root,
        args.work_dir,
        init_checkpoint=checkpoint,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        cfg_options=_cfg_options(args.cfg_option),
    )
    from mmengine.runner import Runner

    Runner.from_cfg(cfg).test()
    return 0


def _cmd_infer(args: argparse.Namespace) -> int:
    checkpoint = _checkpoint_reference(args.checkpoint)
    image = Path(args.image).expanduser()
    if not image.is_file():
        raise FileNotFoundError(f"Image does not exist: {image}")
    entities = (
        [item.strip() for item in args.prompts.split(";") if item.strip()]
        if args.prompts
        else load_dataset_spec(args.dataset)["classes"]
    )
    if not entities:
        raise ValueError("At least one prompt is required")
    prompt = ". ".join(entities) + "."

    from mmdet.apis import DetInferencer
    from mmengine.config import Config

    inference_config = Config.fromfile(str(config_path(args.dataset, "inference")))
    inference_config.model.backbone.init_cfg = None
    if args.bert_model:
        inference_config.model.language_model.name = args.bert_model

    inferencer = DetInferencer(
        model=inference_config, weights=checkpoint, device=args.device
    )
    inferencer(
        str(image),
        texts=prompt,
        custom_entities=True,
        pred_score_thr=args.score_threshold,
        out_dir=args.out_dir,
        no_save_vis=False,
        no_save_pred=False,
        print_result=args.print_result,
    )
    return 0


def _cmd_validate(args: argparse.Namespace) -> int:
    splits = (
        list(load_dataset_spec(args.dataset)["splits"])
        if args.split == "all"
        else [args.split]
    )
    success = True
    for split in splits:
        report = validate_dataset(
            args.dataset,
            args.data_root,
            split=split,
            require_masks=args.require_masks and split == "train",
            check_images=not args.skip_images,
        )
        print(report.summary())
        for warning in report.warnings:
            print(f"warning: {warning}")
        for error in report.errors:
            print(f"error: {error}")
        success &= report.ok
    return 0 if success else 1


def _cmd_doctor(_: argparse.Namespace) -> int:
    packages = ("torch", "mmcv", "mmengine", "mmdet", "transformers")
    ok = True
    for package in packages:
        try:
            module = __import__(package)
            print(f"{package}: {getattr(module, '__version__', 'installed')}")
        except ImportError as error:
            ok = False
            print(f"{package}: MISSING ({error})")
    if ok:
        import torch
        from mmcv.ops import MultiScaleDeformableAttention  # noqa: F401

        import dera.models  # noqa: F401

        print(f"CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"GPU: {torch.cuda.get_device_name(0)}")
    return 0 if ok else 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="dera", description="Train and evaluate DERA reproducibly."
    )
    commands = parser.add_subparsers(dest="command", required=True)

    doctor = commands.add_parser("doctor", help="check the runtime environment")
    doctor.set_defaults(func=_cmd_doctor)

    data = commands.add_parser("data", help="dataset utilities")
    data_commands = data.add_subparsers(dest="data_command", required=True)
    validate = data_commands.add_parser("validate", help="validate COCO data")
    validate.add_argument("--dataset", required=True, choices=available_datasets())
    validate.add_argument("--data-root", required=True)
    validate.add_argument(
        "--split", default="all", choices=("train", "val", "test", "all")
    )
    validate.add_argument("--require-masks", action="store_true")
    validate.add_argument("--skip-images", action="store_true")
    validate.set_defaults(func=_cmd_validate)

    def add_runtime_options(command: argparse.ArgumentParser) -> None:
        command.add_argument("--dataset", required=True, choices=available_datasets())
        command.add_argument("--data-root", required=True)
        command.add_argument("--batch-size", type=int)
        command.add_argument("--num-workers", type=int)
        command.add_argument(
            "--cfg-option", action="append", default=[], metavar="KEY=VALUE"
        )

    train = commands.add_parser("train", help="run one training stage")
    add_runtime_options(train)
    train.add_argument("--stage", required=True, choices=_TRAINING_STAGES)
    train.add_argument("--work-dir", required=True)
    handoff = train.add_mutually_exclusive_group()
    handoff.add_argument("--init-checkpoint")
    handoff.add_argument("--resume")
    train.set_defaults(func=_cmd_train)

    pipeline = commands.add_parser("pipeline", help="run all three stages")
    add_runtime_options(pipeline)
    pipeline.add_argument("--work-dir", required=True)
    pipeline.add_argument("--foundation-init")
    pipeline.set_defaults(func=_cmd_pipeline)

    evaluate = commands.add_parser("evaluate", help="evaluate on the test split")
    add_runtime_options(evaluate)
    evaluate.add_argument("--checkpoint", required=True)
    evaluate.add_argument("--work-dir", default="runs/evaluation")
    evaluate.set_defaults(func=_cmd_evaluate)

    infer = commands.add_parser("infer", help="run mask-free image inference")
    infer.add_argument("--dataset", required=True, choices=available_datasets())
    infer.add_argument("--checkpoint", required=True)
    infer.add_argument("--image", required=True)
    infer.add_argument("--prompts", help="semicolon-separated category names")
    infer.add_argument(
        "--bert-model", help="local path or Hugging Face BERT identifier"
    )
    infer.add_argument("--device", default="cuda:0")
    infer.add_argument("--out-dir", default="outputs")
    infer.add_argument("--score-threshold", type=float, default=0.3)
    infer.add_argument("--print-result", action="store_true")
    infer.set_defaults(func=_cmd_infer)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.func(args))
    except (FileNotFoundError, RuntimeError, ValueError) as error:
        parser.exit(2, f"error: {error}\n")


if __name__ == "__main__":
    sys.exit(main())
