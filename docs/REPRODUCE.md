# Reproducing DERA

This guide contains the supported installation, dataset, training, checkpoint,
and release protocol. Run commands from the repository root.

## 1. Install

The reference setup is Linux with an NVIDIA driver compatible with CUDA 12.1.
The pinned recipe uses Python 3.9, PyTorch 2.1.2/CUDA 12.1, MMCV 2.1.0,
MMEngine 0.10.7, MMDetection 3.3.0, and Transformers 4.28.0. MMCV 2.1.0
respects MMDetection's declared `mmcv < 2.2.0` requirement; this repository
does not bypass that compatibility check.

```bash
conda env create -f environment/environment.yml
conda activate dera
pip install -e .
dera doctor
```

`dera doctor` checks the core packages and DERA registry and reports CUDA/GPU
availability. Resolve every reported error before training. The requirements
file pins direct dependencies, not every transitive dependency; retain the
doctor output and run manifest for each published run.

For an offline BERT copy, point the configuration to its local directory:

```bash
dera train --dataset pidray --stage foundation \
  --data-root data/PIDray --work-dir runs/pidray/foundation \
  --cfg-option model.language_model.name=pretrained/bert-base-uncased
```

Alternatively, build and check the container:

```bash
docker build -f environment/Dockerfile -t dera:0.1.0 .
docker run --rm --gpus all dera:0.1.0 doctor
```

Mount data and outputs rather than copying them into the image:

```bash
export DERA_DATASETS="$(pwd)/data"
export DERA_RUNS="$(pwd)/runs"
docker run --rm --gpus all \
  -v "$DERA_DATASETS":/datasets:ro \
  -v "$DERA_RUNS":/runs \
  dera:0.1.0 data validate \
  --dataset pidray --data-root /datasets/PIDray --require-masks
```

For provenance, the archived experiments used PyTorch 2.4.1, CUDA 12.1,
MMCV 2.2.0, MMEngine 0.10.7, MMDetection 3.3.0 source, and Transformers
4.28.0. That stack required bypassing MMDetection's version guard and is not
the supported release environment.

## 2. Prepare data

DERA does not redistribute datasets, images, or generated masks. Obtain each
dataset from its source and comply with its terms.

| Dataset | Source | Expected layout |
|---|---|---|
| PIDray | [security-dataset](https://github.com/bywang2018/security-dataset) | `train2017/`, `val2017/`, `fulltest/`, `annotations/` |
| CLCXray | [CLCXray](https://github.com/GreysonPhoenix/CLCXray) | `train2017/`, `val2017/`, `test2017/`, `annotations/` |
| STCray | [STING-BEE](https://divs1159.github.io/STING-BEE/) | `train/`, `test/`, `annotations/STCray_{train,test}.json` |

Each YAML file in `dataset_specs/` is the authoritative contract for category
order, relative annotation and image paths, split roles, and mask requirements.
Detection annotations use COCO conventions; boundary supervision additionally
requires a decodable polygon or RLE segmentation for every supervised instance.
No workstation-specific paths are encoded in the specs or configurations.

The configured split protocol is:

- **PIDray:** train with `annotations/instances_train2017.json`, validate with
  `instances_val2017.json`, and test with `fulltest.json`.
- **CLCXray:** train and supervise boundaries with
  `instances_train2017_sam2.json`, validate with `instances_val2017.json`, and
  test with `instances_test2017.json`.
- **STCray:** train with `STCray_train.json`; use the complete
  `STCray_test.json` split for both validation and testing. No train-derived
  validation split is created.

Validate detection data, or include mask validation before boundary training:

```bash
dera data validate --dataset pidray --data-root data/PIDray
dera data validate --dataset pidray --data-root data/PIDray --require-masks
```

The validator checks image references, IDs, category order, box geometry, and,
when requested, mask presence and decoding. PIDray's official full-test JSON
capitalizes three category names differently from train/validation; the shipped
spec preserves the same IDs and order while recording those test-only names.

Use each declared training split for optimization and its declared test split
for evaluation, unless the spec provides an official validation split. Do not
create an undocumented random split. Pseudo-masks supervise only the boundary
stage and are never read at inference. Record their generator, revision and
checkpoint checksum, settings, seed, and output-annotation checksum.

## 3. Train

Stages are isolated and checkpoints transfer in order:

| Stage | Trainable | Frozen |
|---|---|---|
| Foundation | PiDiNet, four PiDiNet projections, Swin at reduced LR, detector modules | BERT |
| Boundary | Two side heads and their fusion convolution | Complete foundation and residual projections |
| Residual | Two zero-initialized residual projections | Complete foundation and boundary head |

Run the complete sequence:

```bash
dera pipeline \
  --dataset pidray \
  --data-root data/PIDray \
  --work-dir runs/pidray
```

The pipeline validates annotations, obtains the foundation configuration's
official Grounding DINO initialization when absent from the cache, trains all
three stages, and passes each selected checkpoint forward. To run stages
separately:

```bash
dera train --dataset pidray --stage foundation \
  --data-root data/PIDray --work-dir runs/pidray/foundation

dera train --dataset pidray --stage boundary \
  --data-root data/PIDray --work-dir runs/pidray/boundary \
  --init-checkpoint "$(cat runs/pidray/foundation/selected_checkpoint)"

dera train --dataset pidray --stage residual \
  --data-root data/PIDray --work-dir runs/pidray/residual \
  --init-checkpoint "$(cat runs/pidray/boundary/selected_checkpoint)"
```

`--init-checkpoint` imports weights into a new stage with a fresh optimizer and
schedule. `--resume` instead restores an interrupted run of the same stage,
including optimizer, scheduler, scaler, and iteration state; the options are
mutually exclusive. A fresh run rejects a nonempty work directory. Each stage
writes `selected_checkpoint` (validation-best when available, otherwise the
latest completed checkpoint) and `run_manifest.json`.

For offline foundation initialization, download the official checkpoint in
advance and pass `--foundation-init /path/to/checkpoint.pth` to the pipeline,
or `--init-checkpoint /path/to/checkpoint.pth` to foundation training.

MMEngine overrides are supported, for example:

```bash
dera train --dataset pidray --stage foundation \
  --data-root data/PIDray --work-dir runs/debug \
  --cfg-option train_dataloader.batch_size=1
```

An overridden run is noncanonical; keep its resolved config. Lower the batch
size if memory is insufficient, but do not silently skip failed samples or
change stage identity, checkpoint lineage, or dataset paths. Before optimizing,
inspect the printed trainable parameters. A smoke step must verify that boundary
training changes only `edge_side_heads` and `edge_fuse`, residual training
changes only `edge_residual_projs`, residual weights and biases start at zero,
and inference succeeds without `gt_masks`.

## 4. Manage checkpoints

Checkpoint binaries are release assets, not Git files. List every public
artifact in [`checkpoints/manifest.json`](../checkpoints/manifest.json).
Foundation training starts from MMDetection's official
`grounding_dino_swin-t_pretrain_obj365_goldg_grit9m_v3det_20231204_095047-b448804b.pth`,
whose SHA-256 is
`b448804bb1af6fa688887f0f2454625edbeeae4e868bc95620e3e6413581051a`.
The download URL is stored in the foundation model configuration.

The artifact types are foundation handoff, boundary handoff, final
residual-adapted inference weights, and optional same-run resume state. Never
use resume state as a stage-to-stage initialization. Each manifest entry must
record the stable name, stage, dataset and category order, URL, byte size,
SHA-256, DERA version and commit, resolved-config path and hash, parent name and
hash, annotation and image-manifest hashes, and included optimizer/scheduler/
scaler state. Final inference weights should omit training state and local
filesystem metadata.

```bash
sha256sum checkpoints/dera_final.pth
```

## 5. Record and verify a run

Before training, use a clean tagged commit, run `dera doctor`, validate data and
masks, verify the initialization hash, and fix the prompt order and random seed.
The release recipe fixes seed `2026`. The archived PIDray foundation run did
not record an explicit seed, so bitwise regeneration of that historical
checkpoint is not claimed.

Keep the resolved config, exact command and commit, package/CUDA/driver/GPU
inventory, dataset spec and annotation hashes, parent-checkpoint hash, seed,
precision, batch size, overrides, log, and checkpoint-selection metric beside
each stage. When refactoring dependencies or code, fixed-input equivalence must
cover the PiDiNet pyramid, projected and four fused Swin features, two side
logits, fused boundary probability, two residual tensors, detector logits and
boxes, plus the exact parameter set changed in every stage.

From a new clone and empty environment, run:

```bash
conda env create -f environment/environment.yml
conda activate dera
pip install -e .
dera doctor
dera data validate --dataset pidray --data-root data/PIDray --require-masks
pytest -q
```

Finally, evaluate with a checksum-verified final checkpoint. Report the dataset
release and split, prompt order, checkpoint hash, environment, resolved config,
seed, and COCO-style metric output; do not compare different split definitions
or prompt sets without stating the difference.
