<h1>
  <strong>DERA</strong>: Detached Edge-Residual Adaptation for Prohibited item Detection
</h1>

---

<p align="center">
  <a href="https://scholar.google.com/citations?user=1NgtYpwAAAAJ&amp;hl=en">Yonathan Michael</a><sup>1</sup>,
  <a href="https://scholar.google.com/citations?user=dLQ1jLkAAAAJ&amp;hl=en">Mohamad Alansari</a><sup>1</sup>,
  <a href="https://scholar.google.com/citations?user=ylX5MEAAAAAJ&amp;hl=en&amp;oi=ao">Mohammed Bennamoun</a><sup>2</sup>,
  <a href="https://scholar.google.com/citations?user=j5K7HPoAAAAJ&amp;hl=en&amp;oi=ao">Dwarikanath Mahapatra</a><sup>1</sup>,
  <a href="https://scholar.google.com/citations?user=jenl24IAAAAJ&amp;hl=en&amp;oi=ao">Andreas Henschel</a><sup>1</sup>,
  <a href="https://scholar.google.com/citations?user=G_2Xpm0AAAAJ&amp;hl=en">Naoufel Werghi</a><sup>1</sup>
</p>

<p align="center">
  <sup>1</sup>Khalifa University, Abu Dhabi, UAE<br>
  <sup>2</sup>University of Western Australia, Perth, Australia
</p>

<p align="center">
  <a href="https://arxiv.org/abs/2609.12411">📄 Paper</a>
  &nbsp;|&nbsp;
  <a href="https://yonathan-kiflom.github.io/DERA/page" title="Project page">🌐 Project Page</a>
  &nbsp;|&nbsp;
  <a href="https://kuacae-my.sharepoint.com/:f:/g/personal/100053679_ku_ac_ae/IgDrLHr0fF4qQbXOknS8al-cAcCLGkqZoM1esbcxdmI_Rzw?e=9tQfio">📥 Weights</a>
</p>

---

DERA is an edge-aware, text-conditioned detector for prohibited-item
localization in X-ray images. It extends Grounding DINO with a parallel
pixel-difference stream, a mask-derived boundary prior, and two lightweight
residual adapters.

## Pipeline

![DERA model overview](assets/Architecture.jpg)

## Install

Python 3.9, CUDA 12.1 and `mmcv==2.1.0` are the reference setup.

```bash
conda env create -f environment/environment.yml
conda activate dera
pip install -e .
dera doctor
```

The Conda and Docker recipes are in
[docs/REPRODUCE.md](docs/REPRODUCE.md).

## Prepare data

Datasets are not distributed in this repository. Arrange a supported dataset
as described in [docs/REPRODUCE.md](docs/REPRODUCE.md), then validate it before
training:

```bash
dera data validate --dataset pidray --data-root data/PIDray --require-masks
```

The boundary stage requires segmentation annotations. Foundation training,
residual training, evaluation, and inference use detection annotations only.

## Train DERA

Run the complete sequence. The first run downloads the official Grounding
DINO initialization referenced by the foundation config:

```bash
dera pipeline \
  --dataset pidray \
  --data-root data/PIDray \
  --work-dir runs/pidray
```

Or run each stage explicitly:

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

`--init-checkpoint` starts a new stage with a fresh optimizer. Use `--resume`
only to continue an interrupted run of the same stage. Each command records
its validation-selected or latest checkpoint in `selected_checkpoint`. The
full stage and checkpoint contract is in
[docs/REPRODUCE.md](docs/REPRODUCE.md).

## Evaluate and infer

To reproduce our results:

```bash
dera evaluate --dataset pidray --data-root data/PIDray \
  --checkpoint checkpoints/dera_final.pth --work-dir runs/pidray/eval

dera infer --dataset pidray --checkpoint checkpoints/dera_final.pth \
  --image sample.png --prompts "knife; scissors; gun" \
  --out-dir outputs
```



## Repository map

```text
dera/           DERA model and training code
third_party/    attributed PiDiNet/PDC-derived code
configs/        shared recipes plus thin dataset overlays
dataset_specs/  dataset layout and category contracts
tests/          unit, phase-isolation, and smoke tests
docs/           concise reproduction guide
```

## Supported datasets

The configuration interface includes PIDray, CLCXray, and STCray. Users must
obtain each dataset from its official source and comply with its terms.

## Intended use and limitations

DERA is intended for reproducible research and method development. It is not a
certified screening system and must not be the sole basis for safety-critical
decisions. Results depend on scanner characteristics, prompts, annotation
policy, dataset bias, and domain shift and small, occluded, overlapping, or weakly
visible objects may be missed, and confidence scores may be miscalibrated.

## License and citation

DERA code is available under Apache-2.0. Third-party components keep
their own terms; PiDiNet's license is included in [THIRD_PARTY_LICENSES.md](THIRD_PARTY_LICENSES.md).

## Citation

If you use DERA in your research, please cite our paper:

```bibtex
@misc{michael2026deradetachededgeresidualadaptation,
  title={DERA: Detached Edge-Residual Adaptation for Prohibited item Detection},
  author={Yonathan Michael and Mohamad Alansari and Mohammed Bennamoun and
          Dwarikanath Mahapatra and Andreas Henschel and Naoufel Werghi},
  year={2026},
  eprint={2609.12411},
  archivePrefix={arXiv},
  primaryClass={cs.CV},
  url={https://arxiv.org/abs/2609.12411},
}
```

Please also cite Grounding DINO, Swin Transformer, PiDiNet, MMDetection, and the datasets used.
