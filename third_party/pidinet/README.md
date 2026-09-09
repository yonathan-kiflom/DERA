# PiDiNet component

DERA ships a compact adaptation of the pixel-difference operators, residual
PDC blocks, and four-level feature pyramid used by its parallel edge stream.
DERA-owned projections, fusion, boundary learning, gating, and residual
adaptation remain outside this directory.

## Provenance

- Project: PiDiNet: Pixel Difference Networks for Efficient Edge Detection
- Repository: <https://github.com/hellozhuo/pidinet>
- Reference commit: `d21aa881ed9c628571636fad39acfe1fad517ebd`
- Reference branch: `master`
- Retrieved: 2026-09-08
- Relevant files: `models/ops.py`, `models/pidinet.py`, `models/config.py`
- License source: upstream `LICENSE` at the reference commit
- Upstream `LICENSE` SHA-256:
  `a6a61f9c094a201639427a73c1317ce13a5412a4e49fefac33f7402ad181e09f`

The extraction was developed from the PiDiNet feature path in BEFUnet:

- Repository: <https://github.com/Omid-Nejati/BEFUnet>
- Reference snapshot: `f25a142f2a3125c29e50e6c818e18ddc3309897e`
- Reference file: `models/pidinet.py`
- Reference-file SHA-256:
  `ceb2233413581637d6c321fd15a48101f9f6566dbaca9a730bd7c5791773d477`

BEFUnet's README acknowledges that its PiDiNet code came from the official
PiDiNet project. The compact adaptation here is not a complete mirror of
either repository. Its pre-extraction DERA research implementation had
SHA-256 `8aaa8c0e7c76024fec2c615e87dbe7aae778ea2910491f9f742f8843a08c440e`.

## Changes for DERA

- Removed standalone training, testing, conversion, and visualization tools.
- Removed unused edge reducers, classifier, dilation, and spatial-attention
  modules.
- Return four intermediate feature maps instead of decoded edge predictions.
- Package central, angular, radial, and ordinary convolution dispatch as a
  small importable module.
- Use PyTorch-native initialization, explicit validation, typing, and library
  documentation.
- Exclude pretrained PiDiNet weights and keep DERA fusion code separate.

The upstream PiDiNet license is reproduced verbatim in [LICENSE](LICENSE). It
begins with a research-purpose restriction and then contains an MIT-style
grant; this repository does not reinterpret those terms. Review
[THIRD_PARTY_LICENSES.md](../../THIRD_PARTY_LICENSES.md) and contact the
PiDiNet authors before commercial use.
