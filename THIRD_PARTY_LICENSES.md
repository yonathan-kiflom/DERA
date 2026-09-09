# Third-party software

DERA-authored code is licensed under Apache-2.0. Dependencies and isolated
adapted components retain their own terms.

MMDetection, MMEngine, MMCV, Transformers, and PyTorch are installed as
external dependencies and are not copied into this repository. The adapted
PiDiNet/PDC extractor is isolated under `third_party/pidinet` with its upstream
license, pinned lineage, modifications, and only the operators and feature
pyramid needed by DERA. DERA-specific fusion, boundary learning, gating, and
residual adaptation remain under `dera/`.

| Component | How DERA uses it | Upstream terms |
|---|---|---|
| MMDetection | Detector framework and Grounding DINO implementation | Apache-2.0 |
| MMEngine | Training and configuration runtime | Apache-2.0 |
| MMCV | Vision operators | Apache-2.0 |
| Grounding DINO | Architecture and pretrained initialization | Apache-2.0 |
| Swin Transformer | Hierarchical visual backbone | MIT |
| BEFUnet | Intermediate implementation reference for the PiDiNet feature path | MIT |
| PiDiNet | PDC operators and compact feature pyramid adapted under `third_party/pidinet` | Upstream license reproduced verbatim in `third_party/pidinet/LICENSE` |

## PiDiNet restriction

PiDiNet's upstream `LICENSE` begins with the following sentence:

> It is just for research purpose, and commercial use should be contacted with authors first.

The same file then contains an MIT-style permission grant. DERA does not
interpret or resolve that apparent ambiguity. The upstream text is preserved
without alteration. Users are responsible for determining whether their use
is permitted and should contact the PiDiNet authors before commercial use.

See [`third_party/pidinet/README.md`](third_party/pidinet/README.md) for the
pinned provenance, hashes, and changes.

Python packages installed as dependencies are not relicensed by this
repository. Consult each installed distribution for its complete license and
notice files.

## BEFUnet notice

The PiDiNet feature path used an MIT-licensed BEFUnet implementation as an
intermediate reference. Its notice is retained below.

```text
MIT License

Copyright (c) 2024 Omid Nejati

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```
