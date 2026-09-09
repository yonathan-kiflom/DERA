# DERA method

DERA improves text-conditioned X-ray object detection by injecting explicit
local boundary evidence into a hierarchical visual backbone. The output task
remains category-conditioned bounding-box detection.

## Architecture

An image is processed in parallel by Swin Transformer and a compact PiDiNet
feature pyramid. At each of four Swin stages, the matching PiDiNet feature is
projected with a learned `1 x 1` convolution, resized when needed, converted to
tokens, and added to the Swin tokens:

```text
PiDiNet feature -> 1x1 projection -> resize -> tokens --+
                                                        +--> fused stage
Swin stage tokens --------------------------------------+
```

The detector consumes the fused multiscale features together with BERT-encoded
category prompts. Grounding DINO's multimodal encoder and query decoder produce
prompt-aligned scores and bounding boxes.

Two early PiDiNet maps also feed the boundary pathway. Separate side heads
produce logits; these logits are concatenated and fused by a `1 x 1`
convolution. The sigmoid boundary map gates the same early PiDiNet features by
elementwise multiplication. Two zero-initialized projections turn the gated
features into residual tokens, which are added at Swin stages 0 and 1. The
original four foundation fusions remain present.

The public MMDetection registry names are intentionally direct:

- `DERAFoundationSwinTransformer`: four-level foundation fusion;
- `DERASwinTransformer`: foundation fusion plus boundary/residual pathway;
- `DERAGroundingDINO`: phase-aware detector and boundary loss.

## Training

| Stage | Objective | Updated parameters | Fixed parameters |
|---|---|---|---|
| Foundation | Token focal classification, L1 box, and GIoU losses | PiDiNet, four foundation projections, Swin at reduced LR, ChannelMapper, text projection, multimodal encoder, decoder, detection head | BERT language backbone |
| Boundary | Focal loss on mask-derived inner contours | Two boundary side heads and boundary-fusion convolution | Complete foundation and residual projections |
| Residual | Original detection objective | Two zero-initialized residual projections | Foundation and learned boundary head |

Checkpoints transfer in the same order:

```text
Grounding DINO -> foundation -> boundary -> residual/final DERA
```

The boundary stage teaches where object contours are while protecting the
semantic detector. The final stage lets the fixed detector select useful
boundary-gated corrections. Zero initialization makes the residual stage start
from exactly the learned foundation function.

## Inference

Inference uses only an X-ray image and category prompts. DERA predicts its
boundary prior internally; instance masks and external edge maps are not
inputs.
