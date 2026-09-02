# robust_ai_image_detector

Reusable components for robust AI-generated image detection.

## Image distortions

`augmentations/distortions.py` provides nine distortion types with five
severity levels each:

| Name | Severity 1 to 5 |
| --- | --- |
| `jpeg` | JPEG quality: 95, 80, 60, 40, 20 |
| `gaussian_blur` | sigma: 0.5, 1.0, 1.5, 2.0, 3.0 |
| `motion_blur` | kernel size: 3, 5, 7, 11, 15 |
| `gaussian_noise` | sigma: 2, 5, 10, 15, 25 |
| `brightness_shift` | magnitude: 0.05, 0.10, 0.20, 0.30, 0.40 |
| `saturation_shift` | magnitude: 0.10, 0.20, 0.35, 0.50, 0.70 |
| `downsample_upscale` | scale: 0.90, 0.75, 0.60, 0.40, 0.25 |
| `random_crop_resize` | retained area: 0.95, 0.90, 0.80, 0.70, 0.60 |
| `jpeg_ai` | target bpp: 1.00, 0.75, 0.50, 0.25, 0.12 |

The input contract is an HWC RGB `numpy.ndarray` with dtype `uint8`. The
returned image has the same shape and dtype. The second return value records
the distortion name, family, severity, seed, and actual parameters.

```python
from PIL import Image
import numpy as np

from augmentations import apply_distortion

image = np.asarray(Image.open("input.png").convert("RGB"), dtype=np.uint8)
distorted, metadata = apply_distortion(
    image,
    distortion_type="gaussian_blur",
    severity=3,
    seed=42,
)
Image.fromarray(distorted).save("output.png")
print(metadata)
```

Install the built-in runtime dependencies with:

```bash
python3 -m pip install -r requirements.txt
```

Python 3.10 or newer is required.

The first eight distortions run with NumPy and Pillow. `jpeg_ai` deliberately
requires a configured copy of the official JPEG AI reference software; it is
never approximated with legacy JPEG. See
[`docs/jpeg_ai_integration_notes.md`](docs/jpeg_ai_integration_notes.md) for
the external setup and command-line integration details.

Verification status: the eight built-in distortions and the JPEG AI command
contract were tested locally. A real JPEG AI encode/decode was not run in the
current environment because the official runtime and model files were not
installed.

## LoRA adapters

`lora/` trains one LoRA adapter per distortion plus one adapter over all of
them, on top of an already trained detector. The detector head is loaded from a
`best.pt` checkpoint and frozen; only the LoRA matrices inside the visual trunk
are trained. Because LoRA is initialised to zero, the adapted model starts out
bit-for-bit identical to the base detector, and an adapter can be detached at
any time to get the base detector back.

Both released detector variants share the same frozen EVA02-CLIP-B/16 trunk and
differ only in their 769-parameter head, so any adapter attaches to either one.
Whether it *helps* the variant it was not trained against is an empirical
question: pass a different `--head` to measure it.

### 1. Build the training subset

15 000 images are drawn from shard_5, which the detector never trained on. The
images used by the native validation split are excluded first, so early stopping
never sees a training image.

```bash
python3 lora/make_subset.py            # writes lora/lora_train_subset.csv
```

### 2. Train

```bash
HEAD=/data2/aidetection/runs/evaclipb_gap_distorted_only/best.pt

# one adapter over every distortion
python3 lora/lora_train.py --mode all --head "$HEAD"

# one adapter per distortion, trained back to back
python3 lora/lora_train.py --mode per-distortion --head "$HEAD"
```

Each adapter lands in `runs_lora/<name>/` as `adapter_config.json` +
`adapter_model.safetensors`, next to a `metadata.json` recording the head it was
paired with, the distortion policy, and its scores. Every run first prints where
LoRA actually attached and refuses to continue if the target regex matched
nothing.

The distortion list comes from `augmentations.BUILTIN_DISTORTION_NAMES`, so a
newly working distortion becomes another adapter with no code change. Use
`--distortions a,b,c` to train a subset, `--resume` to continue an interrupted
nine-adapter run, and `--precision bf16` to trade exact FP32 parity for speed.

### 3. Run inference

```bash
# single image, all eight per-distortion adapters, then their mean
python3 lora/lora_infer.py --mode ensemble --image sample.jpg --head "$HEAD"

# whole test split, single adapter, ROC-AUC and accuracy over
# all / clean / distorted images
python3 lora/lora_infer.py --mode all --split test --head "$HEAD"

# whole test split, per-adapter and ensemble metrics, predictions to CSV
python3 lora/lora_infer.py --mode ensemble --split test --head "$HEAD" \
    --output predictions/lora_ensemble.csv
```

Split runs also score the detector with the adapter switched off, so every
number has a reference point. Pass `--no-baseline` to skip that pass.
