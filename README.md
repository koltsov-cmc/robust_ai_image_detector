# robust_ai_image_detector

Reusable components for robust AI-generated image detection.

## Image distortions

`augmentations/distortions.py` provides 29 distortion types with five
severity levels each:

| Name | Severity 1 to 5 |
| --- | --- |
| `jpeg` | JPEG quality: 95, 80, 60, 40, 20 |
| `gaussian_blur` | sigma: 0.75, 1.0, 1.5, 2.0, 3.0 |
| `motion_blur` | kernel size: 3, 5, 7, 11, 15 |
| `gaussian_noise` | sigma: 2, 5, 10, 15, 25 |
| `brightness_shift` | magnitude: 0.05, 0.10, 0.20, 0.30, 0.40 |
| `saturation_shift` | magnitude: 0.10, 0.20, 0.35, 0.50, 0.70 |
| `downsample_upscale` | scale: 0.80, 0.65, 0.50, 0.35, 0.20 |
| `random_crop_resize` | retained area: 0.95, 0.90, 0.80, 0.70, 0.60 |
| `jpeg_ai` | target bpp: 1.00, 0.75, 0.50, 0.25, 0.12 |
| `lens_blur` | disk radius: 1, 2, 4, 6, 8 |
| `color_shift` | spatial channel shift: 1, 3, 6, 8, 12 |
| `impulse_noise` | density: 0.001, 0.005, 0.010, 0.015, 0.020 |
| `jitter` | displacement: 0.05, 0.10, 0.20, 0.50, 1.00 |
| `quantization` | levels: 20, 16, 13, 10, 7 |
| `linear_contrast_change` | amount: 0.00, 0.15, -0.40, 0.30, -0.60 |
| `multiplicative_noise` | variance: 0.001, 0.005, 0.010, 0.015, 0.035 |
| `pixelate` | strength: 0.01, 0.05, 0.10, 0.20, 0.50 |
| `rgb_shift` | channel radius: 10, 20, 30, 40, 50 |
| `random_aspect_crop_resize` | retained fraction: 0.8, 0.7, 0.6, 0.5, 0.4 |
| `jpeg_recompression_1` | recompressions: 2, 3, 3, 4, 5 |
| `jpeg_recompression_2` | progressively harsher quality ranges |
| `jpeg_recompression_comb` | JPEG/JPEG2000 rounds: 2, 3, 3, 4, 5 |
| `jpeg2000` | compression ratio: 16, 32, 45, 120, 170 |
| `glass_blur` | maximum displacement: 1, 2, 3, 4, 6 |
| `random_tone_curve` | scale: 0.05, 0.15, 0.20, 0.30, 0.40 |
| `clahe` | progressively higher clip-limit ranges |
| `iso_noise` | progressively higher intensity ranges |
| `shot_noise` | progressively higher scale ranges |
| `perspective` | progressively larger warp ranges |

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

For the distortion module alone, install only the lightweight runtime
dependencies:

```bash
python3 -m pip install "numpy>=1.24,<3" "Pillow>=10,<13"
```

The repository-level `requirements.txt` installs the full detector training
stack, including CUDA-enabled PyTorch, and is not required for distortion-only
use. Python 3.10 or newer is required.

The 28 built-in distortions run with NumPy and Pillow. `jpeg_ai` deliberately
requires a configured copy of the official JPEG AI reference software; it is
never approximated with legacy JPEG. See
[`docs/jpeg_ai_integration_notes.md`](docs/jpeg_ai_integration_notes.md) for
the external setup and command-line integration details.

The 20 added lightweight transforms reuse the five-level parameter tables from
`aug_utils_val_private/utils_data.py`, but replace its heavy
PyTorch/Kornia/Albumentations implementations with dependency-free
NumPy/Pillow approximations. Their metadata records the original transform
name, archive SHA-256, and `compatibility="parameter-table-only"` boundary.

Verification status: all 28 built-in distortions were tested at all five
severity levels for deterministic output, preserved shape and dtype, and
JSON-serializable metadata. The JPEG AI command contract was also tested. A
real JPEG AI encode/decode was not run because the official runtime and model
files were not installed.
