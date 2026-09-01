# robust_ai_image_detector

Reusable components for robust AI-generated image detection.

## Image distortions

`augmentations/distortions.py` provides nine distortion types with five
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
