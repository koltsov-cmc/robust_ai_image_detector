# JPEG AI reference software: CLI and integration notes

> This is a team-maintained integration note based on the official JPEG AI
> reference software documentation. It is not an official JPEG committee
> document.

**Checked:** 2026-09-01 (Asia/Shanghai)
**Primary source checkout:** the official GitLab repository's `main` branch, resolved to
`328b1f3fce76a778d306403dc8b89f3f9eac83fd` on the check date. JPEG.org links to this GitLab
project as the JPEG AI Reference Software.

## Source record

- [JPEG.org — JPEG AI Software](https://jpeg.org/jpegai/software.html) (the JPEG committee's
  software page; links to GitLab).
- [Official repository at the checked commit](https://gitlab.com/wg1/jpeg-ai/jpeg-ai-reference-software/-/tree/328b1f3fce76a778d306403dc8b89f3f9eac83fd)
- [README](https://gitlab.com/wg1/jpeg-ai/jpeg-ai-reference-software/-/blob/328b1f3fce76a778d306403dc8b89f3f9eac83fd/README.md)
- [Command-line tools documentation](https://gitlab.com/wg1/jpeg-ai/jpeg-ai-reference-software/-/blob/328b1f3fce76a778d306403dc8b89f3f9eac83fd/docs/architecture/10-command-line-tools.md)
- [Encoding pipeline documentation](https://gitlab.com/wg1/jpeg-ai/jpeg-ai-reference-software/-/blob/328b1f3fce76a778d306403dc8b89f3f9eac83fd/docs/architecture/04-encoding-pipeline.md)
- [Decoding pipeline documentation](https://gitlab.com/wg1/jpeg-ai/jpeg-ai-reference-software/-/blob/328b1f3fce76a778d306403dc8b89f3f9eac83fd/docs/architecture/05-decoding-pipeline.md)

All technical claims below are from the checked repository, unless explicitly labelled
**unresolved** or **source-derived**.

## Confirmed facts

### Repository and license

The official clone URL is:

```bash
git clone https://gitlab.com/wg1/jpeg-ai/jpeg-ai-reference-software.git
cd jpeg-ai-reference-software
```

This is the command in the repository README ([README lines 7–12](https://gitlab.com/wg1/jpeg-ai/jpeg-ai-reference-software/-/blob/328b1f3fce76a778d306403dc8b89f3f9eac83fd/README.md#L7-12)).

The repository's [LICENSE](https://gitlab.com/wg1/jpeg-ai/jpeg-ai-reference-software/-/blob/328b1f3fce76a778d306403dc8b89f3f9eac83fd/LICENSE#L1-31)
identifies the code as BSD-licensed and contains the standard source/binary redistribution
conditions and no-endorsement clause. For compliance, preserve the license text and notices in
redistributions; the license also warns that third-party/contributor and patent rights may apply
separately.

### Host, Python, CUDA, and model/runtime dependencies

The README states Ubuntu Linux 18.04 or later and CUDA 10.2+ or 11.3+; it lists doxygen 1.8.13,
graphviz 2.40.1, and git-lfs 3.0.2 as system packages ([README lines 14–20](https://gitlab.com/wg1/jpeg-ai/jpeg-ai-reference-software/-/blob/328b1f3fce76a778d306403dc8b89f3f9eac83fd/README.md#L14-20)).
The repository setup script additionally installs `python3-dev` and `git-lfs`
([setup_system.sh lines 35–38](https://gitlab.com/wg1/jpeg-ai/jpeg-ai-reference-software/-/blob/328b1f3fce76a778d306403dc8b89f3f9eac83fd/scripts/setup_system.sh#L35-38)).

The documented Conda setup creates Python 3.7, activates the environment, installs
`requirements.txt`, installs pre-commit hooks, and builds the test libraries
([setup_env.sh lines 36–51](https://gitlab.com/wg1/jpeg-ai/jpeg-ai-reference-software/-/blob/328b1f3fce76a778d306403dc8b89f3f9eac83fd/scripts/setup_env.sh#L36-51)).
The key pinned runtime packages include `torch==1.10.2`, `torchvision==0.11.3`,
`numpy==1.19.1`, `opencv-python==4.5.5.62`, `pybind11`, and `pytorch-msssim`; the full list is
in [requirements.txt](https://gitlab.com/wg1/jpeg-ai/jpeg-ai-reference-software/-/blob/328b1f3fce76a778d306403dc8b89f3f9eac83fd/requirements.txt#L1-36).

Model files and the repository image sets are stored with Git LFS. A fresh checkout must run:

```bash
git lfs fetch
git lfs checkout
```

The requirement and rationale are documented in the CLI guide ([command-line tools lines 47–56](https://gitlab.com/wg1/jpeg-ai/jpeg-ai-reference-software/-/blob/328b1f3fce76a778d306403dc8b89f3f9eac83fd/docs/architecture/10-command-line-tools.md#L47-56)) and in the README ([README lines 31–36](https://gitlab.com/wg1/jpeg-ai/jpeg-ai-reference-software/-/blob/328b1f3fce76a778d306403dc8b89f3f9eac83fd/README.md#L31-36)).

The C++ extensions are built by `make build_test_libs`, which calls the `mans` and `direct`
entropy-coder Makefiles ([build_ec_lib.sh lines 44–48](https://gitlab.com/wg1/jpeg-ai/jpeg-ai-reference-software/-/blob/328b1f3fce76a778d306403dc8b89f3f9eac83fd/scripts/build_ec_lib.sh#L44-48)).
Those Makefiles require `g++`, Python's `pybind11` include discovery, `python3-config`, C++14,
and OpenMP (`-std=c++14 -fopenmp`) ([mans Makefile lines 1–10](https://gitlab.com/wg1/jpeg-ai/jpeg-ai-reference-software/-/blob/328b1f3fce76a778d306403dc8b89f3f9eac83fd/src/codec/entropy_coding/cpp_exts/mans/Makefile#L1-10),
[direct Makefile lines 1–10](https://gitlab.com/wg1/jpeg-ai/jpeg-ai-reference-software/-/blob/328b1f3fce76a778d306403dc8b89f3f9eac83fd/src/codec/entropy_coding/cpp_exts/direct/Makefile#L1-10)).
The one-shot repository targets are `make configure` (system + environment), then
`make build_test_libs`; the target meanings are listed in [the CLI guide lines 58–66](https://gitlab.com/wg1/jpeg-ai/jpeg-ai-reference-software/-/blob/328b1f3fce76a778d306403dc8b89f3f9eac83fd/docs/architecture/10-command-line-tools.md#L58-66).

### Encoder invocation

The documented reconstruction encoder invocation is:

```bash
conda activate jpeg_ai_vm
python -m src.reco.coders.encoder <IMAGE_PATH> <OUTPUT_STREAM_PATH> \
  [--set_target_bpp <TARGET_BPPm100>] \
  [--cfg <CFG1> [<CFG2> [<CFG3> ...]]]
```

This is copied from [README lines 54–61](https://gitlab.com/wg1/jpeg-ai/jpeg-ai-reference-software/-/blob/328b1f3fce76a778d306403dc8b89f3f9eac83fd/README.md#L54-61).
The current CLI guide gives the same module with an `.bits` output example and documents the
options ([command-line tools lines 82–102](https://gitlab.com/wg1/jpeg-ai/jpeg-ai-reference-software/-/blob/328b1f3fce76a778d306403dc8b89f3f9eac83fd/docs/architecture/10-command-line-tools.md#L82-102)).

Confirmed encoder arguments:

- `input_path`: positional PNG input.
- `bin_path`: positional output binary/bitstream path.
- `--cfg A.json [B.json ...]`: configuration files, merged left to right.
- `--bpp_idx N`: index into the configured `target_bpps` list (default index 0).
- `--set_target_bpp N`: target bpp multiplied by 100; e.g. `50` denotes a 0.50-bpp target.
- `-r/--rec_path`: optional encoder-side reconstructed PNG.
- `--output_bit_depth {8,10}`: optional reconstruction bit-depth override.
- `--calc_metrics`: print the bpp and enabled quality metrics.

The parser and argument meanings are in [encoder.py lines 45–68](https://gitlab.com/wg1/jpeg-ai/jpeg-ai-reference-software/-/blob/328b1f3fce76a778d306403dc8b89f3f9eac83fd/src/codec/coders/encoder.py#L45-68)
and [the CLI guide lines 88–102](https://gitlab.com/wg1/jpeg-ai/jpeg-ai-reference-software/-/blob/328b1f3fce76a778d306403dc8b89f3f9eac83fd/docs/architecture/10-command-line-tools.md#L88-102).

`--set_target_bpp N` is not merely a label: the encoder source replaces the effective target list
with `[N]`, sets `bpp_idx` to zero, and appends `cfg/BRM/regen_list.json`
([encoder.py lines 121–142](https://gitlab.com/wg1/jpeg-ai/jpeg-ai-reference-software/-/blob/328b1f3fce76a778d306403dc8b89f3f9eac83fd/src/codec/coders/encoder.py#L121-142)).
The bitrate matcher then searches for a beta/operating point; the BRM README says `regen_list.json`
is for per-image/rate matching (up to 10%), while `use_list.json` consumes pre-generated beta lists
([BRM README lines 1–9](https://gitlab.com/wg1/jpeg-ai/jpeg-ai-reference-software/-/blob/328b1f3fce76a778d306403dc8b89f3f9eac83fd/cfg/BRM/README.md#L1-9)).

The repository also documents configuration stacking for anchor/all-tools/selected-tools runs,
for example `--cfg cfg/tools_on.json cfg/profiles/base.json`
([command-line tools lines 104–115](https://gitlab.com/wg1/jpeg-ai/jpeg-ai-reference-software/-/blob/328b1f3fce76a778d306403dc8b89f3f9eac83fd/docs/architecture/10-command-line-tools.md#L104-115)).

### Decoder invocation and integration boundary

The documented decoder invocation is:

```bash
conda activate jpeg_ai_vm
python -m src.reco.coders.decoder <INPUT_STREAM_PATH> <OUTPUT_PNG_IMAGE_PATH>
```

([README lines 64–71](https://gitlab.com/wg1/jpeg-ai/jpeg-ai-reference-software/-/blob/328b1f3fce76a778d306403dc8b89f3f9eac83fd/README.md#L64-71)).
The decoder also accepts `--device {cpu,gpu}`, `--ori_file`, `--calc_metrics`,
`--calc_ptflops`, `--use_yuv 0/1`, and `--output_bit_depth {8,10}`
([decoder.py lines 47–74](https://gitlab.com/wg1/jpeg-ai/jpeg-ai-reference-software/-/blob/328b1f3fce76a778d306403dc8b89f3f9eac83fd/src/codec/coders/decoder.py#L47-74)).

There is deliberately no decoder `--cfg`: the decoder parser is created with `has_cfg=False` and
the common coder forces the decoder configuration to the pipeline description
([decoding pipeline lines 5–12](https://gitlab.com/wg1/jpeg-ai/jpeg-ai-reference-software/-/blob/328b1f3fce76a778d306403dc8b89f3f9eac83fd/docs/architecture/05-decoding-pipeline.md#L5-12),
[CLI guide lines 117–134](https://gitlab.com/wg1/jpeg-ai/jpeg-ai-reference-software/-/blob/328b1f3fce76a778d306403dc8b89f3f9eac83fd/docs/architecture/10-command-line-tools.md#L117-134)).
Thus an integration must retain the repository's `cfg/pipeline.json` and matching model files on
the decoder side; encoder-only config changes are not a substitute for this pipeline.

### Selecting five rate/quality points

The current `cfg/pipeline.json` contains exactly five configured target-rate integers:

```json
"target_bpps": [12, 25, 50, 75, 100]
```

The repository defines these values as bpp multiplied by 100, so these correspond to nominal
targets 0.12, 0.25, 0.50, 0.75, and 1.00 bpp. The list is in [pipeline.json lines 1–5](https://gitlab.com/wg1/jpeg-ai/jpeg-ai-reference-software/-/blob/328b1f3fce76a778d306403dc8b89f3f9eac83fd/cfg/pipeline.json#L1-5),
and the ×100 convention is stated in [the configuration README lines 37–39](https://gitlab.com/wg1/jpeg-ai/jpeg-ai-reference-software/-/blob/328b1f3fce76a778d306403dc8b89f3f9eac83fd/cfg/README.md#L37-39).

For this built-in five-point list, the confirmed selection mechanism is `--bpp_idx 0`, `1`, `2`,
`3`, or `4` in five separate encoder invocations; `--bpp_idx` is explicitly documented as the
index into `target_bpps` ([CLI guide lines 88–95](https://gitlab.com/wg1/jpeg-ai/jpeg-ai-reference-software/-/blob/328b1f3fce76a778d306403dc8b89f3f9eac83fd/docs/architecture/10-command-line-tools.md#L88-95)).
For a single exact target, use the documented `--set_target_bpp N` form, once per target; `N` is
the desired bpp ×100.

The underlying parameter declaration supports a list (`target_bpps`, integer, `nargs='+'`) in
[coding_engine/params.py lines 36–42](https://gitlab.com/wg1/jpeg-ai/jpeg-ai-reference-software/-/blob/328b1f3fce76a778d306403dc8b89f3f9eac83fd/src/codec/coding_tools/coding_engine/params.py#L36-42),
and the evaluation harness passes one target integer per encode job
([reco/scripts/eval.py lines 43–54](https://gitlab.com/wg1/jpeg-ai/jpeg-ai-reference-software/-/blob/328b1f3fce76a778d306403dc8b89f3f9eac83fd/src/reco/scripts/eval.py#L43-54)).
The harness enumerates configured `target_bpps` and applies per-image/per-rate overrides
([CLI guide lines 142–164](https://gitlab.com/wg1/jpeg-ai/jpeg-ai-reference-software/-/blob/328b1f3fce76a778d306403dc8b89f3f9eac83fd/docs/architecture/10-command-line-tools.md#L142-164)).

Do not confuse the `simple`, `base`, and `high` profile files with five quality points: they select
different operating-point/profile network configurations. Rate selection is through `target_bpps`
and the bitrate matcher. The profile/configuration distinction is stated in [cfg/README.md lines 17–43](https://gitlab.com/wg1/jpeg-ai/jpeg-ai-reference-software/-/blob/328b1f3fce76a778d306403dc8b89f3f9eac83fd/cfg/README.md#L17-43).

### Output format

The encoder writes a binary bitstream to the `bin_path`; the documented filename convention is
`<codec>_<name>_<bpp:03d>.bits` (example `JAI_00030_TE_050.bits`), while the reconstruction is
PNG ([CLI guide lines 88–102](https://gitlab.com/wg1/jpeg-ai/jpeg-ai-reference-software/-/blob/328b1f3fce76a778d306403dc8b89f3f9eac83fd/docs/architecture/10-command-line-tools.md#L88-102),
[lines 64–76](https://gitlab.com/wg1/jpeg-ai/jpeg-ai-reference-software/-/blob/328b1f3fce76a778d306403dc8b89f3f9eac83fd/docs/architecture/11-evaluation-and-testing.md#L64-76)).

The output is a raw binary sequence of JPEG AI substreams as implemented by this reference
software. The writer opens the output with `"wb"`, writes the SOC marker, each substream, and the
EOC marker ([bitstream_structure.py lines 108–124](https://gitlab.com/wg1/jpeg-ai/jpeg-ai-reference-software/-/blob/328b1f3fce76a778d306403dc8b89f3f9eac83fd/src/codec/bitstream_structure/bitstream_structure.py#L108-124)).
The architecture documentation describes the sequence as
`SOC · [PIH · TON · SOQ · SOZ · SORP · SORS · UDI · RDI] · EOC`, with optional substreams depending
on configuration ([encoding pipeline lines 239–265](https://gitlab.com/wg1/jpeg-ai/jpeg-ai-reference-software/-/blob/328b1f3fce76a778d306403dc8b89f3f9eac83fd/docs/architecture/04-encoding-pipeline.md#L239-265)).
Use `.bits` for interoperability with the repository's evaluator and decoder filename parser;
the source does not document a separate public container extension or a JPEG/PNG wrapper for the
encoded bitstream.

### Recording actual bpp

The repository's metric implementation defines actual bpp exactly as:

```text
actual_bpp = 8 * os.path.getsize(bitstream_file) / (shape[-2] * shape[-1])
```

This is [MetricsProcessor.compute_bpp](https://gitlab.com/wg1/jpeg-ai/jpeg-ai-reference-software/-/blob/328b1f3fce76a778d306403dc8b89f3f9eac83fd/src/codec/metrics/metrics.py#L578-582).
`shape[-2]` and `shape[-1]` are the reconstruction/image height and width. Therefore the measured
size includes the complete `.bits` file (markers, substream headers, and payload), not only neural
latent payload bytes.

The encoder calls this bpp calculation when `--calc_metrics` is supplied and prints it in the
`=== Metrics ===` result ([coder.py lines 352–373](https://gitlab.com/wg1/jpeg-ai/jpeg-ai-reference-software/-/blob/328b1f3fce76a778d306403dc8b89f3f9eac83fd/src/codec/coders/coder.py#L352-373)).
For an external benchmark, record at minimum: source image width/height, bitstream byte size,
`actual_bpp`, requested target integer (`target_bppm100`), selected config/profile, and whether
the encoder used `--set_target_bpp` search or a pre-generated/default beta list.

## Unresolved or potentially misleading details

1. **No one-shot five-encode shell command is documented.** The repository documents the five
   configured points and the `--bpp_idx` selector, and the evaluation harness enumerates points,
   but it does not provide a standalone command that emits all five direct-CLI outputs in one call.
   Run five invocations (indices 0–4) or use the official evaluation harness; do not claim that one
   direct encoder process writes five streams.
2. **Exact custom-list CLI spelling is not shown in the README.** The source declares
   `target_bpps` with `nargs='+'` and the evaluator uses the internal `-target_bpps` override, but
   the public encoder usage line only documents `--set_target_bpp`. A custom five-value list should
   be supplied through a checked configuration/evaluation path and validated on the exact checkout;
   this note intentionally does not present an undocumented custom command as guaranteed API.
3. **Target bpp is a request, not the measured result.** `--set_target_bpp` activates a search and
   the BRM documentation only promises matching “up to 10%”; always report measured bpp using the
   complete-file formula above.
4. **Version drift matters.** The checked `main` commit is `328b1f3f…`; older GitLab snapshots
   expose an earlier `src.reco`/configuration layout. Pin the commit used for an experiment and
   cite its files, because CLI/configuration details can change.
5. **No claim is made here about normative final file-format certification.** This report describes
   the reference software's binary writer and `.bits` convention. The repository itself is the
   authoritative source for this implementation integration, while standard conformance claims
   should be checked against the applicable JPEG AI standard/conformance documents.
