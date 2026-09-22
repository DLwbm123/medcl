# Missing independent full-supervision Task-CL models

Completion verified and both second-example predictions installed on 2026-09-22.

Started on 2026-09-17 at 10:48 CST. Run ID: `missing_full_task_20260917`.
These are two new independent models for the existing second showcase example.
They are not recovered historical checkpoints or a new continual sequence.

| Task | Dataset | Train slices | Validation slices | Test slices | GPU |
|---|---|---:|---:|---:|---:|
| T3, liver | Lits.h5 | 1,421 | 343 | 496 | RTX 3090, index 2 |
| T4, brain tumor | brain.h5 | 1,963 | 894 | 1,223 | RTX 3090, index 3 |

The original full HDF5 training, validation, and test splits are reused through
read-only references. T4 retains five existing empty training-patient metadata
entries; there are no empty validation or test intervals. This campaign uses
all original training slices, whereas the existing weak-supervision T3/T4
showcase weights used the earlier half-training variant. No direct experimental
comparison between those weights is implied.

Each model starts from seed 42 and uses the existing ZScribbleSeg U-Net with a
binary task head, dense labels, native PCE, 80 epochs, batch size 4, FP32, and
SGD (learning rate 0.03, momentum 0.9, weight decay 0.0001). The polynomial
learning-rate schedule has exponent 0.9 over the complete 80-epoch horizon.
The foreground validation Dice selects `best.pt` once per epoch and at the
end; `last.pt` is retained. The selected model is evaluated on the original
test split once after training, recording both foreground and
background-inclusive Dice. Test results do not select weights.

`scripts/train_missing_showcase.py` calls the existing independent-reference
trainer from an isolated external source snapshot. Its only dataset adapter
uses HDF5's read-only RAM driver to avoid repeated strided NFS reads; no
normalization, resizing, label conversion, or augmentation is replaced.
It uses zero data-loader workers and four PyTorch CPU threads per process.

For reproduction, set `RUN_CONFIG` to a private JSON file with `runtime_root`,
`data_root`, `output`, `task` (`T3` or `T4`), and `device` (`cuda:0`).
The runtime snapshot supplies `runner_core.py`, its existing model modules and
dependencies. Run the entry with `--check` first, then without it. Outputs must
not already exist. On the server, the entry and interpreter use neutral names;
GPU selection and private paths are passed through the environment.

Both real-data checks passed: dense-label forward/backward, finite gradients,
one optimizer update, and strict loading of the corresponding task head.
Each check peaked at 3,017,932,800 allocated GPU bytes. No check weights were
installed into the platform. Training runs independently of SSH/Codex through
detached shell jobs, with stdout logs and an exit-code file per task.

The startup check at 10:50 CST confirmed training PIDs `1285019` (T3) and
`1285020` (T4), each with 4,668 MiB GPU memory, live `running` manifests,
and readable startup logs. Their parent jobs had detached from SSH. The first
epoch had not yet been recorded, so no convergence or completion is claimed.

Private protocol, configurations, data references, runtime snapshot, launch
receipts, checks, and logs are retained under the run directory. Data and
weights are excluded from Git. There is no automatic retry or scheduled
monitoring.

## Completion and installation, 2026-09-22

Both exit-code files contain `0`; both manifests report `complete` with test
evaluation recorded. Each training history contains all 80 epochs (indices
0–79), and both `best.pt` and `last.pt` are present. Each selected checkpoint
loaded strictly into its original task head during the actual showcase export.

| Task | Iterations | Selected epoch (zero-based) | Validation foreground Dice | Test foreground Dice | Test background-inclusive Dice |
|---|---:|---:|---:|---:|---:|
| T3, liver | 28,480 | 35 | 0.947516 | 0.917157 | 0.955094 |
| T4, brain tumor | 39,280 | 49 | 0.879993 | 0.890504 | 0.943272 |

These are the original trainer's recorded aggregate results, not scores for
the showcase case or a continual-learning result matrix. No test evaluation
was repeated during installation. The final recorded training losses are
0.006461 (T3) and 0.005139 (T4).

The existing `scripts/export_showcase_predictions.py` and prepared private
configurations exported the fixed first training case of each task. It checks
the source image against the existing display template before replacing the
label array with model predictions. The resulting files are
`segmentation-task-T3-independent.npz` (711,197 bytes) and
`segmentation-task-T4-independent.npz` (128,751 bytes). Their shapes are
`93×128×128` and `29×128×128`; both contain nonempty foreground and preserve
the existing image and display spacing. Export used one available RTX 3090,
with a neutral process command and no new training.

Both files and their private provenance were installed in the server's
showcase directory; the two files were also installed locally. The live
release successfully loaded all 47 tasks × 2 samples, and all volume examples
passed envelope construction. Browser checks confirmed both new examples'
slice overlays and initialized three-dimensional views. No UI or service
restart was needed, and no platform scoring records were created.
