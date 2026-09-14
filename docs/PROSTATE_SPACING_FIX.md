# Prostate display spacing correction · 2026-09-14

The gallery preparation stored only preview strides `[1,2,2]` as Z/Y/X spacing. The six source HDF5 files do not contain physical geometry. Both the MPR volume loader and segmentation surface conversion already consume spacing correctly; the error originated in the prepared gallery metadata.

After DataP was mounted, each displayed first training case was matched to its original NIfTI. Three interior slices per case were checked after the original OpenCV in-plane resize: image correlations were at least 0.999999999999997, and resized integer label mismatch fractions were zero. Installed preview labels agreed on those slices; normalized uint8 images differed by at most one intensity level at sparse pixels, consistent with normalization rounding. This is sampled case correspondence evidence, not a whole-file identity claim. Private case names, source paths, and detailed evidence remain outside Git.

The preprocessing code resizes each NIfTI `[:,:,z]` from 384×384 to 256×256 without transposing it. Moving the last HDF5 axis to the front therefore maps NIfTI axes **2/0/1** to display **Z/Y/X**. In-plane preview stride 2 then produces 128×128. For this display layout:

```text
preview Z spacing = NIfTI spacing[2] × preview stride[0]
preview Y spacing = NIfTI spacing[0] × original shape[0] / HDF5 height × preview stride[1]
preview X spacing = NIfTI spacing[1] × original shape[1] / HDF5 width  × preview stride[2]
```

This axis mapping matters for I2CVB, whose two in-plane spacings differ. Millimeter units were verified in the selected image headers. Physical axis lengths are restored; patient orientation remains unverified and no direction/affine claim is added.

| Display center | Preview Z (mm) | Preview Y (mm) | Preview X (mm) |
| --- | ---: | ---: | ---: |
| BIDMC | 2.200050 | 1.093752 | 1.093740 |
| HK | 3.600000 | 1.562500 | 1.562500 |
| ISBI | 4.000002 | 1.500000 | 1.500000 |
| UCL | 3.300000 | 1.562500 | 1.562500 |
| ISBI-1.5 | 2.999997 | 1.250000 | 1.250000 |
| I2CVB | 1.249968 | 1.602783 | 1.285566 |

These values describe the selected gallery cases only, not all patients at a center. They must not be applied as task-wide evaluation geometry.

## Installed correction

Local and server galleries each received 15 metadata updates: six dense Domain-CL cases, six corresponding weak cases, two UCL Task-CL variants, and the legacy HK full-supervision example. Only `spacing` and `spacing_source` changed in the NPZ payloads. Images, dense labels, scribbles, HDF5 sources, evaluation files, scores, and checkpoints were not edited. Original NPZs are retained in a sibling `showcase-spacing-backup-20260914` private directory on each host. Local provenance was updated and the server has a private correction record.

The server checked shape and three-slice image moments/foreground counts against the corresponding installed local cases before applying the metadata. All 15 updated files were read successfully. Its existing viewer supports `protocol` spacing and refreshes its case cache using file mtime, so the metadata correction required no web-process restart. NFS rejected extended-attribute copying during the first backup; ordinary content copying with owner-only permissions succeeded.

## Source changes and validation

- Preparation accepts `--prostate-spacing Z Y X` for the first training case's HDF5 grid and `--segmentation-spacing PRIVATE_JSON` for explicit per-source first-case spacing. The preview stride is applied afterward. The verified private spacing map is saved with the local correction evidence for regeneration.
- Local/source UI adds an optional Z/Y/X calibration control and warns when only index spacing is available. Manual geometry is marked `manual, unverified`, applies to all four viewports, and never overwrites stored arrays. Disabling calibration restores asset metadata. The online deployment received the metadata correction; its application code remains on the existing release.
- Python: `test_showcase.py` (6 tests) and `test_volume_platform.py` (13 tests) passed. Coverage includes anisotropic HDF5 sampling, bridge geometry, input validation, manual updates, cache immutability, and reset.
- Frontend: 25 tests passed, including anisotropic mesh bounds and manual-source parsing. Typecheck and production build passed.
- Local browser: changing Z to a temporary calibration value rebuilt MPR and the surface; disabling it restored defaults. After installing the true geometry, BIDMC displayed `2.20005/1.09375/1.09374 (mm; protocol)` with three rendered MPR panels and a rendered segmentation surface. No medical screenshots were committed.
- Public HTTPS health returned `ok`. Full online 3D browser acceptance was not completed: the in-app public navigation timed out, and native Chrome interaction became unavailable. Server metadata validation and local visual acceptance are reported separately.

No smoothing or interpolation was added. Correcting spacing restores physical proportions; it does not create additional acquired slices or improve acquisition resolution.
