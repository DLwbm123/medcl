# Cross-dataset visualization spacing audit

2026-09-22 UI simplification: removed the gallery's spacing/Z-ratio expander,
manual calibration controls, and associated calibration prompt. Both slice
and 3D views continue using the stored asset geometry. The audit below records
the original geometry repair; its manual-control description is historical.

Checked on 2026-09-14 after the prostate geometry repair. Scope: the other 19 volume gallery files, their preparation paths, and shared gallery/result rendering. Classification galleries contain 2D RGB images and have no slice-spacing axis.

## Findings

| Gallery group | Files | Geometry finding |
| --- | ---: | --- |
| MMWHS class-incremental, dense and weak | 6 | Selected source labels match the nominal 1 mm resampling and center crop/pad. Preview `1/2/2` is correct; its source can now be marked verified protocol geometry. |
| Complete seven-label MMWHS | 1 | Independently matched the selected complete case with its 150-slice crop/pad. Preview `1/2/2` is correct. |
| Utah/LGE left atrium, dense and weak | 2 | Matched source labels after resize; retained original Z indices are contiguous. Correct preview spacing: `2.5/3.125/3.125` mm. |
| LiTS liver, dense and weak | 2 | Matched all 93 retained label slices after XY transpose, X flip and uint8 resize. Retained Z indices have stride 2. Correct preview spacing: `1.6/2.8125/2.8125` mm. |
| FeTS brain tumour, dense and weak | 2 | Matched all retained labels after resize. The 29 retained Z indices are contiguous. Correct preview spacing: `1/1.875/1.875` mm. |
| ACDC legacy weak example | 1 | Original header and preview agree: Z/Y/X spacing is `10 / 3.125 / 3.125` mm after in-plane stride 2. |
| OASIS, CTCT, NLST, MRCT plus legacy OASIS | 5 | Prepared NIfTI grids and preview spacing agree. OASIS is `2/2/2`, CTCT `6/3/3`, NLST and MRCT `3/3/3`. CTCT already accounts for preview Z stride 2. These are prepared-grid geometry checks; no new patient-orientation claim is made. |

The initial search in DataP and previously used server dataset locations did not find a verified match. Expanding the search to the other disks on server 44 located the matching cardiac, LGE, FeTS and liver sources. The earlier negative checks against 484 other brain-tumour label volumes and 20 supplementary liver cases were insufficient to conclude that originals were unavailable.

All seven selected source cases were identified through matching reconstructed labels across their retained volumes. Three corresponding image slices were also checked: cardiac, LGE and FeTS correlations were approximately 1.0. Liver image correlations were 0.9783–0.9810; its image intensity preparation is not identical to the located source, but all 93 label slices match exactly after the documented transformation. This is case-correspondence evidence, not a claim of byte-identical source images. Selected source headers report millimeters. Case identifiers and private paths remain outside Git.

The cardiac HDF5 grid follows the preprocessing's nominal 1 mm resampling convention. It must not be given the raw acquisition slice spacing again. LGE and FeTS map NIfTI axes 2/0/1 to display Z/Y/X; liver uses 2/1/0 with an in-plane flip. Resize factors and retained Z stride are applied before preview strides. This restores display proportions; no patient direction/affine is asserted.

Local and server galleries received 13 metadata updates: six files have corrected numeric spacing (dense and weak LGE, liver and brain cases); seven cardiac files retain their numeric spacing and gain verified source metadata. Original NPZ files are retained in a separate backup directory. Only `spacing` and `spacing_source` are changed; the private regeneration map also records the selected HDF5 spacings. These values are case-specific and must not become task-wide evaluation settings. Existing prostate calibration is retained.

## Implemented fixes

- `medcl/showcase.py`: a shared `physical_slice` helper applies the spacing of the two visible axes to the bitmap aspect ratio. All three gallery slice planes, dense/weak overlays, registration checkerboards, and colour fusion use it. Compositing precedes display-only nearest-neighbour resizing; stored images and labels are unchanged.
- The spacing source, missing-geometry warning, and manual calibration controls now apply to both 2D and 3D gallery views. Switching views preserves a temporary calibration; disabling it restores asset metadata.
- `app.py`: segmentation and registration result fallbacks now use preview spacing when displaying static slices after WebGL failure or when the 2D expander is opened.
- `scripts/prepare_showcase.py`: complete cardiac preparation accepts `--cardiac-spacing Z Y X`, measured on the selected HDF5 grid before preview sampling, and records its source. The existing per-source task-gallery mapping already supports the other segmentation datasets.

No new dependency, training, metric change, source-data edit, or task-wide spacing override was introduced.

## Verification

- `test_showcase.py`: 7 tests passed, including anisotropic aspect ratios in all three planes, RGB overlays, source-array immutability, invalid spacing, calibration across view changes, and explicit cardiac geometry.
- `test_app.py`: 5 tests passed.
- `test_volume_platform.py`: 13 tests passed.
- Deployed the updated application and viewer assets to the existing server, retaining the previous release, supervisor configuration, and overwritten installed component files for rollback. Runtime import/syntax checks and an anisotropic display check passed in the server environment. The web process uses a neutral command path. Public HTTPS health returned `ok`.
- Before the source metadata follow-up, browser acceptance used the existing SSH forwarding connection to the deployed server: the MMWHS class T1 page shows the missing-millimeter warning in 2D; its Y slice renders as 128×50 pixels for stored 100×128 data with visible-axis spacing 1:2. The 3D viewer reports index-space geometry, reaches `Ready`, and renders three MPR panels plus a labelled surface. This confirms rendering behaviour, not acquisition geometry for that case. No patient screenshots were published.

## Follow-up geometry verification

Before metadata installation, three source HDF5 preview slices were compared with each installed local case, including the corresponding dense and weak variants. Sampled labels and normalized uint8 images matched exactly for all 13 files. The server gallery matched the same samples before its update, and the written spacing/source fields were read back successfully. Geometry records and source evidence are private; the public report contains no patient images or identifiers.

After installation, the deployed left-atrium 3D viewer reported `2.5/3.125/3.125 (mm; protocol)`, reached `Ready`, and visibly rendered all three MPR panels and the surface. The old missing-spacing warning was absent. No application restart was necessary; gallery loading invalidates its cache when file modification times change.
