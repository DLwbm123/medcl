# Cross-dataset visualization spacing audit

Checked on 2026-09-14 after the prostate geometry repair. Scope: the other 19 volume gallery files, their preparation paths, and shared gallery/result rendering. Classification galleries contain 2D RGB images and have no slice-spacing axis.

## Findings

| Gallery group | Files | Geometry finding |
| --- | ---: | --- |
| MMWHS class-incremental, dense and weak | 6 | HDF5 lacks spacing. Located preprocessing requests a nominal 1 mm isotropic grid, so `[1,2,2]` after preview sampling may be correct. Exact selected-case preprocessing provenance remains unverified. |
| Complete seven-label MMWHS | 1 | HDF5 lacks spacing; this case has a different depth from class-incremental preparation. Do not transfer class-task geometry automatically. |
| Utah/LGE left atrium, dense and weak | 2 | In-plane resize and foreground-slice filtering require original dimensions, spacing, and retained slice indices. No corresponding original was verified. |
| LiTS liver, dense and weak | 2 | Preprocessing resizes slices and retains only foreground slices at even original Z indices. Uniform retained indices would require twice the original Z spacing, before preview sampling. The source case and uniformity remain unverified. |
| FeTS brain tumour, dense and weak | 2 | In-plane resize and enhancing-tumour slice selection require checking whether retained Z indices are contiguous. No corresponding original was verified. |
| ACDC legacy weak example | 1 | Original header and preview agree: Z/Y/X spacing is `10 / 3.125 / 3.125` mm after in-plane stride 2. |
| OASIS, CTCT, NLST, MRCT plus legacy OASIS | 5 | Prepared NIfTI grids and preview spacing agree. OASIS is `2/2/2`, CTCT `6/3/3`, NLST and MRCT `3/3/3`. CTCT already accounts for preview Z stride 2. These are prepared-grid geometry checks; no new patient-orientation claim is made. |

Search covered relevant DataP dataset directories and authorized server dataset locations. A compatibility check against 484 available brain-tumour label volumes found no matching selected case under the located preprocessing rules. Twenty supplementary liver cases also provided no verified match; their additional annotations are not equivalent to the original LiTS labels. These negative checks do not prove the originals are absent from every disk. Case identifiers and private paths remain outside Git.

The 13 unverified volume files retain their existing index-space geometry. No guessed millimeter values were written. Existing prostate calibration was retained. Nonuniform slice selection cannot be repaired by entering a single Z spacing: it requires recovering slice positions and constructing a regular display grid.

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
- Browser acceptance used the existing SSH forwarding connection to the deployed server: the MMWHS class T1 page shows the missing-millimeter warning in 2D; its Y slice renders as 128×50 pixels for stored 100×128 data with visible-axis spacing 1:2. The 3D viewer reports index-space geometry, reaches `Ready`, and renders three MPR panels plus a labelled surface. This confirms rendering behaviour, not acquisition geometry for that case. No patient screenshots were published.
