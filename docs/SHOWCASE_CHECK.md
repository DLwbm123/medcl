# Four-case gallery acceptance · 2026-09-07

The homepage and task center expose classification, fully supervised segmentation,
weakly supervised segmentation, and registration examples without requiring a
submission or developer mode. Existing Cornerstone components are reused.

| Example | Display | Private preview shape |
|---|---|---|
| PathMNIST | Training image and original class | 28 × 28 RGB |
| Prostate | Training volume and complete source segmentation | 24 × 128 × 128 |
| ACDC | Image, original sparse scribble, complete source segmentation | 10 × 128 × 108 |
| OASIS | Real fixed/moving pair, target-state reference, checkerboard and color fusion | 96 × 112 × 80 |

These are reference replays, not trained-model results. The registration reference
is the fixed image; it is not a computed warped moving image or dense deformation
ground truth. No scores, confidence values, evaluation records, or method
comparisons are generated. Browser wording is “示例展示” and “对齐参考”.

Validation: 22 focused Python checks passed across `test_showcase.py`,
`test_homepage_ui.py`, and `test_app.py`. The gallery check covers all four entry
points, slice switching, envelope decoding, sparse/unlabelled pixel handling,
checkerboard pixels, invalid input rejection, and absence of scored jobs.
Browser inspection confirmed all four examples, segmentation overlays, OASIS
color fusion, and Ready state in both segmentation and registration 3D viewers.
Original ACDC NIfTI spacing corrected the initial flat index-space display;
the viewer receives 10 / 3.125 / 3.125 mm after in-plane downsampling. OASIS uses
2 / 2 / 2 mm from its prepared NIfTI header. Prostate spacing remains index-space;
patient orientation is not asserted.

Medical arrays and source provenance stay in the owner-only local state directory.
The public change contains code, preparation instructions and this check summary;
it contains no source paths, medical images, screenshots, labels, or checkpoints.
