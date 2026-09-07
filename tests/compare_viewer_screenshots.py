"""Compare lossless, actual Cornerstone canvas exports from the synthetic UI."""
from pathlib import Path
import json
import numpy as np
from PIL import Image


def compare(directory: Path) -> dict:
    original = np.asarray(Image.open(directory / 'pixels-original-3d.png'))
    original_mpr = np.asarray(Image.open(directory / 'pixels-original-mpr.png'))
    results = {}
    for mode in ('zero', 'contrast'):
        changed = np.asarray(Image.open(directory / f'pixels-{mode}-3d.png'))
        changed_mpr = np.asarray(Image.open(directory / f'pixels-{mode}-mpr.png'))
        assert changed.shape == original.shape
        assert changed_mpr.shape == original_mpr.shape
        results[mode] = {
            '3d_pixels': int(original.shape[0] * original.shape[1]),
            '3d_differing_pixels': int(np.any(original != changed, axis=2).sum()),
            'max_3d_channel_delta': int(np.abs(original.astype(int) - changed.astype(int)).max()),
            'mpr_differing_pixels': int(np.any(original_mpr != changed_mpr, axis=2).sum()),
        }
        assert results[mode]['3d_differing_pixels'] == 0, results[mode]
        assert results[mode]['mpr_differing_pixels'] > 0, results[mode]
    return results


if __name__ == '__main__':
    directory = Path(__file__).resolve().parents[1] / 'docs' / 'segmentation-3d-evidence'
    result = json.dumps(compare(directory), indent=2) + '\n'
    (directory / 'image-independence.json').write_text(result)
    print(result)
