import numpy as np


def create_cell_id_map(stat, iscell, shape):
    mask = np.zeros(shape, dtype=np.int32)
    for i, roi in enumerate(stat):
        if iscell[i, 0] != 1:
            continue
        xpix = roi['xpix']
        ypix = roi['ypix']
        # Only set pixels where mask is 0 to avoid overwriting overlapping ROIs
        for x, y in zip(xpix, ypix):
            if mask[y, x] == 0:
                mask[y, x] = i  # 0-based Suite2p stat index
    return mask
