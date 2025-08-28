# roimatch_gui/overlap_fast.py
import numpy as np

def _pairs_from_bboxes(labelsA, labelsB, centroidsA, centroidsB, dmax_px=8):
    """Quick candidate pairs: bbox/centroid filter."""
    # Use centroid distance threshold only (fast & robust)
    from scipy.spatial import cKDTree
    tA = cKDTree(centroidsA)
    pairs = []
    for j, cj in enumerate(centroidsB):
        idx = tA.query_ball_point(cj, r=dmax_px)
        for i in idx:
            pairs.append((i, j))
    return np.array(pairs, dtype=np.int32)

def centroids_from_labelmap(L):
    ids = np.unique(L)
    ids = ids[ids > 0]
    ys, xs = np.nonzero(L)
    vals = L[ys, xs]
    sums = {}
    for k, y, x in zip(vals.tolist(), ys.tolist(), xs.tolist()):
        if k not in sums: sums[k] = [0, 0, 0]
        sums[k][0] += y; sums[k][1] += x; sums[k][2] += 1
    centroids = np.zeros((ids.size, 2), dtype=float)
    for idx, k in enumerate(ids):
        y, x, n = sums[int(k)]
        centroids[idx] = (y / n, x / n)
    return ids, centroids

def iou_sparse(LA, LB, pairs, idsA=None, idsB=None):
    """Compute IoU for selected label pairs; returns (pairs, iou, areaA, areaB)."""
    if idsA is None:
        idsA = np.unique(LA); idsA = idsA[idsA > 0]
    if idsB is None:
        idsB = np.unique(LB); idsB = idsB[idsB > 0]
    maxA = idsA.max() if idsA.size else 0
    maxB = idsB.max() if idsB.size else 0

    # flatten -> linear hist trick
    H, W = LA.shape
    flat = LA.astype(np.int64) * (maxB + 1) + LB.astype(np.int64)
    flat = flat.ravel()
    flat = flat[flat > 0]  # drop background combos
    combos, inter = np.unique(flat, return_counts=True)
    a_ids = combos // (maxB + 1)
    b_ids = combos % (maxB + 1)

    # map label → compact index
    mapA = {int(k): i for i, k in enumerate(idsA.tolist())}
    mapB = {int(k): i for i, k in enumerate(idsB.tolist())}
    areaA = np.bincount([mapA[int(k)] for k in LA[LA > 0].ravel()], minlength=idsA.size)
    areaB = np.bincount([mapB[int(k)] for k in LB[LB > 0].ravel()], minlength=idsB.size)

    # build a dict for quick lookup of intersections
    inter_dict = {(int(a), int(b)): int(c) for a, b, c in zip(a_ids, b_ids, inter)}

    iou = np.zeros(pairs.shape[0], dtype=np.float32)
    for idx, (ia, ib) in enumerate(pairs):
        a_lab = int(idsA[ia]); b_lab = int(idsB[ib])
        inter_ab = inter_dict.get((a_lab, b_lab), 0)
        if inter_ab == 0:
            continue
        ua = areaA[ia] + areaB[ib] - inter_ab
        if ua > 0:
            iou[idx] = inter_ab / ua
    return iou
