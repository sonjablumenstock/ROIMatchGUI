# roimatch_gui/utils/alignment_constellation.py
from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree
from skimage.measure import regionprops
from skimage.transform import SimilarityTransform, AffineTransform
from skimage.measure import ransac


# ------------------------- Low-level geometry helpers -------------------------

def _centroids_from_labelmap(lbl: np.ndarray):
    """
    Extract centroid coordinates (x, y), ROI ids, and areas from a labeled ROI map.
    lbl: 2D int array, 0=background, >0 ROI ids
    """
    props = regionprops(lbl)
    cents = np.array([p.centroid[::-1] for p in props], dtype=float)  # (x, y)
    ids = np.array([p.label for p in props], dtype=int)
    areas = np.array([p.area for p in props], dtype=float)
    return cents, ids, areas


def _median_nn(cents: np.ndarray) -> float:
    """Median nearest-neighbor distance (pixels)."""
    if len(cents) < 2:
        return 1.0
    k = min(6, len(cents))
    d, _ = cKDTree(cents).query(cents, k=k)
    nn = d[:, 1]  # nearest non-self neighbor
    nn = nn[np.isfinite(nn)]
    return float(np.median(nn)) if nn.size else 1.0


def _pca_angle(cents: np.ndarray) -> float:
    """
    Orientation angle (radians) of the first principal component of the constellation.
    """
    c = cents - cents.mean(axis=0, keepdims=True)
    if c.shape[0] < 2:
        return 0.0
    _, _, Vt = np.linalg.svd(c, full_matrices=False)
    v = Vt[0]
    return float(np.arctan2(v[1], v[0]))


def _local_fingerprint(cents: np.ndarray, k: int = 5, scale: float | None = None):
    """
    Scale-invariant distance fingerprints for each centroid:
    sorted distances to k nearest neighbors, divided by `scale` (median NN if None).

    Returns
    -------
    fp : (N, k) float array
        Per-point sorted neighbor distances (dimensionless).
    scale : float
        The scale used for normalization (pixels).
    """
    if len(cents) == 0:
        return np.zeros((0, 0), dtype=float), 1.0
    k = min(k, max(1, len(cents) - 1))
    tree = cKDTree(cents)
    dists, _ = tree.query(cents, k=k + 1)  # includes self at col 0
    if dists.shape[1] <= 1:
        return np.zeros((len(cents), 0), dtype=float), 1.0
    if scale is None:
        scale = _median_nn(cents)
    fp = np.sort(dists[:, 1:], axis=1) / max(scale, 1e-6)
    return fp, float(scale)


def _candidate_pairs(
    fp_mov: np.ndarray,
    fp_ref: np.ndarray,
    tol: float = 0.25,
    kquery: int = 2,
    mutual: bool = True,
):
    """
    Candidate centroid correspondences via KD-tree in fingerprint space (L1 metric).
    `tol` operates in normalized units (dimensionless), thanks to _local_fingerprint.
    """
    if fp_mov.size == 0 or fp_ref.size == 0:
        return []

    tr = cKDTree(fp_ref)
    tm = cKDTree(fp_mov)

    # initial (i_mov, j_ref) list
    pairs = []
    for i, v in enumerate(fp_mov):
        d, j = tr.query(v, k=min(kquery, max(1, len(fp_ref))))
        d = np.atleast_1d(d); j = np.atleast_1d(j)
        for di, ji in zip(d, j):
            if np.isfinite(di) and di < tol:
                pairs.append((i, int(ji)))

    # enforce mutuality (optional but helpful for small-N)
    if mutual and pairs:
        keep = []
        for i, j in pairs:
            d2, i2 = tm.query(fp_ref[j], k=min(kquery, max(1, len(fp_mov))))
            d2 = np.atleast_1d(d2); i2 = np.atleast_1d(i2)
            ok = any((int(ii) == i) and np.isfinite(dd) and dd < tol for dd, ii in zip(d2, i2))
            if ok:
                keep.append((i, j))
        pairs = keep

    # deduplicate by keeping best L1 match per i_mov
    if pairs:
        best = {}
        for i, j in pairs:
            di = np.linalg.norm(fp_mov[i] - fp_ref[j], ord=1)
            if (i not in best) or (di < best[i][0]):
                best[i] = (di, j)
        pairs = [(i, j) for i, (_, j) in best.items()]
    return pairs


def _fit_transform_ransac(
    cents_mov: np.ndarray,
    cents_ref: np.ndarray,
    pairs,
    model: str = 'similarity',
    residual_threshold: float = 3.0,
    max_trials: int = 4000,
):
    """
    Robust initial transform with RANSAC.
    """
    if len(pairs) < 3:
        return None, None, np.inf
    src = cents_mov[[i for i, _ in pairs]]
    dst = cents_ref[[j for _, j in pairs]]
    Model = SimilarityTransform if model == 'similarity' else AffineTransform
    model_robust, inliers = ransac(
        (src, dst),
        Model,
        min_samples=(2 if model == 'similarity' else 3),
        residual_threshold=float(residual_threshold),
        max_trials=int(max_trials),
    )
    if model_robust is None or inliers is None or inliers.sum() < 3:
        return None, None, np.inf
    resid = np.median(np.linalg.norm(model_robust(src[inliers]) - dst[inliers], axis=1))
    return model_robust, inliers, float(resid)


def _icp_refine(
    cents_mov: np.ndarray,
    cents_ref: np.ndarray,
    T0=None,
    max_iter: int = 30,
    tol: float = 1e-3,
    cutoff: float = 8.0,
    model: str = 'similarity',
):
    """
    Lightweight point-to-nearest ICP to polish the transform.
    """
    T = (SimilarityTransform() if model == 'similarity' else AffineTransform()) if T0 is None else T0
    tree = cKDTree(cents_ref)
    prev_err = np.inf

    for _ in range(max_iter):
        moved = T(cents_mov)
        dist, idx = tree.query(moved, k=1)
        mask = dist < cutoff
        if mask.sum() < 3:
            break
        src = cents_mov[mask]
        dst = cents_ref[idx[mask]]
        Model = SimilarityTransform if model == 'similarity' else AffineTransform
        T = Model()
        T.estimate(src, dst)
        err = np.median(np.linalg.norm(T(src) - dst, axis=1))
        if abs(prev_err - err) < tol:
            break
        prev_err = err
    return T, float(prev_err)


def _invertible(T, det_eps: float = 1e-8, cond_max: float = 1e8) -> bool:
    """
    Numerical sanity check for transform invertibility.
    """
    P = getattr(T, "params", None)
    if P is None or P.shape != (3, 3) or not np.isfinite(P).all():
        return False
    J = P[:2, :2]
    try:
        d = np.linalg.det(J)
        c = np.linalg.cond(J)
    except Exception:
        return False
    return (abs(d) > det_eps) and (c < cond_max)


# ------------------------- Public entrypoint -------------------------

def align_by_roi_constellation(
    lbl_ref: np.ndarray,
    lbl_mov: np.ndarray,
    k: int = 5,
    fp_tol: float | None = 0.25,
    cutoff: float | None = 8.0,
    model: str = 'similarity',
):
    """
    Align mov -> ref using ROI-centroid constellations (geometry-first).

    Strategy:
      1) Build scale-invariant kNN distance fingerprints for each centroid.
      2) Candidate matches via KD-tree in fingerprint space (mutual, deduped).
      3) RANSAC to bootstrap a Similarity (or Affine) transform.
      4) ICP refinement in pixel space.
      5) Return transform + QC metrics. Reject non-invertible results.

    Parameters
    ----------
    lbl_ref, lbl_mov : 2D int arrays
        Labeled ROI maps (0 background, >0 ROI ids).
    k : int
        Max nearest neighbors per fingerprint (will decrease automatically when ROIs are few).
    fp_tol : float or None
        L1 tolerance in fingerprint space (dimensionless). If None, uses a data-driven value.
    cutoff : float or None
        ICP pairing & QC cutoff in pixels. If None, adapts to 0.4 * median NN of ref.
    model : {'similarity','affine'}
        Preferred model to try first. If it fails, we may retry with the other.

    Returns
    -------
    (Transform, info) on success, or (None, {'reason': ...}) on failure.
    """
    c_ref, _, _ = _centroids_from_labelmap(lbl_ref)
    c_mov, _, _ = _centroids_from_labelmap(lbl_mov)

    # Require at least 3 points to estimate similarity/affine robustly
    if len(c_ref) < 3 or len(c_mov) < 3:
        return None, {'reason': 'too_few_rois', 'n_ref': len(c_ref), 'n_mov': len(c_mov)}

    N = min(len(c_ref), len(c_mov))
    k_eff = max(2, min(k, N - 1))

    # Scale-invariant fingerprints + adaptive pixel thresholds
    fp_ref, s_ref = _local_fingerprint(c_ref, k=k_eff, scale=None)
    fp_mov, s_mov = _local_fingerprint(c_mov, k=k_eff, scale=None)

    medNN_ref = _median_nn(c_ref)
    residual_thr = 0.15 * medNN_ref                  # RANSAC residual (px)
    cutoff_eff   = (0.40 * medNN_ref) if cutoff is None else float(cutoff)

    # Fingerprint tolerance: use provided or estimate from 1-NN fingerprint distances
    if fp_tol is None:
        if fp_ref.size and fp_mov.size:
            dmin, _ = cKDTree(fp_ref).query(fp_mov, k=1)
            med = float(np.median(dmin)) if np.isfinite(dmin).any() else 0.25
            mad = float(np.median(np.abs(dmin - med))) if np.isfinite(dmin).any() else 0.05
            tol_use = float(np.clip(med + 1.5 * (mad + 1e-6), 0.10, 0.60))
        else:
            tol_use = 0.25
    else:
        tol_use = float(fp_tol)

    # Candidate pairs in normalized (dimensionless) space
    pairs = _candidate_pairs(fp_mov, fp_ref, tol=tol_use, kquery=2, mutual=True)

    # Attempt with chosen model first
    def _try(model_name: str):
        T0, inliers, r = _fit_transform_ransac(
            c_mov, c_ref, pairs,
            model=model_name,
            residual_threshold=residual_thr,
            max_trials=6000 if model_name == 'similarity' else 8000,
        )
        if T0 is None:
            return None, None, np.inf
        T, err = _icp_refine(c_mov, c_ref, T0=T0, cutoff=cutoff_eff, model=model_name)
        return T, (inliers.sum() if inliers is not None else 0), float(err)

    T, ninl, icp_err = _try(model_name=model)

    # If failed or too few inliers, retry with the other model
    if (T is None) or (ninl < 3):
        alt_model = 'affine' if model == 'similarity' else 'similarity'
        T, ninl, icp_err = _try(model_name=alt_model)
        model_used = alt_model
    else:
        model_used = model

    # If still nothing reasonable, build a coarse PCA+scale init then ICP (similarity)
    if T is None:
        theta = _pca_angle(c_ref) - _pca_angle(c_mov)
        s0 = s_ref / max(s_mov, 1e-6)
        R0 = SimilarityTransform(scale=s0, rotation=theta)
        mov_rot = R0(c_mov)
        t = c_ref.mean(axis=0) - mov_rot.mean(axis=0)
        T0 = SimilarityTransform(scale=s0, rotation=theta, translation=t)
        T, icp_err = _icp_refine(c_mov, c_ref, T0=T0, cutoff=cutoff_eff, model='similarity')
        model_used = 'similarity_pca_init'

    # QC: centroid proximity & invertibility
    moved = T(c_mov)
    dist, _ = cKDTree(c_ref).query(moved, k=1)
    overlap_score = float((dist < cutoff_eff).mean())

    info = {
        'n_ref': int(len(c_ref)),
        'n_mov': int(len(c_mov)),
        'k_eff': int(k_eff),
        'medNN_ref': float(medNN_ref),
        'fp_scale_ref': float(s_ref),
        'fp_scale_mov': float(s_mov),
        'tol_used': float(tol_use),
        'model': model_used,
        'icp_resid': float(icp_err),
        'overlap_score': float(overlap_score),
    }

    if not _invertible(T):
        return None, {'reason': 'degenerate_transform', **info}

    return T, info


#
# from scipy.spatial import cKDTree
# from skimage.measure import regionprops
# from skimage.transform import SimilarityTransform, AffineTransform
# from skimage.measure import ransac
# import numpy as np
#
# def _centroids_from_labelmap(lbl):
#     props = regionprops(lbl)
#     cents = np.array([p.centroid[::-1] for p in props], dtype=float)  # (x, y)
#     ids = np.array([p.label for p in props], dtype=int)
#     areas = np.array([p.area for p in props], dtype=float)
#     return cents, ids, areas
#
# def _local_fingerprint(cents, k=6):
#     """
#     Per-ROI kNN distance *ratios*, scale-invariant.
#     For each point: sorted distances to k neighbors, divided by the median of those distances.
#     """
#     k = min(k, max(1, len(cents)-1))
#     tree = cKDTree(cents)
#     dists, _ = tree.query(cents, k=k+1)  # includes self
#     D = np.sort(dists[:, 1:], axis=1)    # drop self
#     med = np.median(D, axis=1, keepdims=True)
#     med[med == 0] = 1.0
#     return D / med   # shape (N, k), dimensionless
#
# def _candidate_pairs(fp_mov, fp_ref, tol, kquery=5):
#     """
#     KD-tree in fingerprint space (dimensionless). We keep top-kquery matches
#     below tolerance in L1 distance.
#     """
#     tree = cKDTree(fp_ref)
#     pairs = []
#     for i, v in enumerate(fp_mov):
#         d, j = tree.query(v, k=min(kquery, len(fp_ref)))
#         if np.isscalar(d):
#             d, j = np.array([d]), np.array([j])
#         for di, ji in zip(d, j):
#             if di < tol:
#                 pairs.append((i, ji))
#     return pairs
#
# def _fit_transform_ransac(cents_mov, cents_ref, pairs, model='similarity', resid_thr=3.0, max_trials=4000):
#     if len(pairs) < 3:
#         return None, None, np.inf
#     src = cents_mov[[i for i, _ in pairs]]
#     dst = cents_ref[[j for _, j in pairs]]
#     Model = SimilarityTransform if model == 'similarity' else AffineTransform
#     model_robust, inliers = ransac(
#         (src, dst), Model,
#         min_samples=2 if model == 'similarity' else 3,
#         residual_threshold=resid_thr,
#         max_trials=max_trials,
#     )
#     if model_robust is None or inliers is None or inliers.sum() < 3:
#         return None, None, np.inf
#     resid = np.median(np.linalg.norm(model_robust(src[inliers]) - dst[inliers], axis=1))
#     return model_robust, inliers, float(resid)
#
# def _icp_refine(cents_mov, cents_ref, T0=None, max_iter=25, tol=1e-3, cutoff=10.0, model='similarity'):
#     T = (SimilarityTransform() if model == 'similarity' else AffineTransform()) if T0 is None else T0
#     tree = cKDTree(cents_ref)
#     prev_err = np.inf
#     for _ in range(max_iter):
#         moved = T(cents_mov)
#         dist, idx = tree.query(moved, k=1)
#         mask = dist < cutoff
#         if mask.sum() < 3:
#             break
#         src = cents_mov[mask]
#         dst = cents_ref[idx[mask]]
#         Model = SimilarityTransform if model == 'similarity' else AffineTransform
#         T = Model()
#         T.estimate(src, dst)
#         err = np.median(np.linalg.norm(T(src) - dst, axis=1))
#         if abs(prev_err - err) < tol:
#             break
#         prev_err = err
#     return T, float(prev_err)
#
# def align_by_roi_constellation(lbl_ref, lbl_mov, k=6, fp_tol=None, cutoff=10.0, model='similarity'):
#     """
#     Try Similarity first with adaptive fingerprint tolerance; on failure, retry with Affine.
#     """
#     c_ref, _, _ = _centroids_from_labelmap(lbl_ref)
#     c_mov, _, _ = _centroids_from_labelmap(lbl_mov)
#     if len(c_ref) < 5 or len(c_mov) < 5:
#         return None, {'reason': 'too_few_rois', 'n_ref': len(c_ref), 'n_mov': len(c_mov)}
#
#     fp_ref = _local_fingerprint(c_ref, k=k)
#     fp_mov = _local_fingerprint(c_mov, k=k)
#
#     # Adaptive tolerance: median L1 distance between each mov fp and nearest ref fp
#     tree = cKDTree(fp_ref)
#     dmin, _ = tree.query(fp_mov, k=1)
#     base_tol = np.median(dmin) + 1.5 * (np.median(np.abs(dmin - np.median(dmin))) + 1e-6)
#     tol_use = float(fp_tol) if fp_tol is not None else float(max(0.15, min(0.6, base_tol)))
#
#     # --- Attempt 1: Similarity ---
#     pairs = _candidate_pairs(fp_mov, fp_ref, tol=tol_use, kquery=7)
#     T0, inliers, r = _fit_transform_ransac(c_mov, c_ref, pairs, model='similarity', resid_thr=4.0, max_trials=6000)
#     if T0 is None or (inliers is not None and inliers.sum() < 6):
#         # --- Attempt 2: Affine (more flexible) ---
#         T0, inliers, r = _fit_transform_ransac(c_mov, c_ref, pairs, model='affine', resid_thr=4.0, max_trials=8000)
#         if T0 is None:
#             return None, {'reason': 'ransac_failed'}
#
#         T, err = _icp_refine(c_mov, c_ref, T0=T0, cutoff=cutoff, model='affine')
#         model_used = 'affine'
#     else:
#         T, err = _icp_refine(c_mov, c_ref, T0=T0, cutoff=cutoff, model='similarity')
#         model_used = 'similarity'
#
#     # Overlap proxy for QC
#     moved = T(c_mov)
#     d, _ = cKDTree(c_ref).query(moved, k=1)
#     overlap_score = float((d < cutoff).mean())
#
#     info = {
#         'n_ref': int(len(c_ref)),
#         'n_mov': int(len(c_mov)),
#         'ransac_inliers': int(inliers.sum()) if inliers is not None else 0,
#         'ransac_resid': float(r),
#         'icp_resid': float(err),
#         'overlap_score': float(overlap_score),
#         'model': model_used,
#         'tol_used': tol_use,
#     }
#     return T, info
