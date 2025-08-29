# import numpy as np
# from scipy.spatial import cKDTree
# from skimage.measure import regionprops
# from skimage.transform import SimilarityTransform, AffineTransform, warp
# from skimage.measure import ransac
#
# def _centroids_from_labelmap(lbl):
#     # lbl: 2D int array, 0=background, >0 ROI ids
#     props = regionprops(lbl)
#     cents = np.array([p.centroid[::-1] for p in props], dtype=float)  # (x, y)
#     ids = np.array([p.label for p in props], dtype=int)
#     areas = np.array([p.area for p in props], dtype=float)
#     return cents, ids, areas
#
# def _local_fingerprint(cents, k=5):
#     # distances to k nearest neighbors (sorted) for each point
#     tree = cKDTree(cents)
#     dists, _ = tree.query(cents, k=k+1)  # includes self
#     return np.sort(dists[:, 1:], axis=1)  # drop self
#
# def _candidate_pairs(fp_mov, fp_ref, tol=3.0, kquery=3):
#     tree = cKDTree(fp_ref)
#     pairs = []
#     for i, v in enumerate(fp_mov):
#         d, j = tree.query(v, k=kquery)
#         if np.isscalar(d):
#             d, j = np.array([d]), np.array([j])
#         for di, ji in zip(d, j):
#             if di < tol:
#                 pairs.append((i, ji))
#     return pairs
#
# def _fit_transform_ransac(cents_mov, cents_ref, pairs, model='similarity'):
#     if len(pairs) < 3:
#         return None, None, np.inf
#     src = cents_mov[[i for i, _ in pairs]]
#     dst = cents_ref[[j for _, j in pairs]]
#     Model = SimilarityTransform if model == 'similarity' else AffineTransform
#     model_robust, inliers = ransac(
#         (src, dst),
#         Model,
#         min_samples=2 if model == 'similarity' else 3,
#         residual_threshold=3.0,
#         max_trials=2000,
#     )
#     if model_robust is None or inliers is None or inliers.sum() < 3:
#         return None, None, np.inf
#     resid = np.median(np.linalg.norm(model_robust(src[inliers]) - dst[inliers], axis=1))
#     return model_robust, inliers, float(resid)
#
# def _icp_refine(cents_mov, cents_ref, T0=None, max_iter=20, tol=1e-3, cutoff=8.0, model='similarity'):
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
# def align_by_roi_constellation(lbl_ref, lbl_mov, k=5, fp_tol=3.0, cutoff=8.0, model='similarity'):
#     """
#     Returns (TransformObject, info_dict) or (None, {'reason': ...})
#     info_dict contains: n_ref, n_mov, ransac_inliers, ransac_resid, icp_resid, overlap_score
#     """
#     c_ref, id_ref, a_ref = _centroids_from_labelmap(lbl_ref)
#     c_mov, id_mov, a_mov = _centroids_from_labelmap(lbl_mov)
#     if len(c_ref) < 5 or len(c_mov) < 5:
#         return None, {'reason': 'too_few_rois', 'n_ref': len(c_ref), 'n_mov': len(c_mov)}
#
#     fp_ref = _local_fingerprint(c_ref, k=k)
#     fp_mov = _local_fingerprint(c_mov, k=k)
#
#     pairs = _candidate_pairs(fp_mov, fp_ref, tol=fp_tol)
#     T0, inliers, r = _fit_transform_ransac(c_mov, c_ref, pairs, model=model)
#     if T0 is None:
#         return None, {'reason': 'ransac_failed'}
#
#     T, err = _icp_refine(c_mov, c_ref, T0=T0, cutoff=cutoff, model=model)
#
#     moved = T(c_mov)
#     tree = cKDTree(c_ref)
#     dist, _ = tree.query(moved, k=1)
#     overlap_score = float((dist < cutoff).mean())
#
#     info = {
#         'n_ref': int(len(c_ref)),
#         'n_mov': int(len(c_mov)),
#         'ransac_inliers': int(inliers.sum()) if inliers is not None else 0,
#         'ransac_resid': float(r),
#         'icp_resid': float(err),
#         'overlap_score': float(overlap_score),
#     }
#     return T, info


from scipy.spatial import cKDTree
from skimage.measure import regionprops
from skimage.transform import SimilarityTransform, AffineTransform
from skimage.measure import ransac
import numpy as np

def _centroids_from_labelmap(lbl):
    props = regionprops(lbl)
    cents = np.array([p.centroid[::-1] for p in props], dtype=float)  # (x, y)
    ids = np.array([p.label for p in props], dtype=int)
    areas = np.array([p.area for p in props], dtype=float)
    return cents, ids, areas

def _local_fingerprint(cents, k=6):
    """
    Per-ROI kNN distance *ratios*, scale-invariant.
    For each point: sorted distances to k neighbors, divided by the median of those distances.
    """
    k = min(k, max(1, len(cents)-1))
    tree = cKDTree(cents)
    dists, _ = tree.query(cents, k=k+1)  # includes self
    D = np.sort(dists[:, 1:], axis=1)    # drop self
    med = np.median(D, axis=1, keepdims=True)
    med[med == 0] = 1.0
    return D / med   # shape (N, k), dimensionless

def _candidate_pairs(fp_mov, fp_ref, tol, kquery=5):
    """
    KD-tree in fingerprint space (dimensionless). We keep top-kquery matches
    below tolerance in L1 distance.
    """
    tree = cKDTree(fp_ref)
    pairs = []
    for i, v in enumerate(fp_mov):
        d, j = tree.query(v, k=min(kquery, len(fp_ref)))
        if np.isscalar(d):
            d, j = np.array([d]), np.array([j])
        for di, ji in zip(d, j):
            if di < tol:
                pairs.append((i, ji))
    return pairs

def _fit_transform_ransac(cents_mov, cents_ref, pairs, model='similarity', resid_thr=3.0, max_trials=4000):
    if len(pairs) < 3:
        return None, None, np.inf
    src = cents_mov[[i for i, _ in pairs]]
    dst = cents_ref[[j for _, j in pairs]]
    Model = SimilarityTransform if model == 'similarity' else AffineTransform
    model_robust, inliers = ransac(
        (src, dst), Model,
        min_samples=2 if model == 'similarity' else 3,
        residual_threshold=resid_thr,
        max_trials=max_trials,
    )
    if model_robust is None or inliers is None or inliers.sum() < 3:
        return None, None, np.inf
    resid = np.median(np.linalg.norm(model_robust(src[inliers]) - dst[inliers], axis=1))
    return model_robust, inliers, float(resid)

def _icp_refine(cents_mov, cents_ref, T0=None, max_iter=25, tol=1e-3, cutoff=10.0, model='similarity'):
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

def align_by_roi_constellation(lbl_ref, lbl_mov, k=6, fp_tol=None, cutoff=10.0, model='similarity'):
    """
    Try Similarity first with adaptive fingerprint tolerance; on failure, retry with Affine.
    """
    c_ref, _, _ = _centroids_from_labelmap(lbl_ref)
    c_mov, _, _ = _centroids_from_labelmap(lbl_mov)
    if len(c_ref) < 5 or len(c_mov) < 5:
        return None, {'reason': 'too_few_rois', 'n_ref': len(c_ref), 'n_mov': len(c_mov)}

    fp_ref = _local_fingerprint(c_ref, k=k)
    fp_mov = _local_fingerprint(c_mov, k=k)

    # Adaptive tolerance: median L1 distance between each mov fp and nearest ref fp
    tree = cKDTree(fp_ref)
    dmin, _ = tree.query(fp_mov, k=1)
    base_tol = np.median(dmin) + 1.5 * (np.median(np.abs(dmin - np.median(dmin))) + 1e-6)
    tol_use = float(fp_tol) if fp_tol is not None else float(max(0.15, min(0.6, base_tol)))

    # --- Attempt 1: Similarity ---
    pairs = _candidate_pairs(fp_mov, fp_ref, tol=tol_use, kquery=7)
    T0, inliers, r = _fit_transform_ransac(c_mov, c_ref, pairs, model='similarity', resid_thr=4.0, max_trials=6000)
    if T0 is None or (inliers is not None and inliers.sum() < 6):
        # --- Attempt 2: Affine (more flexible) ---
        T0, inliers, r = _fit_transform_ransac(c_mov, c_ref, pairs, model='affine', resid_thr=4.0, max_trials=8000)
        if T0 is None:
            return None, {'reason': 'ransac_failed'}

        T, err = _icp_refine(c_mov, c_ref, T0=T0, cutoff=cutoff, model='affine')
        model_used = 'affine'
    else:
        T, err = _icp_refine(c_mov, c_ref, T0=T0, cutoff=cutoff, model='similarity')
        model_used = 'similarity'

    # Overlap proxy for QC
    moved = T(c_mov)
    d, _ = cKDTree(c_ref).query(moved, k=1)
    overlap_score = float((d < cutoff).mean())

    info = {
        'n_ref': int(len(c_ref)),
        'n_mov': int(len(c_mov)),
        'ransac_inliers': int(inliers.sum()) if inliers is not None else 0,
        'ransac_resid': float(r),
        'icp_resid': float(err),
        'overlap_score': float(overlap_score),
        'model': model_used,
        'tol_used': tol_use,
    }
    return T, info
