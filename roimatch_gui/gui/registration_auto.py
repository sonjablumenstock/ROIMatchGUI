# roimatch_gui/registration_auto.py
from __future__ import annotations
import numpy as np
import cv2
from skimage.registration import phase_cross_correlation, optical_flow_tvl1
from scipy.ndimage import map_coordinates, gaussian_filter
from ..utils.alignment_constellation import align_by_roi_constellation
from ..utils.mask_utils import create_cell_id_map

# ---------- helpers (module scope) ----------
def compute_session_transform_constellation(ref_session, mov_session, model='similarity',
                                            k=5, fp_tol=3.0, cutoff=8.0):
    """
    Compute geometric transform from mov_session -> ref_session using ROI constellations.
    Returns (TransformObject, info_dict) or (None, {'reason': ...})
    """
    # Use each session's native mean image size for its label map
    ref_shape = getattr(ref_session, "mean_image", None).shape
    mov_shape = getattr(mov_session, "mean_image", None).shape

    lbl_ref = create_cell_id_map(ref_session.stat, ref_session.iscell, shape=ref_shape)
    lbl_mov = create_cell_id_map(mov_session.stat, mov_session.iscell, shape=mov_shape)

    T, info = align_by_roi_constellation(lbl_ref, lbl_mov, k=k, fp_tol=fp_tol, cutoff=cutoff, model=model)
    return T, info


def _as_gray_uint8(img):
    """Return a 2D uint8 grayscale image (0..255) for OpenCV."""
    img = np.asarray(img)
    if img.ndim == 3:
        # collapse color → gray (luminosity weights)
        img = (0.114 * img[..., 0] + 0.587 * img[..., 1] + 0.299 * img[..., 2])
    img = img.astype(np.float32)
    # scale robustly to 0..255
    p2, p98 = np.percentile(img, (2, 98))
    if not np.isfinite(p2) or not np.isfinite(p98) or p98 <= p2:
        img = np.clip(img, 0, 1)
        return (img * 255).astype(np.uint8)
    img = np.clip((img - p2) / (p98 - p2), 0, 1)
    return (img * 255).astype(np.uint8)

def _as_float_gray(img):
    """Return a 2D float32 grayscale image scaled 0..1 (for skimage flow, etc.)."""
    g = _as_gray_uint8(img)
    return (g.astype(np.float32) / 255.0)

def _preprocess(img, do_clahe=True):
    """Returns (float_gray_0to1, uint8_gray)."""
    g8 = _as_gray_uint8(img)
    if do_clahe:
        # CLAHE expects uint8; then convert to float
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        g8 = clahe.apply(g8)
    gf = (g8.astype(np.float32) / 255.0)
    gf = gaussian_filter(gf, 0.8)
    return gf, g8

def _small_rotation_search(src, dst, max_deg=2.0, step=0.5):
    best = (0.0, -np.inf, (0.0, 0.0))
    for ang in np.arange(-max_deg, max_deg + 1e-6, step):
        M = cv2.getRotationMatrix2D((dst.shape[1] / 2, dst.shape[0] / 2), ang, 1.0)
        rot = cv2.warpAffine(src, M, (dst.shape[1], dst.shape[0]), flags=cv2.INTER_LINEAR)
        shift, _, _ = phase_cross_correlation(dst, rot, upsample_factor=10)
        # NCC proxy: negative MSE after shift
        r = -np.mean((np.roll(rot, (-int(round(shift[0])), -int(round(shift[1]))), axis=(0, 1)) - dst) ** 2)
        if r > best[1]:
            best = (np.radians(ang), r, (shift[1], shift[0]))  # (theta, score, (dx,dy))
    return best

def _maybe_affine(src_u8, dst_u8, rigid_ok: bool, ransac_thresh=3.0):
    if rigid_ok:
        return None
    if src_u8.dtype != np.uint8:
        src_u8 = _as_gray_uint8(src_u8)
    if dst_u8.dtype != np.uint8:
        dst_u8 = _as_gray_uint8(dst_u8)

    orb = cv2.ORB_create(nfeatures=5000, scaleFactor=1.2, WTA_K=2)
    kp1, des1 = orb.detectAndCompute(src_u8, None)
    kp2, des2 = orb.detectAndCompute(dst_u8, None)
    if des1 is None or des2 is None or len(kp1) < 8 or len(kp2) < 8:
        return None

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    matches = bf.knnMatch(des1, des2, k=2)
    good = [m for m, n in matches if m.distance < 0.75 * n.distance] if matches else []
    if len(good) < 8:
        return None

    pts1 = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    pts2 = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
    M, inliers = cv2.estimateAffinePartial2D(
        pts1, pts2, method=cv2.RANSAC, ransacReprojThreshold=ransac_thresh
    )
    return M

def _nonrigid(src, dst):
    # TV-L1 optical flow returns (vy, vx)
    flow = optical_flow_tvl1(dst, src)  # align src→dst
    return flow  # (2, H, W)

# ---------- transform container ----------
class ComposeTransform:
    """Holds rigid (shift+rot), affine (2x3), and dense flow (u,v)."""
    def __init__(self, shift=(0.0, 0.0), theta=0.0, affine=None, flow=None, out_shape=None):
        self.shift = np.asarray(shift, dtype=float)
        self.theta = float(theta)
        self.affine = affine  # 2x3
        self.flow = flow      # (2, H, W) in template coords (dy, dx)
        self.out_shape = out_shape

    def warp_image(self, img, order=1):
        H, W = self.out_shape
        yy, xx = np.mgrid[0:H, 0:W]
        # inverse warp grid
        coords = np.stack([yy, xx], axis=0).astype(np.float32)

        # inverse dense flow
        if self.flow is not None:
            coords = coords - self.flow

        # inverse affine
        if self.affine is not None:
            grid = np.stack([coords[1].ravel(), coords[0].ravel()], axis=1)
            inv = cv2.invertAffineTransform(self.affine)
            grid = cv2.transform(grid[None, :, :], inv)[0]
            coords = np.stack(
                [grid[:, 1].reshape(H, W), grid[:, 0].reshape(H, W)], axis=0
            )

        # inverse rigid (rot+shift)
        if self.theta != 0.0 or np.any(self.shift != 0):
            M = cv2.getRotationMatrix2D((W / 2, H / 2), np.degrees(self.theta), 1.0)
            M[:, 2] += self.shift[::-1]  # shift in (x,y)
            invM = cv2.invertAffineTransform(M)
            grid = np.stack([coords[1].ravel(), coords[0].ravel()], axis=1)
            grid = cv2.transform(grid[None, :, :], invM)[0]
            coords = np.stack(
                [grid[:, 1].reshape(H, W), grid[:, 0].reshape(H, W)], axis=0
            )

        return map_coordinates(img, [coords[0], coords[1]], order=order, mode='nearest')

# ---------- public API ----------
def compute_session_transform(mean_frame, template_frame, need_nonrigid=True):
    """Returns ComposeTransform that maps session→template."""
    t = ComposeTransform(out_shape=template_frame.shape)
    s_f, s_u8 = _preprocess(mean_frame)         # float + uint8
    d_f, d_u8 = _preprocess(template_frame)

    # rigid (small rotation + translation) on float images
    theta, score, (dx, dy) = _small_rotation_search(s_f, d_f)
    t.theta = float(theta)
    t.shift = np.array([dx, dy], dtype=float)

    rigid_ok = score > -0.02

    # affine via ORB+RANSAC on uint8 images
    aff = _maybe_affine(s_u8, d_u8, rigid_ok=rigid_ok)
    if aff is not None:
        t.affine = aff
        # bring the float image forward for nonrigid
        s_f = cv2.warpAffine(s_f, aff, (d_f.shape[1], d_f.shape[0]), flags=cv2.INTER_LINEAR)

    # optional nonrigid on float images
    if need_nonrigid:
        flow = _nonrigid(s_f, d_f)
        t.flow = flow.astype(np.float32)

    return t

def warp_label_map(label_map, transform: ComposeTransform, out_shape=None):
    """Nearest-neighbor warp to preserve integer ROI labels."""
    if out_shape is not None:
        transform.out_shape = out_shape
    return transform.warp_image(label_map.astype(np.float32), order=0).astype(np.int32)

def warp_image(image, transform: ComposeTransform, out_shape=None, order=1):
    if out_shape is not None:
        transform.out_shape = out_shape
    return transform.warp_image(image.astype(np.float32), order=order)
