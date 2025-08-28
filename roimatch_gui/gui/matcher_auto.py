# roimatch_gui/matcher_auto.py
from __future__ import annotations
import numpy as np
from itertools import combinations
from scipy.optimize import linear_sum_assignment
import networkx as nx
from .overlap_fast import centroids_from_labelmap, _pairs_from_bboxes, iou_sparse

DEFAULTS = dict(iou_min=0.25, dmax_px=8.0, area_ratio=(0.5, 2.0))

def _areas_from_ids(L, ids):
    areas = np.bincount(L[L>0].ravel(), minlength=int(ids.max())+1)
    return np.array([areas[int(k)] for k in ids], dtype=np.int32)

def assign_pair(LA, LB, params=DEFAULTS):
    idsA, cA = centroids_from_labelmap(LA)
    idsB, cB = centroids_from_labelmap(LB)
    if idsA.size == 0 or idsB.size == 0:
        return np.empty((0,2), dtype=int), np.empty((0,), dtype=float)

    pairs = _pairs_from_bboxes(idsA, idsB, cA, cB, dmax_px=params['dmax_px'])
    if pairs.size == 0:
        return np.empty((0,2), dtype=int), np.empty((0,), dtype=float)

    iou = iou_sparse(LA, LB, pairs, idsA, idsB)

    # area compat filter
    areaA = _areas_from_ids(LA, idsA); areaB = _areas_from_ids(LB, idsB)
    keep = []
    for k, (ia, ib) in enumerate(pairs):
        if iou[k] < params['iou_min']:
            continue
        ratio = areaA[ia] / max(1, areaB[ib])
        if params['area_ratio'][0] <= ratio <= params['area_ratio'][1]:
            keep.append(k)
    if not keep:
        return np.empty((0,2), dtype=int), np.empty((0,), dtype=float)

    keep = np.array(keep, dtype=int)
    pairs_kept = pairs[keep]
    iou_kept = iou[keep]

    # Hungarian on cost = 1 - IoU
    nA = idsA.size; nB = idsB.size
    cost = np.ones((nA, nB), dtype=np.float32)
    for (ia, ib), v in zip(pairs_kept, iou_kept):
        cost[ia, ib] = 1.0 - v
    row_ind, col_ind = linear_sum_assignment(cost)
    chosen = []
    scores = []
    for ia, ib in zip(row_ind, col_ind):
        if cost[ia, ib] < (1.0 - params['iou_min']):
            chosen.append([ia, ib]); scores.append(1.0 - cost[ia, ib])
    if not chosen:
        return np.empty((0,2), dtype=int), np.empty((0,), dtype=float)

    chosen = np.array(chosen, dtype=int)
    scores = np.array(scores, dtype=float)
    # map compact indices back to label IDs
    return np.stack([idsA[chosen[:,0]], idsB[chosen[:,1]]], axis=1), scores

def groups_from_all_sessions(labelmaps_by_sess: dict[str, np.ndarray], params=DEFAULTS):
    sess_ids = list(labelmaps_by_sess.keys())
    G = nx.Graph()
    for s in sess_ids:
        L = labelmaps_by_sess[s]
        ids = np.unique(L); ids = ids[ids>0]
        for k in ids:
            G.add_node((s, int(k)))

    for sa, sb in combinations(sess_ids, 2):
        LA = labelmaps_by_sess[sa]; LB = labelmaps_by_sess[sb]
        pairs, scores = assign_pair(LA, LB, params=params)
        for (a_lab, b_lab), sc in zip(pairs, scores):
            G.add_edge((sa, int(a_lab)), (sb, int(b_lab)), weight=float(sc))

    # connected components = putative global cells
    comps = list(nx.connected_components(G))
    # enforce ≤1 ROI per session per group (prune lightest edges if violated)
    cleaned = []
    for comp in comps:
        by_sess = {}
        for node in comp:
            by_sess.setdefault(node[0], []).append(node)
        ok = all(len(v) == 1 for v in by_sess.values())
        if ok:
            cleaned.append(comp); continue
        # resolve with max-weight spanning tree then pick ≤1 per session
        H = G.subgraph(comp).copy()
        T = nx.maximum_spanning_tree(H, weight='weight')
        # greedy pack by session
        seen_sess = set(); group = set()
        for u, v, d in sorted(T.edges(data=True), key=lambda e: e[2]['weight'], reverse=True):
            for node in (u, v):
                if node[0] in seen_sess:
                    continue
                group.add(node); seen_sess.add(node[0])
        cleaned.append(group)

    # produce groups: list of dict {session_id: roi_idx}
    out = []
    for comp in cleaned:
        g = {}
        for (s, k) in comp:
            g[s] = int(k)
        out.append(g)
    return out
