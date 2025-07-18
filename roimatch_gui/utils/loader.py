import os
import numpy as np
from .mask_utils import create_cell_id_map
from .match_data import Experiment

def load_suite2p_experiment(folder):
    ops = np.load(os.path.join(folder, "ops.npy"), allow_pickle=True).item()
    stat = np.load(os.path.join(folder, "stat.npy"), allow_pickle=True)
    iscell = np.load(os.path.join(folder, "iscell.npy"), allow_pickle=True)
    mean_img = ops.get("meanImg", None)

    shape = mean_img.shape if mean_img is not None else (512, 512)
    cell_id_map = create_cell_id_map(stat, iscell, shape)

    return Experiment(
        path=folder,
        ops=ops,
        stat=stat,
        iscell=iscell,
        mean_image=mean_img,
        cell_id_map=cell_id_map
    )
