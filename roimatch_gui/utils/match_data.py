import numpy as np

# class MatchData:
#     def __init__(self):
#         self.rois = []  # list of Experiment objects
#         self.mapping = []  # optional: final allSessionMapping array
#         self.comparison_matrix = None
#         self.ref_image = None  # first experiment's mean image

class MatchData:
    def __init__(self):
        self.rois = []                      # List of Experiment objects (each contains stat, iscell, etc.)
        self.session_files = []            # Full paths to the loaded experiment folders
        self.transforms = {}               # Dict of (ref_index, session_index) → transform
        self.ref_index = None              # Index of reference session
        self.ref_image = None              # Mean image of reference session

        self.mapping = []                  # Optional: final allSessionMapping array (legacy)
        self.all_session_mapping = []      # Main match result: list of matched ROI groups (by index)

        self.comparison_matrix = None      # Optional: used if you implement matching diagnostics

        self.roiMapRegistered = []         # List of 2D arrays: labeled masks aligned to reference space
        self.meanFrameRegistered = []      # List of 2D arrays: mean images aligned to reference space

class Experiment:
    def __init__(self, path, ops, stat, iscell, mean_image, cell_id_map):
        self.path = path
        self.ops = ops
        self.stat = stat
        self.iscell = iscell
        self.mean_image = mean_image
        self.cell_id_map = cell_id_map

        self.mean_registered = None
        self.roi_map_registered = None
        self.transformation = None
        self.cell_count = int(np.sum(iscell[:, 0] == 1))
        self.committed = np.zeros(self.cell_count, dtype=int)