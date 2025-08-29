import sys
import os
import numpy as np
import uuid
import pandas as pd
from pathlib import Path
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QLabel, QVBoxLayout, QWidget, QPushButton,
    QFileDialog, QListWidget, QHBoxLayout, QMessageBox, QLineEdit
)
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from roimatch_gui.utils.loader import load_suite2p_experiment
from roimatch_gui.utils.match_data import MatchData

from skimage.transform import estimate_transform
from skimage.transform import warp
from .point_match_window import PointMatchWindow
from ..utils.mask_utils import create_cell_id_map
from .registration_auto import (
    compute_session_transform,
    compute_session_transform_constellation,
    warp_label_map,
    warp_image,
)
from .matcher_auto import groups_from_all_sessions



# def mean_from_ops(s2p_folder: str):
#     ops_path = Path(s2p_folder) / "ops.npy"
#     ops = np.load(ops_path, allow_pickle=True).item()
#     mean_img = ops["meanImg"]  # minimal: assume it exists
#     # ensure 2D float32
#     mean_img = np.asarray(mean_img, dtype=np.float32)
#     assert mean_img.ndim == 2, "ops['meanImg'] must be 2D"
#     return mean_img

def launch_gui():
    app = QApplication(sys.argv)
    window = ROIApp()
    window.show()
    sys.exit(app.exec_())


class ROIApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.status_ref_label = QLabel()
        self.status_sessions_label = QLabel()
        self.status_alignment_label = QLabel()
        self.match_data = MatchData()
        self.setWindowTitle("ROIMatch GUI")
        self.setGeometry(100, 100, 1200, 800)

        self.init_ui()

    # noinspection PyUnresolvedReferences
    def init_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout()

        # --- Create Buttons ---
        load_state_button = QPushButton("Load previous Analysis")
        load_state_button.clicked.connect(self.load_session_state)

        load_button = QPushButton("Add Experiment")
        load_button.clicked.connect(self.load_data)

        ref_button = QPushButton("Set as Reference")
        ref_button.clicked.connect(self.set_as_reference)

        align_button = QPushButton("Align to Reference")
        align_button.clicked.connect(self.align_to_reference)

        auto_match_button = QPushButton("Auto-Match All Sessions")
        auto_match_button.clicked.connect(self.auto_match_rois)
        auto_match_button.clicked.connect(self.plot_matched_roi_outlines)

        full_auto_button = QPushButton("Full Auto (No Manual Alignment)")
        full_auto_button.clicked.connect(self.on_full_auto_clicked)

        # --- New button: Align by ROI Constellation ---
        btn_align_const = QPushButton("Align by ROI Constellation")
        btn_align_const.setToolTip("Estimate transform from ROI centroids (robust to mean-image contrast)")
        btn_align_const.clicked.connect(self.align_selected_session_by_constellation)

        qc_button = QPushButton("Full Auto QC")
        qc_button.clicked.connect(self.show_full_auto_qc)

        save_button = QPushButton("Save Matches")
        save_button.clicked.connect(lambda: self.save_uuid_matches())

        show_matches_button = QPushButton("Show Matched ROI Outlines")
        show_matches_button.clicked.connect(self.plot_matched_roi_outlines)

        match_button = QPushButton("Manual Matching / Debug mode")
        match_button.clicked.connect(self.match_selected_experiments)

        remove_button = QPushButton("Remove Selected Session")
        remove_button.clicked.connect(self.remove_selected_session)

        self.overlap_thresh_label = QLabel("Overlap threshold (e.g. 0.5):")
        self.overlap_thresh_input = QLineEdit("0.2")

        # 🔴 Reset button (styled red)
        reset_button = QPushButton("Reset / Clear All")
        reset_button.setStyleSheet("QPushButton { background-color: #e74c3c; color: white; font-weight: bold; }")
        reset_button.clicked.connect(self.reset_all)

        # --- Workflow Buttons Row ---
        workflow_btn_layout = QHBoxLayout()
        workflow_btn_layout.addWidget(load_state_button)
        workflow_btn_layout.addWidget(load_button)
        workflow_btn_layout.addWidget(ref_button)
        workflow_btn_layout.addWidget(align_button)
        workflow_btn_layout.addWidget(auto_match_button)
        workflow_btn_layout.addWidget(full_auto_button)
        workflow_btn_layout.addWidget(btn_align_const)
        workflow_btn_layout.addWidget(save_button)
        workflow_btn_layout.addWidget(show_matches_button)
        layout.addLayout(workflow_btn_layout)

        # --- Status Panel ---
        status_layout = QVBoxLayout()
        status_layout.addWidget(QLabel("<b>Status:</b>"))
        status_layout.addWidget(self.status_ref_label)
        status_layout.addWidget(self.status_sessions_label)
        status_layout.addWidget(self.status_alignment_label)
        layout.addLayout(status_layout)

        # --- Secondary Row ---
        secondary_layout = QHBoxLayout()
        secondary_layout.addWidget(remove_button)
        secondary_layout.addWidget(match_button)
        secondary_layout.addWidget(qc_button)
        secondary_layout.addWidget(self.overlap_thresh_label)
        secondary_layout.addWidget(self.overlap_thresh_input)
        layout.addLayout(secondary_layout)

        # 🔻 Place reset button in its own row below the others
        layout.addWidget(reset_button)

        # --- Session List and Plot ---
        self.session_list = QListWidget()
        layout.addWidget(QLabel("Loaded Experiments (full paths):"))
        layout.addWidget(self.session_list)

        self.canvas = FigureCanvas(Figure(figsize=(6, 6)))
        self.ax = self.canvas.figure.add_subplot(111)
        layout.addWidget(self.canvas)

        central_widget.setLayout(layout)

    def shorten(self, path):
        parts = os.path.normpath(path).split(os.sep)
        return "_".join(parts[-5:-2])

    def progress_start(self, total):
        # e.g., QProgressDialog or status text with a counter
        self._prog_total = total;
        self._prog = 0

    def progress_step(self, msg=""):
        self._prog += 1
        self.statusBar().showMessage(f"[{self._prog}/{self._prog_total}] {msg}")


    def update_session_list_display(self):
        self.session_list.clear()
        session_files = getattr(self.match_data, 'session_files', [])
        for i, path in enumerate(session_files):
            label = f"{'reference' if i == getattr(self.match_data, 'ref_index', -1) else f'session {i}'}: {self.shorten(path)}"
            self.session_list.addItem(label)

    def update_status_labels(self):
        # Reference
        if hasattr(self.match_data, 'ref_index'):
            ref_path = self.match_data.rois[self.match_data.ref_index].path
            # Inside update_status_labels
            self.status_ref_label.setText(f"✅ Reference set: {self.shorten(ref_path)}")

        else:
            self.status_ref_label.setText("❌ Reference not set")

        # Sessions
        n_sessions = len(self.match_data.rois)
        self.status_sessions_label.setText(f"📁 Sessions loaded: {n_sessions}")

        # Alignment
        ref_idx = getattr(self.match_data, 'ref_index', None)
        aligned = []
        missing = []

        if ref_idx is not None:
            for i in range(n_sessions):
                if i == ref_idx:
                    continue
                if (ref_idx, i) in getattr(self.match_data, 'transforms', {}):
                    aligned.append(i)
                else:
                    missing.append(i)

            self.status_alignment_label.setText(
                f"📐 Aligned: {len(aligned)} of {n_sessions - 1} sessions\n"
                f"{'🟢 All aligned!' if not missing else f'❗ Missing: {missing}'}"
            )
        else:
            self.status_alignment_label.setText("📐 Alignment status: n/a (no reference set)")

    def load_session_state(self):
        import pickle
        from PyQt5.QtWidgets import QFileDialog, QMessageBox

        filename, _ = QFileDialog.getOpenFileName(
            self,
            "Load GUI State",
            "",
            "Pickle Files (*.pkl);;All Files (*)"
        )
        if not filename:
            return

        try:
            with open(filename, "rb") as f:
                state = pickle.load(f)
        except Exception as e:
            QMessageBox.critical(self, "Load Error", f"Could not load file:\n{str(e)}")
            return

        # Load experiments
        self.match_data = MatchData()
        session_files = state.get("session_files", [])
        for folder in session_files:
            try:
                exp = load_suite2p_experiment(folder)
                self.match_data.rois.append(exp)
            except Exception as e:
                print(f"❌ Failed to load session: {folder}\n{e}")

        # Restore attributes
        self.match_data.session_files = session_files
        self.match_data.ref_index = state.get("ref_index")
        self.match_data.transforms = state.get("transforms", {})
        self.match_data.roiMapRegistered = state.get("roiMapRegistered", [])
        self.match_data.meanFrameRegistered = state.get("meanFrameRegistered", [])
        self.match_data.all_session_mapping = state.get("all_session_mapping", [])

        # Update session list GUI
        self.session_list.clear()
        for folder in session_files:
            self.session_list.addItem(folder)

        self.update_status_labels()
        QMessageBox.information(self, "Loaded", f"Restored GUI state from:\n{filename}")
        self.update_session_list_display()

    def load_data(self):
        folder = QFileDialog.getExistingDirectory(self, "Select Suite2p Folder", "./data")
        if not folder:
            return
        exp = load_suite2p_experiment(folder)
        self.match_data.rois.append(exp)

        # Ensure session_files exists and append the loaded path
        if not hasattr(self.match_data, 'session_files'):
            self.match_data.session_files = []
        self.match_data.session_files.append(folder)

        self.session_list.addItem(exp.path)

        if len(self.match_data.rois) == 1:
            self.match_data.ref_image = exp.mean_image
            self.match_data.ref_index = 0

        self.update_status_labels()
        self.update_session_list_display()

    def handle_session_reorder(self):
        """
        Called after drag-and-drop reordering. Updates the internal ordering of sessions in match_data.
        """
        new_order = [self.session_list.row(self.session_list.item(i)) for i in range(self.session_list.count())]
        old_order = list(range(len(self.match_data.session_files)))

        # Map displayed labels to index in match_data.session_files
        label_to_index = {}
        for i, path in enumerate(self.match_data.session_files):
            label = f"{'reference' if i == self.match_data.ref_index else f'session {i}'}: {self.shorten(path)}"
            label_to_index[label] = i

        reordered_indices = []
        for i in range(self.session_list.count()):
            label = self.session_list.item(i).text()
            if label in label_to_index:
                reordered_indices.append(label_to_index[label])
            else:
                print(f"⚠️  Could not find index for label: {label}")

        if len(reordered_indices) == len(self.match_data.session_files):
            self.apply_session_reordering(reordered_indices)
        else:
            print("❌ Reordering failed: mismatch in session count.")

    def apply_session_reordering(self, new_order):
        if not new_order or len(new_order) != len(self.match_data.rois):
            QMessageBox.warning(self, "Invalid", "Invalid session order.")
            return

        self.match_data.rois = [self.match_data.rois[i] for i in new_order]
        self.match_data.session_files = [self.match_data.session_files[i] for i in new_order]

        # Recalculate reference index
        old_ref_path = self.match_data.rois[self.match_data.ref_index].path
        try:
            self.match_data.ref_index = next(i for i, r in enumerate(self.match_data.rois) if r.path == old_ref_path)
        except StopIteration:
            self.match_data.ref_index = 0  # fallback

        self.update_session_list_display()
        self.update_status_labels()

    def remove_selected_session(self):
        idx = self.session_list.currentRow()
        if idx == -1:
            QMessageBox.warning(self, "No Selection", "Please select a session to remove.")
            return

        removed_path = self.match_data.session_files[idx]
        del self.match_data.rois[idx]
        del self.match_data.session_files[idx]

        if hasattr(self.match_data, "ref_index"):
            if idx == self.match_data.ref_index:
                self.match_data.ref_index = 0 if self.match_data.rois else None
            elif idx < self.match_data.ref_index:
                self.match_data.ref_index -= 1

        QMessageBox.information(self, "Removed", f"Removed session:\n{removed_path}")
        self.update_session_list_display()
        self.update_status_labels()

    def set_as_reference(self):
        idx = self.session_list.currentRow()
        if idx == -1:
            QMessageBox.warning(self, "No selection", "Please select a session first.")
            return
        ref_exp = self.match_data.rois[idx]
        self.match_data.ref_image = ref_exp.mean_image
        self.match_data.ref_index = idx
        QMessageBox.information(self, "Reference Set", f"Reference set to:{ref_exp.path}")
        self.update_status_labels()
        self.update_session_list_display()

    def align_to_reference(self):
        idx = self.session_list.currentRow()
        if idx == -1 or not hasattr(self.match_data, 'ref_index'):
            QMessageBox.warning(self, "Missing", "Please select a session and set a reference.")
            return
        if idx == self.match_data.ref_index:
            QMessageBox.warning(self, "Invalid", "Cannot align reference to itself.")
            return

        ref_exp = self.match_data.rois[self.match_data.ref_index]
        mov_exp = self.match_data.rois[idx]

        def on_points_selected(fixed_pts, moving_pts, *args):
            if len(fixed_pts) < 3:
                QMessageBox.warning(self, "Too few points", "You need at least 3 points to compute a transform.")
                return

            from skimage.transform import estimate_transform
            tform = estimate_transform('affine', src=moving_pts, dst=fixed_pts)

            if not hasattr(self.match_data, 'transforms'):
                self.match_data.transforms = {}

            self.match_data.transforms[(self.match_data.ref_index, idx)] = tform
            session_label = self.session_list.item(idx).text()
            QMessageBox.information(self, "Transform",
                                    f"Alignment of {session_label} to reference completed and transform stored.")

            #QMessageBox.information(self, "Transform", f"Affine transform estimated and stored for session {idx}.")
            self.update_status_labels()

        self.point_match_window = PointMatchWindow(ref_exp.mean_image, mov_exp.mean_image, callback=on_points_selected)
        self.point_match_window.show()

    # def add_experiment(self):
    #     folder = QFileDialog.getExistingDirectory(self, "Select Suite2p Folder", "./data")
    #     if not folder:
    #         return
    #     exp = load_suite2p_experiment(folder)
    #     self.match_data.rois.append(exp)
    #
    #     # Add to list view
    #     self.exp_list.addItem(folder)
    #
    #     # Set reference image if first one
    #     if len(self.match_data.rois) == 1:
    #         self.match_data.ref_image = exp.mean_image

    def match_selected_experiments(self):
        if len(self.match_data.rois) < 2:
            QMessageBox.warning(self, "Error", "Need at least two experiments loaded.")
            return

        fixed_exp = self.match_data.rois[0]
        moving_exp = self.match_data.rois[1]

        def on_points_selected(fixed_pts, moving_pts, *args):
            if len(fixed_pts) < 3:
                QMessageBox.warning(self, "Too few points", "At least 3 points required.")
                return

            tform = estimate_transform('affine', src=moving_pts, dst=fixed_pts)
            if not hasattr(self.match_data, 'transforms'):
                self.match_data.transforms = {}
            self.match_data.transforms[(0, 1)] = tform

            QMessageBox.information(self, "Success", "Transform computed and stored.")

        self.point_match_window = PointMatchWindow(fixed_exp.mean_image, moving_exp.mean_image, callback=on_points_selected)
        self.point_match_window.show()

    def auto_match_rois(self):
        try:
            overlap_thresh = float(self.overlap_thresh_input.text())
        except ValueError:
            QMessageBox.warning(self, "Invalid Input", "Please enter a numeric overlap threshold.")
            return

        all_rois = self.match_data.rois
        n_sessions = len(all_rois)
        if n_sessions < 2:
            QMessageBox.warning(self, "Not enough sessions", "Need at least 2 loaded sessions.")
            return

        if not hasattr(self.match_data, 'ref_index'):
            QMessageBox.warning(self, "Reference Missing", "Please set a reference session before auto-matching.")
            return

        ref_idx = self.match_data.ref_index

        if not hasattr(self.match_data, 'transforms') or not self.match_data.transforms:
            QMessageBox.warning(
                self,
                "Missing Transforms",
                "No transforms are stored yet. Please align all sessions to the reference before auto-matching."
            )
            return

        missing_transforms = []
        for i in range(n_sessions):
            if i == ref_idx:
                continue
            if (ref_idx, i) not in self.match_data.transforms:
                missing_transforms.append(i)

        if missing_transforms:
            session_names = [self.session_list.item(i).text() for i in missing_transforms]
            msg = "\n".join(session_names)
            QMessageBox.warning(
                self,
                "Missing Transforms",
                f"Please align the following sessions to the reference before auto-matching:\n{msg}"
            )
            return

        self.run_automatic_matching(self.match_data, overlap_threshold=overlap_thresh)
        QMessageBox.information(self, "Auto-Matching Complete",
                                f"Matched {len(self.match_data.all_session_mapping)} ROI groups.")

    def run_automatic_matching(self, match_data, overlap_threshold=0.2):
        """
        Automatically match ROIs across all loaded sessions using warped ROI masks and overlap criteria.
        Mimics the original MATLAB logic with added safety and debug output.
        """
        print("➡️  Starting run_automatic_matching")

        n_sessions = len(match_data.rois)
        if n_sessions < 2:
            print("❌ Need at least 2 sessions loaded.")
            return

        if not hasattr(match_data, 'ref_index'):
            print("❌ Reference index not set.")
            return

        ref_idx = match_data.ref_index
        ref_shape = match_data.rois[ref_idx].mean_image.shape
        print(f"📐 Reference index: {ref_idx}, image shape: {ref_shape}")

        #warped_masks = []
        match_data.roiMapRegistered= []
        match_data.meanFrameRegistered = []

        # Generate warped masks

        for i, exp in enumerate(match_data.rois):
            print(f"🔄 Generating mask for session {i}...")

            roi_mask = create_cell_id_map(exp.stat, exp.iscell, shape=ref_shape)

            if i == ref_idx:
                warped = roi_mask
                warped_mean = exp.mean_image
            else:
                key = (ref_idx, i)
                if key in match_data.transforms:
                    tform = match_data.transforms[key]
                    warped = warp(roi_mask, inverse_map=tform.inverse, order=0,
                                  preserve_range=True).astype(int)
                    warped_mean = warp(exp.mean_image, inverse_map=tform.inverse,
                                       preserve_range=True).astype(np.float32)
                else:
                    print(f"⚠️  Missing transform for session {i} (key: {key}). Skipping.")
                    warped = np.zeros_like(roi_mask)
                    warped_mean = np.zeros_like(exp.mean_image)

            print(f"✅ Mask ready for session {i}: shape={warped.shape}, max_id={warped.max()}")
            match_data.roiMapRegistered.append(warped)
            match_data.meanFrameRegistered.append(warped_mean)

        # Assign after loop
        warped_masks = match_data.roiMapRegistered

        # Matching across all sessions
        all_matches = []
        committed = [set() for _ in range(n_sessions)]

        for i in range(n_sessions):
            unique_ids_i = np.unique(warped_masks[i])
            unique_ids_i = unique_ids_i[unique_ids_i != 0]  # skip background

            for roi_id in unique_ids_i:
                if roi_id in committed[i]:
                    continue

                match_group = [None] * n_sessions
                match_group[i] = roi_id

                pix_i = set(zip(*np.where(warped_masks[i] == roi_id)))

                for j in range(n_sessions):
                    if j == i:
                        continue

                    candidates = np.unique(warped_masks[j])
                    candidates = candidates[candidates != 0]
                    best_overlap = 0
                    best_match = None

                    for roi_j in candidates:
                        if roi_j in committed[j]:
                            continue
                        pix_j = set(zip(*np.where(warped_masks[j] == roi_j)))
                        intersection = pix_i & pix_j
                        union_len = max(len(pix_i), len(pix_j))
                        if union_len == 0:
                            continue
                        overlap = len(intersection) / union_len

                        if overlap > best_overlap:
                            best_overlap = overlap
                            best_match = roi_j

                    if best_overlap >= overlap_threshold:
                        match_group[j] = best_match

                if sum(1 for x in match_group if x is not None) >= 2:
                    for j, roi in enumerate(match_group):
                        if roi is not None:
                            committed[j].add(roi)
                    all_matches.append(match_group)

        match_data.all_session_mapping = all_matches
        print(f"✅ Finished matching: {len(all_matches)} ROI groups matched.")

    def choose_template_session(self, sessions):
        """Pick the highest-SNR session as template (std of mean_image)."""
        best = None
        best_std = -1.0
        for s in sessions:
            img = getattr(s, "mean_image", None)
            if img is None:
                continue
            val = float(np.std(img))
            if val > best_std:
                best_std, best = val, s
        return best or sessions[0]

    def on_full_auto_clicked(self):
        self.run_full_auto_match()
        # if we have groups, draw immediately
        if getattr(self.match_data, "matched_groups", None) or getattr(self.match_data, "all_session_mapping", None):
            self.plot_matched_roi_outlines()

    def run_full_auto_match(self):
        """
        Full automatic matching with no manual alignment.
        - Computes session->template transforms
        - Warps mean images and ROI label maps into template space
        - Runs multi-session grouping
        - Saves CSV/PKL with UUIDs
        """
        try:
            # --- Sessions source of truth ---
            sessions = getattr(self.match_data, "rois", [])
            if len(sessions) < 2:
                self.statusBar().showMessage("Load ≥2 sessions for full-auto.")
                return

            # --- Parse IoU threshold from UI safely ---
            try:
                iou_min = float(self.overlap_thresh_input.text())
            except Exception:
                iou_min = 0.25  # sane default

            # --- Ensure each session has a session_id we can use as a stable key ---
            #     Prefer a short label from path; fall back to index.
            session_ids = []
            for idx, s in enumerate(sessions):
                if not hasattr(s, "session_id") or not getattr(s, "session_id"):
                    label = getattr(s, "path", None)
                    s.session_id = self.shorten(label) if label else f"session_{idx}"
                session_ids.append(s.session_id)

            # # --- Choose template (by SNR) ---
            # template_sess = self.choose_template_session(sessions)
            # self.match_data.template_index = sessions.index(template_sess)  # <-- add this line
            # template = getattr(template_sess, "mean_image", None)
            # if template is None:
            #     QMessageBox.critical(self, "Error", "Template session has no mean_image.")
            #     return
            # H, W = template.shape

            # --- Choose template session ---
            ref_idx = getattr(self.match_data, "ref_index", None)
            if ref_idx is not None and 0 <= ref_idx < len(sessions):
                template_sess = sessions[ref_idx]  # <-- prefer user's reference
            else:
                template_sess = self.choose_template_session(sessions)  # fallback: highest SNR

            template = getattr(template_sess, "mean_image", None)
            if template is None:
                QMessageBox.critical(self, "Error", "Template session has no mean_image.")
                return

            # store for plotting/QC
            self.match_data.template_index = sessions.index(template_sess)
            H, W = template.shape

            # --- Work buffers ---
            from roimatch_gui.utils.mask_utils import create_cell_id_map
            labelmaps_reg = {}
            transforms = {}

            # Progress (N sessions + 1 grouping step)
            self.progress_start(total=len(sessions) + 1)

            # --- Register each session to the template and expose registered maps ---
            for s in sessions:
                mean_img = getattr(s, "mean_image", None)
                if mean_img is None:
                    QMessageBox.critical(self, "Error", "A session is missing mean_image; cannot proceed.")
                    return

                # Build label map in the session’s native space
                try:
                    roi_mask_native = create_cell_id_map(s.stat, s.iscell, shape=mean_img.shape)
                except Exception as e:
                    QMessageBox.critical(self, "Error", f"Failed to build label map for {s.session_id}:\n{e}")
                    return


                # Compute transform session->template
                T = compute_session_transform(mean_img, template, need_nonrigid=False)
                transforms[s.session_id] = T

                # Expose registered maps for QC and plotting
                s.meanFrameRegistered = warp_image(mean_img, T, out_shape=(H, W), order=1)
                s.roiMapRegistered = warp_label_map(roi_mask_native, T, out_shape=(H, W))

                # Use registered label map for matching
                labelmaps_reg[s.session_id] = s.roiMapRegistered

                self.progress_step(msg=f"Registered {s.session_id}")

            # --- Multi-session grouping ---
            params = dict(iou_min=iou_min, dmax_px=8.0, area_ratio=(0.5, 2.0))
            groups = groups_from_all_sessions(labelmaps_reg, params=params)
            self.progress_step(msg="Built groups")

            # --- Build CSV/PKL with UUID per group; columns are session_ids in GUI order ---
            rows = []
            for g in groups:  # g is a dict: {session_id: roi_label, ...}
                row = {"uuid": str(uuid.uuid4())}
                for sid in session_ids:
                    row[sid] = int(g[sid]) if sid in g else -1
                rows.append(row)
            df = pd.DataFrame(rows, columns=["uuid"] + session_ids)

            # --- Choose a safe output directory ---
            if getattr(self.match_data, "output_dir", None):
                base_dir = Path(self.match_data.output_dir)
            else:
                # fall back to parent of first session path
                first_path = getattr(sessions[0], "path", None)
                base_dir = Path(first_path).parent if first_path else Path.cwd()
            out_dir = base_dir / "roimatch_full_auto"
            out_dir.mkdir(parents=True, exist_ok=True)

            csv_path = out_dir / "matches_full_auto.csv"
            pkl_path = out_dir / "matches_full_auto.pkl"
            df.to_csv(csv_path, index=False)
            df.to_pickle(pkl_path)

            # --- Update GUI state ---
            self.match_data.matched_groups = groups  # list[dict(session_id->roi)]
            self.match_data.transforms = transforms  # dict(session_id->transform)
            self.match_data.roiMapRegistered = [s.roiMapRegistered for s in sessions]
            self.match_data.meanFrameRegistered = [s.meanFrameRegistered for s in sessions]

            self.statusBar().showMessage(f"Full-auto matching done → {csv_path.name}")

        except Exception as e:
            import traceback
            traceback.print_exc()
            QMessageBox.critical(self, "Full Auto Error", f"{e}")


    # def align_selected_session_by_constellation(self):
    #     from skimage.transform import warp as _skwarp
    #     """
    #     Align ALL loaded sessions to the current reference using ROI-constellation alignment,
    #     store transforms and registered maps, then run auto-match across sessions.
    #     """
    #     # Preconditions
    #     if not getattr(self.match_data, "rois", None) or len(self.match_data.rois) < 2:
    #         QMessageBox.warning(self, "Need Sessions", "Load at least two sessions first.")
    #         return
    #     ref_idx = getattr(self.match_data, "ref_index", None)
    #     if ref_idx is None or ref_idx < 0:
    #         QMessageBox.warning(self, "No Reference", "Set a reference session first.")
    #         return
    #
    #     try:
    #         sessions = self.match_data.rois
    #         n_sessions = len(sessions)
    #         ref_sess = sessions[ref_idx]
    #         ref_shape = getattr(ref_sess, "mean_image", None).shape
    #
    #         # Containers in the expected format
    #         self.match_data.transforms = getattr(self.match_data, "transforms", {})
    #         self.match_data.roiMapRegistered = [None] * n_sessions
    #         self.match_data.meanFrameRegistered = [None] * n_sessions
    #
    #         # Reference goes in as-is
    #         self.match_data.roiMapRegistered[ref_idx] = create_cell_id_map(
    #             ref_sess.stat, ref_sess.iscell, shape=ref_shape
    #         )
    #         self.match_data.meanFrameRegistered[ref_idx] = ref_sess.mean_image
    #
    #         # Align every non-ref session using constellation
    #         failures = []
    #         for i, mov_sess in enumerate(sessions):
    #             if i == ref_idx:
    #                 continue
    #
    #             T, info = compute_session_transform_constellation(ref_sess, mov_sess)
    #             if T is None:
    #                 failures.append((i, (info or {}).get("reason", "unknown")))
    #                 continue
    #
    #             # Persist transform under (ref_idx, i) (this is what the rest of the app expects)
    #             self.match_data.transforms[(ref_idx, i)] = T
    #
    #             # Build native label map, then warp into reference space
    #             lbl_mov_native = create_cell_id_map(
    #                 mov_sess.stat, mov_sess.iscell, shape=mov_sess.mean_image.shape
    #             )
    #             # lbl_mov_reg = warp_label_map(lbl_mov_native, T, out_shape=ref_shape)
    #             # self.match_data.roiMapRegistered[i] = lbl_mov_reg
    #             #
    #             # # Registered mean (for plotting)
    #             # try:
    #             #     mean_reg = warp_image(mov_sess.mean_image, T, out_shape=ref_shape, order=1)
    #             #     self.match_data.meanFrameRegistered[i] = mean_reg
    #             # except Exception:
    #             #     pass
    #
    #             lbl_mov_reg = _skwarp(
    #                 lbl_mov_native,
    #                 inverse_map=T.inverse,
    #                 output_shape=ref_shape,
    #                 order=0,
    #                 preserve_range=True,
    #             ).astype(np.int32)
    #             self.match_data.roiMapRegistered[i] = lbl_mov_reg
    #
    #             # Mean image (linear)
    #             try:
    #                 mean_reg = _skwarp(
    #                     mov_sess.mean_image,
    #                     inverse_map=T.inverse,
    #                     output_shape=ref_shape,
    #                     order=1,
    #                     preserve_range=True,
    #                 ).astype(np.float32)
    #                 self.match_data.meanFrameRegistered[i] = mean_reg
    #             except Exception:
    #                 pass
    #
    #         # UI / status
    #         if failures:
    #             msg = "\n".join([f"session {idx}: {reason}" for idx, reason in failures])
    #             QMessageBox.warning(self, "Some sessions failed to align (constellation)", msg)
    #
    #         # Kick off auto-match across all registered maps
    #         try:
    #             overlap_thresh = float(self.overlap_thresh_input.text())
    #         except ValueError:
    #             overlap_thresh = 0.2
    #
    #         self.run_automatic_matching(self.match_data, overlap_threshold=overlap_thresh)
    #
    #         # Feedback
    #         n_groups = len(getattr(self.match_data, "all_session_mapping", []) or [])
    #         self.status_alignment_label.setText(
    #             f"Constellation aligned {n_sessions - 1 - len(failures)}/{n_sessions - 1} sessions · "
    #             f"auto-matched groups: {n_groups}"
    #         )
    #         QMessageBox.information(
    #             self, "Done",
    #             f"Aligned (constellation) and auto-matched.\n"
    #             f"Aligned: {n_sessions - 1 - len(failures)}/{n_sessions - 1}\n"
    #             f"Groups: {n_groups}"
    #         )
    #
    #         # Update displays/status
    #         self.update_status_labels()
    #         if hasattr(self, "refresh_registered_preview"):
    #             self.refresh_registered_preview()
    #
    #     except Exception as e:
    #         QMessageBox.critical(self, "Error during Constellation Alignment", str(e))

    def align_selected_session_by_constellation(self):
        """
        Batch-align ALL non-reference sessions to the reference using ROI-constellation alignment.
        Does NOT run auto-matching. Use 'Auto-Match All Sessions' afterwards.
        """
        # Preconditions
        if not getattr(self.match_data, "rois", None) or len(self.match_data.rois) < 2:
            QMessageBox.warning(self, "Need Sessions", "Load at least two sessions first.")
            return
        ref_idx = getattr(self.match_data, "ref_index", None)
        if ref_idx is None or ref_idx < 0:
            QMessageBox.warning(self, "No Reference", "Set a reference session first.")
            return

        try:
            sessions = self.match_data.rois
            n_sessions = len(sessions)
            ref_sess = sessions[ref_idx]
            ref_shape = getattr(ref_sess, "mean_image", None).shape

            # Reset/ensure containers
            self.match_data.transforms = getattr(self.match_data, "transforms", {})

            # Optionally prep registered maps for preview (not required for auto-match later)
            self.match_data.roiMapRegistered = [None] * n_sessions
            self.match_data.meanFrameRegistered = [None] * n_sessions
            self.match_data.roiMapRegistered[ref_idx] = create_cell_id_map(
                ref_sess.stat, ref_sess.iscell, shape=ref_shape
            )
            self.match_data.meanFrameRegistered[ref_idx] = ref_sess.mean_image

            successes, failures = [], []

            # Constellation-align every non-ref session
            from skimage.transform import warp as _skwarp
            for i, mov_sess in enumerate(sessions):
                if i == ref_idx:
                    continue

                T, info = compute_session_transform_constellation(ref_sess, mov_sess)
                if T is None:
                    failures.append((i, (info or {}).get("reason", "unknown")))
                    continue

                # Persist transform under (ref_idx, i) so status panel recognizes it
                self.match_data.transforms[(ref_idx, i)] = T
                successes.append((i, (info or {}).get("model", "similarity")))

                # (Optional) build registered maps for visualization
                try:
                    lbl_mov_native = create_cell_id_map(
                        mov_sess.stat, mov_sess.iscell, shape=mov_sess.mean_image.shape
                    )
                    lbl_mov_reg = _skwarp(
                        lbl_mov_native, inverse_map=T.inverse, output_shape=ref_shape,
                        order=0, preserve_range=True
                    ).astype(np.int32)
                    self.match_data.roiMapRegistered[i] = lbl_mov_reg

                    mean_reg = _skwarp(
                        mov_sess.mean_image, inverse_map=T.inverse, output_shape=ref_shape,
                        order=1, preserve_range=True
                    ).astype(np.float32)
                    self.match_data.meanFrameRegistered[i] = mean_reg
                except Exception:
                    # Visualization is best-effort; alignment success still counts
                    pass

            # Update status panel and show a summary
            self.update_status_labels()
            aligned_n = len(successes)
            missing_n = (n_sessions - 1) - aligned_n
            self.status_alignment_label.setText(
                f"Constellation aligned {aligned_n}/{n_sessions - 1} sessions · "
                f"Missing: {[i for i, _ in failures] if failures else 'None'}"
            )

            if failures:
                msg = "\n".join([f"session {idx}: {reason}" for idx, reason in failures])
                QMessageBox.warning(self, "Constellation alignment—manual needed for:",
                                    msg)
            else:
                QMessageBox.information(self, "Constellation alignment", "All sessions aligned.")

            # (No auto-match here — run 'Auto-Match All Sessions' afterwards manually)

        except Exception as e:
            QMessageBox.critical(self, "Error during Constellation Alignment", str(e))


    def save_uuid_matches(self):
        import pickle
        import uuid as _uuid
        import csv
        import os

        # Decide which match structure to use
        groups = None
        if getattr(self.match_data, "all_session_mapping", None):
            groups = self.match_data.all_session_mapping  # list[list or None]
            structure = "legacy"
        elif getattr(self.match_data, "matched_groups", None):
            # Convert dicts -> list aligned to current session order
            sessions = getattr(self.match_data, "rois", [])
            session_ids = [getattr(s, "session_id", f"session_{i}") for i, s in enumerate(sessions)]
            groups = []
            for g in self.match_data.matched_groups:
                groups.append([g.get(sid, None) for sid in session_ids])
            structure = "full_auto"
        else:
            QMessageBox.warning(self, "Missing Matches",
                                "No matched ROIs found. Please run Auto-Match or Full Auto first.")
            return

        session_paths = getattr(self.match_data, "session_files", [])
        if not session_paths:
            # try to rebuild from ROI objects if needed
            sessions = getattr(self.match_data, "rois", [])
            session_paths = [getattr(s, "path", f"session_{i}") for i, s in enumerate(sessions)]

        # --- Determine common parent directory for save dialog ---
        try:
            common_prefix = os.path.commonpath(session_paths)
            default_save_dir = common_prefix
        except ValueError:
            default_save_dir = "/"

        # --- Ask user where to save ---
        filename, _ = QFileDialog.getSaveFileName(
            self,
            "Save Matches",
            os.path.join(default_save_dir, "roi_matches_uuid"),
            "CSV Files (*.csv);;All Files (*)"
        )
        if not filename:
            return  # User cancelled

        base_path = os.path.splitext(filename)[0]

        # --- Generate shortened session labels ---
        session_labels = [self.shorten(p) for p in session_paths]

        # --- Save CSV ---
        try:
            csv_path = base_path + ".csv"
            with open(csv_path, "w", newline="") as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(["uuid"] + session_labels)
                for row in groups:
                    writer.writerow([str(_uuid.uuid4())] + [r if r is not None else 'None' for r in row])
            QMessageBox.information(self, "Saved", f"Saved matches to:\n{csv_path}")
        except Exception as e:
            QMessageBox.critical(self, "Save Error", f"Could not save matches:\n{str(e)}")

        # --- Also save GUI state snapshot (optional & best-effort) ---
        try:
            state_path = f"{base_path}_state.pkl"
            save_data = {
                "session_files": getattr(self.match_data, "session_files", []),
                "ref_index": getattr(self.match_data, "ref_index", None),
                "transforms": getattr(self.match_data, "transforms", {}),
                "roiMapRegistered": getattr(self.match_data, "roiMapRegistered", []),
                "meanFrameRegistered": getattr(self.match_data, "meanFrameRegistered", []),
                # Keep original structures to avoid ambiguity on load
                "all_session_mapping": getattr(self.match_data, "all_session_mapping", []),
                "matched_groups": getattr(self.match_data, "matched_groups", []),
            }
            with open(state_path, "wb") as f:
                pickle.dump(save_data, f)
            print(f"✅ Saved GUI state to {state_path}")
        except Exception as e:
            QMessageBox.warning(self, "Partial Save", f"Matches saved, but failed to save GUI state:\n{str(e)}")

    def show_mean_image(self):
        ops = self.data["ops"]
        stat = self.data["stat"]
        iscell = self.data["iscell"]

        print("iscell shape:", iscell.shape)
        print("iscell dtype:", iscell.dtype)
        print("iscell contents:\n", iscell[:10])

        mean_img = ops.get("meanImg", None)
        if mean_img is None:
            self.ax.clear()
            self.ax.text(0.5, 0.5, "No meanImg found", ha='center', va='center')
            self.canvas.draw()
            return

        self.ax.clear()

        # Contrast enhancement
        vmin = np.percentile(mean_img, 2)
        vmax = np.percentile(mean_img, 98)
        self.ax.imshow(mean_img, cmap='gray', vmin=vmin, vmax=vmax)

        # ✅ Get indices where iscell == 1
        iscell = np.asarray(iscell)  # ensure ndarray
        good_cell_indices = np.where(iscell[:, 0] == 1)[0]

        for idx in good_cell_indices:
            if idx >= len(stat):  # just in case
                continue
            roi = stat[idx]
            xpix = roi['xpix']
            ypix = roi['ypix']

            self.ax.plot(xpix, ypix, '.', markersize=1.5, color='lime', alpha=0.5)

            if 'med' in roi:
                y, x = roi['med']
                self.ax.plot(x, y, 'o', markersize=2, color='red', alpha=0.6)

        self.ax.set_title("Mean Image + ROIs (iscell only)")
        self.ax.axis('off')
        self.canvas.draw()


    def plot_matched_roi_outlines(self):
        """
        Plot matched ROI outlines in the correct common space.

        Background is chosen as:
          - Full Auto:   self.match_data.template_index (template session)
          - Manual flow: self.match_data.ref_index (user-chosen reference)
          - Fallback:    0
        Works with either:
          - legacy self.match_data.all_session_mapping (list of lists)
          - new     self.match_data.matched_groups     (list of dicts: session_id -> roi label)
        """
        # Need registered maps & means
        if not getattr(self.match_data, "roiMapRegistered", None) or not getattr(self.match_data, "meanFrameRegistered",
                                                                                 None):
            QMessageBox.warning(self, "Missing Data",
                                "No registered ROI maps found. Run Auto-Match or Full Auto first.")
            return

        sessions = getattr(self.match_data, "rois", [])
        if not sessions:
            QMessageBox.warning(self, "Missing Data", "No sessions loaded.")
            return

        # --- Choose background index: prefer template_index (Full Auto), else ref_index (manual), else 0
        bg_idx = getattr(self.match_data, "template_index", None)
        if bg_idx is None:
            bg_idx = getattr(self.match_data, "ref_index", 0)
        bg_idx = 0 if bg_idx is None else int(bg_idx)

        # --- Get background image (registered mean)
        try:
            mean_img = self.match_data.meanFrameRegistered[bg_idx]
        except Exception:
            QMessageBox.warning(self, "Missing Data",
                                "Background mean image not found. Run Auto-Match or Full Auto first.")
            return

        # --- Normalize match groups to list-of-indices aligned to current session order
        session_ids = [getattr(s, "session_id", f"session_{i}") for i, s in enumerate(sessions)]
        if getattr(self.match_data, "all_session_mapping", None):
            groups_for_plot = self.match_data.all_session_mapping  # legacy list[list or None]
        elif getattr(self.match_data, "matched_groups", None):
            groups_for_plot = []
            for g in self.match_data.matched_groups:  # dict(session_id -> roi_label)
                groups_for_plot.append([g.get(sid, None) for sid in session_ids])
        else:
            QMessageBox.warning(self, "Missing Data", "No matches found. Run Auto-Match or Full Auto first.")
            return

        # --- Draw
        self.ax.clear()
        vmin, vmax = np.percentile(mean_img, [2, 98])
        self.ax.imshow(mean_img, cmap='gray', vmin=vmin, vmax=vmax)

        from matplotlib.cm import get_cmap
        from skimage.measure import find_contours
        cmap = get_cmap('tab10')

        for group in groups_for_plot:  # each group aligns to session order
            color = cmap(np.random.randint(0, 10))
            for sess_idx, roi_idx in enumerate(group):
                if roi_idx is None or roi_idx == -1:
                    continue
                label_map = self.match_data.roiMapRegistered[sess_idx]
                if label_map is None:
                    continue
                mask = (label_map == roi_idx).astype(np.uint8)
                if mask.max() == 0:
                    continue
                for contour in find_contours(mask, level=0.5):
                    self.ax.plot(contour[:, 1], contour[:, 0], color=color, linewidth=1)

        self.ax.set_title("Matched ROI Outlines")
        self.ax.axis('off')
        self.canvas.draw()




    def show_full_auto_qc(self):
        """
        Compute IoU of each matched ROI vs the reference session and show a histogram.
        Works with either all_session_mapping or matched_groups.
        """
        # Need registered maps
        if not hasattr(self.match_data, 'roiMapRegistered') or not self.match_data.roiMapRegistered:
            QMessageBox.warning(self, "Missing Data", "No registered ROI maps found. Run Full Auto first.")
            return

        sessions = getattr(self.match_data, "rois", [])
        if not sessions:
            QMessageBox.warning(self, "Missing Data", "No sessions loaded.")
            return

        ref_idx = getattr(self.match_data, 'ref_index', 0)
        ref_map = self.match_data.roiMapRegistered[ref_idx]

        # Normalize groups
        session_ids = [getattr(s, "session_id", f"session_{i}") for i, s in enumerate(sessions)]
        if getattr(self.match_data, "all_session_mapping", None):
            groups_for_plot = self.match_data.all_session_mapping
        elif getattr(self.match_data, "matched_groups", None):
            groups_for_plot = []
            for g in self.match_data.matched_groups:
                groups_for_plot.append([g.get(sid, None) for sid in session_ids])
        else:
            QMessageBox.warning(self, "Missing Data", "No matches found. Run Auto-Match or Full Auto first.")
            return

        def iou(mask_a, mask_b):
            inter = np.logical_and(mask_a, mask_b).sum()
            union = np.logical_or(mask_a, mask_b).sum()
            return float(inter) / float(union) if union > 0 else 0.0

        # Collect IoUs against reference
        ious = []
        for group in groups_for_plot:
            ref_roi = group[ref_idx]
            if ref_roi is None or ref_roi == -1:
                continue
            ref_mask = (ref_map == ref_roi)
            for sess_idx, roi_idx in enumerate(group):
                if sess_idx == ref_idx or roi_idx in (None, -1):
                    continue
                m = (self.match_data.roiMapRegistered[sess_idx] == roi_idx)
                ious.append(iou(ref_mask, m))

        # Plot histogram in the main canvas
        self.ax.clear()
        if len(ious) == 0:
            self.ax.text(0.5, 0.5, "No IoUs to plot (no groups overlap with reference).",
                         ha='center', va='center')
        else:
            self.ax.hist(ious, bins=20)
            self.ax.set_xlabel("IoU vs Reference")
            self.ax.set_ylabel("# of matches")
            self.ax.set_title(f"Full Auto QC: IoU to reference (n={len(ious)})\n"
                              f"median={np.median(ious):.2f}, mean={np.mean(ious):.2f}")
        self.canvas.draw()

    def reset_all(self):
        reply = QMessageBox.question(
            self, "Confirm Reset",
            "Are you sure you want to clear all loaded sessions and reset the workspace?",
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No
        )
        if reply == QMessageBox.Yes:
            self.match_data = MatchData()  # Reset everything
            self.session_list.clear()
            self.ax.clear()
            self.canvas.draw()

            # 🔄 Clear status labels
            self.status_ref_label.setText("")
            self.status_sessions_label.setText("")
            self.status_alignment_label.setText("")

            #QMessageBox.information(self, "Reset Complete", "All sessions and matches have been cleared.")






def main():
    launch_gui()