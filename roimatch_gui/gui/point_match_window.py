import numpy as np
from PyQt5.QtWidgets import (
    QMainWindow, QPushButton, QVBoxLayout, QHBoxLayout, QWidget, QMessageBox
)
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from roimatch_gui.utils.image_utils import enhance_contrast


class ZoomableImageCanvas(FigureCanvas):
    def __init__(self, image, title, color):
        self.fig = Figure(figsize=(5, 5))
        super().__init__(self.fig)
        self.ax = self.fig.add_subplot(111)
        self.ax.set_title(title)
        self.ax.imshow(image, cmap='gray')
        self.ax.axis('off')

        self.points = []
        self.labels = []
        self.label_annotations = []
        self.undo_stack = []
        self.color = color

        self.mpl_connect("button_press_event", self.on_click)

        self.original_xlim = self.ax.get_xlim()
        self.original_ylim = self.ax.get_ylim()

    def reset_view(self):
        self.ax.set_xlim(self.original_xlim)
        self.ax.set_ylim(self.original_ylim)
        self.draw()

    def zoom(self, factor, center=None):
        if center is None:
            center = (
                0.5 * (self.ax.get_xlim()[0] + self.ax.get_xlim()[1]),
                0.5 * (self.ax.get_ylim()[0] + self.ax.get_ylim()[1])
            )

        cx, cy = center
        xlim = self.ax.get_xlim()
        ylim = self.ax.get_ylim()

        x_range = (xlim[1] - xlim[0]) * factor
        y_range = (ylim[1] - ylim[0]) * factor

        self.ax.set_xlim([cx - x_range / 2, cx + x_range / 2])
        self.ax.set_ylim([cy - y_range / 2, cy + y_range / 2])
        self.draw()

    def pan(self, dx_data, dy_data):
        xlim = self.ax.get_xlim()
        ylim = self.ax.get_ylim()
        self.ax.set_xlim(xlim[0] + dx_data, xlim[1] + dx_data)
        self.ax.set_ylim(ylim[0] + dy_data, ylim[1] + dy_data)
        self.draw()

    def on_click(self, event):
        pass

    def add_point(self, x, y, label=None):
        self.points.append((x, y))
        idx = len(self.points)
        label_text = str(label) if label is not None else str(idx)
        dot = self.ax.plot(x, y, 'o', color=self.color, markersize=6)[0]
        annotation = self.ax.text(x, y, label_text, fontsize=10, color=self.color)
        self.labels.append(dot)
        self.label_annotations.append(annotation)
        self.draw()

    def undo(self):
        if self.points:
            self.labels[-1].remove()
            self.label_annotations[-1].remove()
            self.points.pop()
            self.labels.pop()
            self.label_annotations.pop()
            self.draw()

    def get_points(self):
        return np.array(self.points)

    def get_labels(self):
        return [ann.get_text() for ann in self.label_annotations]


class PointMatchWindow(QMainWindow):
    def __init__(self, fixed_img, moving_img, callback=None):
        super().__init__()
        self.setWindowTitle("Select Control Points")
        self.setGeometry(200, 200, 1200, 800)

        self.fixed_img = enhance_contrast(fixed_img)
        self.moving_img = enhance_contrast(moving_img)
        self.callback = callback

        self.pan_mode_active = False
        self.last_pan_pos = None

        self.init_ui()

    def init_ui(self):
        main_widget = QWidget()
        main_layout = QVBoxLayout()
        img_layout = QHBoxLayout()

        self.fixed_canvas = ZoomableImageCanvas(self.fixed_img, "Reference Image", color='red')
        self.moving_canvas = ZoomableImageCanvas(self.moving_img, "Moving Image", color='green')

        self.fixed_canvas.mpl_connect("button_press_event", self.on_mouse_press)
        self.fixed_canvas.mpl_connect("motion_notify_event", self.on_mouse_move)
        self.fixed_canvas.mpl_connect("button_release_event", self.on_mouse_release)
        self.moving_canvas.mpl_connect("button_press_event", self.on_mouse_press)
        self.moving_canvas.mpl_connect("motion_notify_event", self.on_mouse_move)
        self.moving_canvas.mpl_connect("button_release_event", self.on_mouse_release)

        zoom_in_btn = QPushButton("🔍 Zoom In")
        zoom_out_btn = QPushButton("🔎 Zoom Out")
        reset_btn = QPushButton("🔁 Reset View")
        fullscreen_btn = QPushButton("🖥️ Full Screen")
        pan_btn = QPushButton("✋ Pan")
        undo_btn = QPushButton("↩️ Undo")
        done_btn = QPushButton("✅ Done")
        cancel_btn = QPushButton("❌ Cancel")

        zoom_in_btn.clicked.connect(self.zoom_in)
        zoom_out_btn.clicked.connect(self.zoom_out)
        reset_btn.clicked.connect(self.reset_views)
        fullscreen_btn.setCheckable(True)
        fullscreen_btn.clicked.connect(self.toggle_fullscreen)
        pan_btn.setCheckable(True)
        pan_btn.clicked.connect(self.toggle_pan_mode)
        undo_btn.clicked.connect(self.undo_last_point)
        done_btn.clicked.connect(self.on_done)
        cancel_btn.clicked.connect(self.close)

        controls_layout = QHBoxLayout()
        for btn in [zoom_in_btn, zoom_out_btn, pan_btn, undo_btn, reset_btn, fullscreen_btn, done_btn, cancel_btn]:
            controls_layout.addWidget(btn)

        img_layout.addWidget(self.fixed_canvas)
        img_layout.addWidget(self.moving_canvas)

        main_layout.addLayout(controls_layout)
        main_layout.addLayout(img_layout)
        main_widget.setLayout(main_layout)
        self.setCentralWidget(main_widget)

    def on_mouse_press(self, event):
        if self.pan_mode_active and event.inaxes:
            self.last_pan_pos = (event.x, event.y)
        elif event.inaxes == self.fixed_canvas.ax:
            x, y = event.xdata, event.ydata
            self.fixed_canvas.add_point(x, y)
        elif event.inaxes == self.moving_canvas.ax:
            x, y = event.xdata, event.ydata
            self.moving_canvas.add_point(x, y)

    def on_mouse_move(self, event):
        if self.pan_mode_active and self.last_pan_pos and event.xdata and event.ydata:
            dx = event.x - self.last_pan_pos[0]
            dy = event.y - self.last_pan_pos[1]
            self.last_pan_pos = (event.x, event.y)

            dx_data = -dx / event.inaxes.bbox.width * (event.inaxes.get_xlim()[1] - event.inaxes.get_xlim()[0])
            dy_data = -dy / event.inaxes.bbox.height * (event.inaxes.get_ylim()[1] - event.inaxes.get_ylim()[0])

            self.fixed_canvas.pan(dx_data, dy_data)
            self.moving_canvas.pan(dx_data, dy_data)

    def on_mouse_release(self, event):
        self.last_pan_pos = None

    def toggle_pan_mode(self, checked):
        self.pan_mode_active = checked

    def zoom_in(self):
        center = self.get_center()
        self.fixed_canvas.zoom(0.8, center)
        self.moving_canvas.zoom(0.8, center)

    def zoom_out(self):
        center = self.get_center()
        self.fixed_canvas.zoom(1.25, center)
        self.moving_canvas.zoom(1.25, center)

    def reset_views(self):
        self.fixed_canvas.reset_view()
        self.moving_canvas.reset_view()

    def toggle_fullscreen(self, checked):
        if checked:
            self.showFullScreen()
        else:
            self.showNormal()

    def get_center(self):
        ax = self.fixed_canvas.ax
        return (
            0.5 * (ax.get_xlim()[0] + ax.get_xlim()[1]),
            0.5 * (ax.get_ylim()[0] + ax.get_ylim()[1])
        )

    def undo_last_point(self):
        self.fixed_canvas.undo()
        self.moving_canvas.undo()


    def on_done(self):
        fixed_pts = self.fixed_canvas.get_points()
        moving_pts = self.moving_canvas.get_points()
        fixed_labels = self.fixed_canvas.get_labels()
        moving_labels = self.moving_canvas.get_labels()

        # ✅ Check for equal number of points
        if len(fixed_pts) != len(moving_pts):
            QMessageBox.warning(
                self,
                "Mismatch",
                f"Uneven number of control points!\n"
                f"Left: {len(fixed_pts)}, Right: {len(moving_pts)}.\n"
                f"Please label the same number of points in both images."
            )
            return

        if self.callback:
            self.callback(fixed_pts, moving_pts, fixed_labels, moving_labels)

        self.close()

    def get_points(self):
        return self.fixed_canvas.get_points(), self.moving_canvas.get_points()
