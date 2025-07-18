import sys
import numpy as np
from PyQt5.QtWidgets import QApplication
from roimatch_gui.gui.point_match_window import PointMatchWindow

if __name__ == "__main__":
    app = QApplication(sys.argv)

    # Load two test images (or random matrices for now)
    img1 = np.random.rand(512, 512)
    img2 = np.random.rand(512, 512)

    def callback(fixed_pts, moving_pts):
        print("FIXED:", fixed_pts)
        print("MOVING:", moving_pts)

    window = PointMatchWindow(img1, img2, callback=callback)
    window.show()
    sys.exit(app.exec_())