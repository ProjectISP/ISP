import os

from isp import IMAGES_PATH
from isp.Gui import UiMainFrame, pw, pqg, qt


class MainFrame(pw.QMainWindow, UiMainFrame):

    def __init__(self):
        super(MainFrame, self).__init__()

        # Set up the user interface from Designer.
        self.setupUi(self)

        icon1 = pqg.QPixmap(os.path.join(IMAGES_PATH, '02.png'))
        icon2 = pqg.QPixmap(os.path.join(IMAGES_PATH, '03.png'))
        icon3 = pqg.QPixmap(os.path.join(IMAGES_PATH, '04.png'))
        icon4 = pqg.QPixmap(os.path.join(IMAGES_PATH, '05.png'))
        icon5 = pqg.QPixmap(os.path.join(IMAGES_PATH, '01.png'))
        iconLogo = pqg.QPixmap(os.path.join(IMAGES_PATH, 'LOGO.png'))

        self.LOGO.setPixmap(iconLogo)
        self.labelManage.setPixmap(icon1)
        self.labelseismogram.setPixmap(icon2)
        self.labelearthquake.setPixmap(icon3)
        self.labelMTI.setPixmap(icon4)
        self.labelarray.setPixmap(icon5)

        self.SeismogramAnalysis.clicked.connect(self.onClickSeismogramAnalysis)
        # self.ArrayAnalysis.clicked.connect(self.array)

    # Press esc key event
    def keyPressEvent(self, e):
        if e.key() == qt.Key_Escape:
            self.close()

    def onClickSeismogramAnalysis(self):
        pass

