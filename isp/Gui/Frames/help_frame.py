from isp.Gui import pw, pyc
from isp.Gui.Frames.uis_frames import UiHelp




class HelpDoc(pw.QFrame, UiHelp):

    def __init__(self):
        super(HelpDoc, self).__init__()
        self.setupUi(self)
        path = "/Users/robertocabieces/Documents/ISPshare/isp/ISP_Documentation/Responsive HTML5/index.htm"
        url = pyc.QUrl.fromLocalFile(path)
        self.widget.load(url)

