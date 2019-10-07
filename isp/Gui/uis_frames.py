# Add new ui designers here. The *.ui files must be placed inside resources/designer_uis
from isp.Gui.Utils.pyqt_utils import load_ui_designers

# Add the new UiFrame to the imports at Gui.__init__
UiMainFrame = load_ui_designers("MainFrame.ui")
UiSeismogramFrame = load_ui_designers("SeismogramFrame.ui")

