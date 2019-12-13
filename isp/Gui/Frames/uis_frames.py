# Add new ui designers here. The *.ui files must be placed inside resources/designer_uis
from isp.Gui.Utils.pyqt_utils import load_ui_designers

# Add the new UiFrame to the imports at Frames.__init__
UiMainFrame = load_ui_designers("MainFrame.ui")
UiTimeFrequencyFrame = load_ui_designers("TimeFrequencyFrame.ui")
UiEarthquakeAnalysisFrame = load_ui_designers("EarthquakeAnalysisFrame.ui")
UiPaginationWidget = load_ui_designers("PaginationWidget.ui")
UiFilterGroupBox = load_ui_designers("FilterGroupBox.ui")
UiEventInfoGroupBox = load_ui_designers("EventInfoGroupBox.ui")



