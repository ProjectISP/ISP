from isp.Gui import pw
from isp.Gui.Frames.uis_frames import UiSettingsDialog

class SettingsDialog(pw.QDialog, UiSettingsDialog):
    def __init__(self, parent=None):
        super(SettingsDialog, self).__init__(parent)
        self.setupUi(self)

        if parent is not None:
            self.setWindowTitle(parent.windowTitle())
            self.setWindowIcon(parent.windowIcon())

    def getParameters(self):
        param_dict = {}
        for i in range(self.formLayout.rowCount()):
            field = self.formLayout.itemAt(i, pw.QFormLayout.FieldRole)
            label = self.formLayout.itemAt(i, pw.QFormLayout.LabelRole)
            if hasattr(field, 'widget') and hasattr(label, 'widget'): 
                field = field.widget()
                label = label.widget()
                if hasattr(label, 'text') and hasattr(field, 'value'):
                    field = field.value()
                    label = label.text()
                    param_dict[label] = field 

        return param_dict
