from isp.Gui import pqg, pw
from isp.Gui.Frames import BaseFrame
from isp.Gui.Frames.frequencytime import FrequencyTimeFrame
from isp.Gui.Frames.uis_frames import UiNoise
from isp.Gui.Frames.efg_frame import EGFFrame
from isp.Gui.Frames.parameters import ParametersSettings
from isp.Gui.Frames.help_frame import HelpDoc
from isp.Gui.Frames.setting_dialog_noise import SettingsDialogNoise


class NoiseFrame(BaseFrame, UiNoise):

    def __init__(self):
        super(NoiseFrame, self).__init__()
        self.setupUi(self)

        self.setWindowTitle('Seismic Ambient Noise')
        self.setWindowIcon(pqg.QIcon(':\icons\map-icon.png'))

        # Settings dialog
        self.settings_dialog = SettingsDialogNoise(self)

        # Parameters settings
        self.parameters = ParametersSettings()

        # Create tabs and add them to tabWidget
        # TODO: sharing parameters and settings this way. Should they be shared?
        # Or they are only specific to EGFFrame?
        self.egf_frame = EGFFrame(self.parameters, self.settings_dialog)
        self.ft_frame = FrequencyTimeFrame()
        self.tabWidget.addTab(self.egf_frame, 'EGFs')
        self.tabWidget.addTab(self.ft_frame, 'Frequency Time Analysis')

        # Help Documentation
        self.help = HelpDoc()

        # Actions
        self.actionSet_Parameters.triggered.connect(self.open_parameters_settings)
        self.actionOpen_Settings.triggered.connect(self.settings_dialog.show)

        # Shortcuts
        self.shortcut_open = pw.QShortcut(pqg.QKeySequence('Ctrl+L'), self)
        self.shortcut_open.activated.connect(self.open_parameters_settings)

    def open_parameters_settings(self):
        self.parameters.show()

