from isp.Gui import pqg, pw
from isp.Gui.Frames import BaseFrame
from isp.Gui.Frames.frequencytime import FrequencyTimeFrame
from isp.Gui.Frames.dispersion_maps_frame import EGFDispersion
from isp.Gui.Frames.uis_frames import UiNoise
from isp.Gui.Frames.efg_frame import EGFFrame
from isp.Gui.Frames.parameters import ParametersSettings
from isp.Gui.Frames.setting_dialog_noise import SettingsDialogNoise
from isp.Gui.Frames.project_frame_dispersion import Project

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

        # Dispersion Curves Project

        self.dispersion_project = Project()

        # Create tabs and add them to tabWidget
        # TODO: sharing parameters and settings this way. Should they be shared?
        # Or they are only specific to EGFFrame?
        self.egf_frame = EGFFrame(self.parameters, self.settings_dialog)
        self.ft_frame = FrequencyTimeFrame(self.dispersion_project)
        self.dsp_frame = EGFDispersion()
        self.tabWidget.addTab(self.egf_frame, 'EGFs')
        self.tabWidget.addTab(self.ft_frame, 'Frequency Time Analysis')
        self.tabWidget.addTab(self.dsp_frame, 'Dispersion Maps')

        # Actions
        self.actionSet_Parameters.triggered.connect(self.open_parameters_settings)
        self.actionOpen_Settings.triggered.connect(self.settings_dialog.show)
        self.actionNew_Project.triggered.connect(self.dispersion_project.show)

        # Shortcuts
        self.shortcut_open = pw.QShortcut(pqg.QKeySequence('Ctrl+L'), self)
        self.shortcut_open.activated.connect(self.open_parameters_settings)

    def open_parameters_settings(self):
        self.parameters.show()

