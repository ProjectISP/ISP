import os
from sys import platform
from isp.Gui import pw
from isp.Gui.Frames import MessageDialog
from isp.Gui.Frames.uis_frames import UiLineStations
import nvector as nv
import pandas as pd
from isp.Gui.Utils.pyqt_utils import add_save_load


@add_save_load()
class CreateLineStations(pw.QDialog, UiLineStations):
    def __init__(self, parent=None):
        super(CreateLineStations, self).__init__(parent)
        self.setupUi(self)

        if parent is not None:
            self.setWindowTitle(parent.windowTitle())
            self.setWindowIcon(parent.windowIcon())

        self.df = None
        self.runDesignBtn.clicked.connect(self.generate_geographic_points)
        self.saveDesignBtn.clicked.connect(self.save_design)


    def generate_geographic_points(self):

        """

        Parameters
        ----------
        ref_lat
        ref_lon
        azimuth
        initial_distance
        num_points
        inter_distance

        Returns
        -------
        Latitude;Longitude;Network;Station
        35.8;-3.0;WM;ARNO
        36.9;-4.0;XX;OBS1
        37.2;-5.0;WM;SFS
        """
        try:
            # Initialize the WGS-84 ellipsoid model
            wgs84 = nv.FrameE(name='WGS84')

            # Create a DataFrame to store the latitude and longitude points
            df = pd.DataFrame(columns=['Latitude', 'Longitude'])
            test = self.latDB.value()
            # Define the starting point
            starting_point = wgs84.GeoPoint(latitude=self.latDB.value(), longitude=self.lonDB.value(), degrees=True)

            network = "NW"
            sta = "STA"
            # Calculate and store each point
            for i in range(self.numStations.value()):
                # Distance for the current point
                distance = self.shift_distance.value() + i * self.inter_station.value()

                # Move the point by azimuth and distance from the starting point
                new_point, _azimuthb = starting_point.displace(distance=distance*1e3, azimuth=self.azimuth.value(), degrees=True)

                # Append the coordinates to the DataFrame
                df = df.append({
                    'Latitude': new_point.latitude_deg,
                    'Longitude': new_point.longitude_deg,
                    'Network': network,
                    'Station': sta+str(i)
                }, ignore_index=True)

            self.df = df
            print(df)
            md = MessageDialog(self)
            md.set_info_message("Design done, proceed to save it")

        except:
            md = MessageDialog(self)
            md.set_error_message("Please provide correct values to make the design")


    def save_design(self):

        if isinstance(self.df, pd.DataFrame):
            root_path = os.path.dirname(os.path.abspath(__file__))

            if "darwin" == platform:
                dir_path = pw.QFileDialog.getExistingDirectory(self, 'Select Directory', root_path)
            else:
                dir_path = pw.QFileDialog.getExistingDirectory(self, 'Select Directory', root_path,
                                                               pw.QFileDialog.DontUseNativeDialog)

            name = self.NameLE.text()
            if name == "":
                name = "current_proj.txt"

            file_path = os.path.join(dir_path, name)
            if isinstance(file_path, str):
                self.df.to_csv(file_path, sep=";", index=False)
        else:
            md = MessageDialog(self)
            md.set_error_message("Please first make a designn")







