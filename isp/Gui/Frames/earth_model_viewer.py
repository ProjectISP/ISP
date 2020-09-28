from isp.Gui import pw
from isp.Gui.Frames.uis_frames import UiEarth_model_viewer
import numpy as np
import matplotlib.pyplot as plt
from isp.Gui.Utils.pyqt_utils import add_save_load, BindPyqtObject
from isp.earthquakeAnalisysis.NLLGrid import NLLGrid
import os
import cartopy.crs as ccrs
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER


@add_save_load()
class EarthModelViewer(pw.QFrame,UiEarth_model_viewer):
        """
        EarthModelViewr contains the functionability to plot *buf from NonLinLoc

                :param params required to initialize the class:


        """
        def __init__(self):
            super(EarthModelViewer, self).__init__()
            self.setupUi(self)

            self.selectDirBtn.clicked.connect(lambda: self.set_path())
            self.plotBtn.clicked.connect(lambda: self.plot())
            #self.load_modelBtn.clicked.connect(lambda: self.info_model())

        def set_path(self):
            selected_file = pw.QFileDialog.getOpenFileName(self, "Select file *.buf")
            self.path_file = selected_file[0]
            if isinstance(self.path_file, str):
                self.pathLE.setText(self.path_file)
                self.info_grid()


        def info_grid(self):

            grd = NLLGrid(self.path_file)
            self.InfoTX.setPlainText("Model Primary Information")
            self.InfoTX.appendPlainText("Basename: {name} ".format(name=os.path.basename(grd.basename)))
            self.InfoTX.appendPlainText("nx: {nx} ny: {ny} nz: {nz}".format(nx=grd.nx, ny=grd.ny, nz=grd.nz))
            self.InfoTX.appendPlainText("x_orig: {x_orig} y_orig: {y_orig} y_orig: {y_orig}".format(x_orig=grd.x_orig,
                                      y_orig=grd.y_orig,z=grd.z_orig))
            self.InfoTX.appendPlainText("nx: {dx} ny: {dy} nz: {dz}".format(dx=grd.dx, dy=grd.dy, dz=grd.dz))
            self.InfoTX.appendPlainText("transform {transform} LatOrig {LatOrig:.2f} LonOrig {LonOrig:.2f}".format(
                transform = grd.proj_name, LatOrig=grd.orig_lat, LonOrig=grd.orig_lon))


        def plot(self):
            k = np.pi / 180
            file = self.pathLE.text()
            color_map = plt.cm.get_cmap('jet')
            reversed_color_map = color_map.reversed()
            grd = NLLGrid(file)
            dx = grd.dx
            dy = grd.dy
            dz = grd.dz

            x0 = round(grd.orig_lon+(grd.x_orig/(111.111*np.cos(k*grd.orig_lat))))
            x1 = round(grd.orig_lon-(grd.x_orig/(111.111*np.cos(k*grd.orig_lat))))
            y0 = round(grd.orig_lat+(grd.y_orig/111.111))
            y1 = round(grd.orig_lat-(grd.y_orig/111.111))
            xx = round((self.lonDB.value()-x0)*dx*111.111*np.cos(k*grd.orig_lat))
            yy = round((self.latDB.value()-y0)*dy*111.111)
            zz = round(dz * self.depthDB.value())
            print(xx, yy, zz)
            extent = [x0-1, x1+1, y0-1, y1+1]
            print(extent)


            grid = grd.array

            if grd.z_orig * dz < zz <  grid.shape[2]* dz:
                z = grid[:, :, zz].transpose()
                y = np.linspace(y0, y1, grid.shape[1])
                x = np.linspace(x0, x1, grid.shape[0])

                x, y = np.meshgrid(x, y)
                self.map_widget.ax.set_extent(extent, crs=ccrs.PlateCarree())
                self.map_widget.ax.coastlines()
                cs = self.map_widget.ax.contourf(x, y, z, levels=100, cmap=plt.cm.get_cmap('jet'), alpha=0.2)
                self.cb = self.map_widget.fig.colorbar(cs, orientation='horizontal', fraction=0.05, extend='both', pad=0.15,
                                                       label='Velocity (km/s)')

                if yy < grid.shape[1]:
                    try:
                        depth = np.linspace(0, grid.shape[2], grid.shape[2])
                        y = np.linspace(y0, y1, grid.shape[1])
                        depth, lat = np.meshgrid(depth, y)
                        z1 = grid[yy, :, :]
                        self.map_widget.lat.contourf(depth, lat, z1, levels=100, cmap=plt.cm.get_cmap('jet'), alpha=0.2)
                        self.map_widget.lat.set_ylim((y0-1, y1+1))
                    except:
                        pass

                if xx< grid.shape[0]:
                    try:
                        depth = np.linspace(0, grid.shape[2], grid.shape[2])
                        x = np.linspace(x0, x1, grid.shape[0])
                        depth, lon = np.meshgrid(depth, x)
                        z1 = grid[:, xx, :]
                        self.map_widget.lon.contourf(lon, depth, z1, levels=100, cmap=plt.cm.get_cmap('jet'), alpha=0.2)
                        self.map_widget.lon.xaxis.tick_top()
                        self.map_widget.lon.yaxis.tick_right()
                        self.map_widget.lon.invert_yaxis()
                    except:
                        pass

                self.map_widget.lon.set_xlim((x0-1, x0+1))
                gl = self.map_widget.ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                                                  linewidth=0.2, color='gray', alpha=0.2, linestyle='-')

                gl.top_labels = False
                gl.left_labels = False
                gl.xlines = False
                gl.ylines = False
                gl.xformatter = LONGITUDE_FORMATTER
                gl.yformatter = LATITUDE_FORMATTER
                self.map_widget.fig.canvas.draw()





