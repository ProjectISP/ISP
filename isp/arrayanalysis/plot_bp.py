import os
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
import numpy as np
import numpy.ma as ma
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.ndimage import gaussian_filter
import matplotlib as mpl
import scipy.ndimage
from matplotlib import gridspec
from isp import ROOT_DIR

def plot_bp(power_matrix, coords, step, st):


    mpl.rcParams['figure.figsize'] = (10,8)
    mpl.rcParams['figure.subplot.left'] = 0.1
    mpl.rcParams['figure.subplot.right'] = 0.96
    mpl.rcParams['figure.subplot.bottom'] = 0.07
    mpl.rcParams['figure.subplot.top'] = 0.97
    mpl.rcParams['figure.subplot.wspace'] = 0.25
    mpl.rcParams['figure.subplot.hspace'] = 0.16

    os.environ["CARTOPY_USER_BACKGROUNDS"] = os.path.join(ROOT_DIR, "maps")
    start = np.max([tr.stats.starttime for tr in st])

    Writer = animation.writers['ffmpeg']

    writer = Writer(fps=1, metadata=dict(artist='Me'), bitrate=2800)
    end = len(power_matrix[0, 0, :])-step

    xmin = coords[0]
    xmax = coords[1]
    ymin = coords[2]
    ymax = coords[3]

    #
    fig = plt.figure()
    gs = gridspec.GridSpec(1, 2, width_ratios=[3, 1])
    ax1 = plt.subplot(gs[0],projection=ccrs.PlateCarree())
    ax2 = plt.subplot(gs[1])

    ax1.background_img(name='ne_shaded', resolution='high')

    t = []
    pow_max = []

    for k in range(0, end, 1):
        power = power_matrix[:, :, k:k + step]
        pow_max.append(np.max(np.mean(power, axis=2)))
        t.append(k)

    ax2.plot(pow_max, t, '.-')

    ims = []
    time = str(start)

    #ax1.set_title(time, fontsize=16)
    i = 0
    for j in range(0, end, 1):

        power = power_matrix[:, :, j:j+step]
        pow = np.mean(power, axis=2)
        pow = scipy.ndimage.zoom(pow, 10, order=3)
        pow = gaussian_filter(pow, sigma=2)
        pow = np.clip(pow, a_min=0.0, a_max=1.0)
        pow = ma.masked_where(pow <= 0.4, pow)

        cmap = plt.get_cmap("jet")
        cs = ax1.imshow(pow, interpolation='bessel',
                   origin='lower', extent=[xmin, xmax, ymin, ymax], cmap=cmap,
                   vmax=1.0, vmin=0.0, alpha = 0.5 , transform=ccrs.PlateCarree())

        gl = ax1.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                          linewidth=0.2, color='gray', alpha=0.2, linestyle='-')

        gl.top_labels = False
        gl.right_labels = False
        gl.xlines = False
        gl.ylines = False
        gl.xformatter = LONGITUDE_FORMATTER
        gl.yformatter = LATITUDE_FORMATTER

        max_point, = ax2.plot(pow_max[0:i+1], t[0:i+1], '.', color='red')

        ax2.set_ylabel("Seconds from "+time, fontsize=16)

        if j ==0:
            cbar = fig.colorbar(cs, ax=ax1, extend='both', orientation='vertical', pad=0.05)
            cbar.ax.set_ylabel("ZLCC")

        ims.append([cs, max_point])

        i = i+1
        #ax.coastlines(resolution='10m', color='white', linewidth=0.15)

    im_ani = animation.ArtistAnimation(fig, ims, blit=True)

    im_ani.save('im.mp4', writer=writer)
    print("end movie maker")


def plot_cum(power_matrix, coords, step, st):

    j = 0
    mpl.rcParams['figure.figsize'] = (10,8)
    mpl.rcParams['figure.subplot.left'] = 0.1
    mpl.rcParams['figure.subplot.right'] = 0.89
    mpl.rcParams['figure.subplot.bottom'] = 0.07
    mpl.rcParams['figure.subplot.top'] = 0.94
    mpl.rcParams['figure.subplot.wspace'] = 0.25
    mpl.rcParams['figure.subplot.hspace'] = 0.16

    os.environ["CARTOPY_USER_BACKGROUNDS"] = os.path.join(ROOT_DIR, "maps")
    start = np.max([tr.stats.starttime for tr in st])
    #
    fig = plt.figure()

    ax1 = fig.add_subplot(1, 1, 1,projection=ccrs.PlateCarree())


    ax1.background_img(name='ne_shaded', resolution='high')

    xmin = coords[0]
    xmax = coords[1]
    ymin = coords[2]
    ymax = coords[3]

    ax1.set_title(str(start), fontsize=14)

    power = power_matrix[:, :, j:j+step]
    pow = np.mean(power, axis=2)
    min = pow.max()- 0.4*pow.max()

    max = pow.max()



    pow = scipy.ndimage.zoom(pow, 10, order=3)
    pow = gaussian_filter(pow, sigma=2)
    pow = np.clip(pow, a_min=min, a_max=max)
    pow = ma.masked_where(pow <= min, pow)

    #cmap = cmocean.cm.dense
    cmap = plt.get_cmap("jet")
    cs = ax1.imshow(pow, interpolation='bessel',
               origin='lower', extent=[xmin, xmax, ymin, ymax], cmap=cmap,
               vmin=min, vmax=max, alpha = 0.35 , transform=ccrs.PlateCarree())

    ax1.coastlines(resolution='10m', color='white', linewidth=0.15)
    gl = ax1.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                      linewidth=0.2, color='gray', alpha=0.2, linestyle='-')

    gl.top_labels = False
    gl.right_labels = False
    gl.xlines = False
    gl.ylines = False
    gl.xformatter = LONGITUDE_FORMATTER
    gl.yformatter = LATITUDE_FORMATTER


    cbar = fig.colorbar(cs, ax=ax1, extend='both', orientation='vertical', pad=0.05)
    cbar.ax.set_ylabel("ZLCC")

    plt.show()
