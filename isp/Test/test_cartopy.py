import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.io.img_tiles import GoogleTiles
from matplotlib.offsetbox import AnchoredText


def main():
    # fig = plt.figure()
    # ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
    proj = ccrs.PlateCarree()
    fig, ax = plt.subplots(1, 1, subplot_kw=dict(projection=proj))
    # ax = axes[0]
    # ax.set_extent([80, 170, -45, 30], crs=ccrs.PlateCarree())

    # Put a background image on for nice sea rendering.
    ax.stock_img()

    # Create a feature for States/Admin 1 regions at 1:50m from Natural Earth
    # states_provinces = cfeature.NaturalEarthFeature(
    #     category='cultural',
    #     name='admin_1_states_provinces_lines',
    #     scale='50m',
    #     facecolor='none')
    #
    # SOURCE = 'Natural Earth'
    # LICENSE = 'public domain'
    #
    # ax.add_feature(cfeature.LAND)
    # ax.add_feature(cfeature.COASTLINE)
    # ax.add_feature(states_provinces, edgecolor='gray')
    #
    # # Add a text annotation for the license information to the
    # # the bottom right corner.
    # text = AnchoredText(r'$\mathcircled{{c}}$ {}; license: {}'
    #                     ''.format(SOURCE, LICENSE), loc=4, prop={'size': 12}, frameon=True)
    # ax.add_artist(text)

    plt.show()


def plot_google_maps():
    fig = plt.figure(figsize=(10, 10))

    tiler = GoogleTiles(style="satellite")
    mercator = tiler.crs
    ax = plt.axes(projection=mercator)

    ax.set_extent((153, 153.2, -26.6, -26.4))

    zoom = 12
    ax.add_image(tiler, zoom)

    # even 1:10m are too coarse for .2 degree square
    #ax.coastlines('10m')

    home_lat, home_lon = -26.5258277, 153.0912987
    # Add a marker for home
    plt.plot(home_lon, home_lat, marker='o', color='red', markersize=5,
             alpha=0.7, transform=ccrs.Geodetic())

    plt.show()


if __name__ == '__main__':
    # main()
    plot_google_maps()
