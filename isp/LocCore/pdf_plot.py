import matplotlib
import numpy as np
from matplotlib import pyplot as plt

def plot_scatter(scatter_x, scatter_y, scatter_z, pdf, ellipse):
    print("Plotting PDF")
    matplotlib.use("Qt5Agg")
    pdf = np.array(pdf) / np.max(pdf)

    left, width = 0.06, 0.65
    bottom, height = 0.1, 0.65
    spacing = 0.02
    rect_scatter = [left, bottom, width, height]
    rect_scatterlon = [left, bottom + height + spacing, width, 0.2]
    rect_scatterlat = [left + width + spacing, bottom, 0.2, height]

    fig = plt.figure(figsize=(10, 8))

    # plot lon,lat
    ax_scatter = plt.axes(rect_scatter)
    ax_scatter.tick_params(direction='in', top=True, right=True, labelsize=10)
    plt.scatter(scatter_x, scatter_y, s=10, c=pdf, alpha=0.5, marker=".", cmap=plt.cm.jet)
    plt.xlabel("Longitude", fontsize=10)
    plt.ylabel("Latitude", fontsize=10)

    # plot lon, depth
    ax_scatx = plt.axes(rect_scatterlon)
    ax_scatx.tick_params(direction='in', labelbottom=False, labelsize=10)
    plt.scatter(scatter_x, scatter_z, s=10, c=pdf, alpha=0.5, marker=".", cmap=plt.cm.jet)
    plt.ylabel("Depth (km)", fontsize=10)
    # plt.gca().invert_yaxis()

    # plot depth, lat
    ax_scatx = plt.axes(rect_scatterlat)
    ax_scatx.tick_params(direction='in', labeltop=False, labelleft=False, labelsize=10)
    plt.scatter(scatter_z, scatter_y, s=10, c=pdf, alpha=0.5, marker=".", cmap=plt.cm.jet)
    plt.xlabel("Depth (km)", fontsize=10)

    cax = plt.axes([0.95, 0.1, 0.02, 0.8])
    plt.colorbar(cax=cax)

    ## ploting uncertainty ellipse

    azimuth = math_angle_to_geo_azimuth(ellipse['azimuth'])
    azimuth_minor = azimuth + 90

    # Generate angles from 0 to 2*pi
    angles = np.linspace(0, 2 * np.pi, 100)

    # Calculate the major and minor axes vectors
    major_axis = np.array([np.cos(np.radians(azimuth)), np.sin(np.radians(azimuth))]) * ellipse['smax']
    minor_axis = np.array([np.cos(np.radians(azimuth_minor)), np.sin(np.radians(azimuth_minor))]) * ellipse['smin']

    # Calculate ellipse points
    den = 111.2 * np.cos(np.radians(ellipse['latitude']))
    x_points = ellipse['longitude'] + (major_axis[0] * np.cos(angles) + minor_axis[0] * np.sin(angles))/den
    y_points = ellipse['latitude'] + (major_axis[1] * np.cos(angles) + minor_axis[1] * np.sin(angles))/111.2

    # Plot major and minor axes
    plt.quiver(ellipse['longitude'], ellipse['latitude'], major_axis[0], major_axis[1], angles='xy', scale_units='xy',
               scale=1, color='r')
    plt.quiver(ellipse['longitude'], ellipse['latitude'], minor_axis[0], minor_axis[1], angles='xy', scale_units='xy',
               scale=1, color='b')

    # Plot the ellipse
    ax_scatter.plot(x_points, y_points)

    # Fill the area inside the ellipse
    plt.fill_between(x_points, y_points, color='gray', alpha=0.3)

    # Set aspect ratio to 'equal' for a proper ellipse display
    #plt.gca().set_aspect('equal', adjustable='box')
    # plt.clim(0, 1)
    plt.show()


def plot_ellipse(azimuth_major, azimuth_minor, length_major, length_minor, center=(0, 0), num_points=100):
    # Generate angles from 0 to 2*pi
    angles = np.linspace(0, 2 * np.pi, num_points)
    azimuth_major = math_angle_to_geo_azimuth(azimuth_major)
    azimuth_minor = azimuth_major +90
    # Calculate the major and minor axes vectors
    major_axis = np.array([np.cos(np.radians(azimuth_major)), np.sin(np.radians(azimuth_major))]) * length_major
    minor_axis = np.array([np.cos(np.radians(azimuth_minor)), np.sin(np.radians(azimuth_minor))]) * length_minor

    # Calculate ellipse points
    x_points = center[0] + major_axis[0] * np.cos(angles) + minor_axis[0] * np.sin(angles)
    y_points = center[1] + major_axis[1] * np.cos(angles) + minor_axis[1] * np.sin(angles)

    # Plot the ellipse
    plt.plot(x_points, y_points, label='Ellipse')

    # Fill the area inside the ellipse
    plt.fill_between(x_points, y_points, color='gray', alpha=0.3)

    # Plot major and minor axes
    plt.quiver(center[0], center[1], major_axis[0], major_axis[1], angles='xy', scale_units='xy', scale=1, color='r', label='Major Axis')
    plt.quiver(center[0], center[1], minor_axis[0], minor_axis[1], angles='xy', scale_units='xy', scale=1, color='b', label='Minor Axis')

    # Set aspect ratio to 'equal' for a proper ellipse display
    plt.gca().set_aspect('equal', adjustable='box')

    # Set labels and legend
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.legend()

    # Show the plot
    plt.show()

def math_angle_to_geo_azimuth(math_angle):
    """
    Convert a mathematical angle to a geographic azimuth.
    :param math_angle: Angle in the mathematical convention (counterclockwise from the positive x-axis).
    :return: Geographic azimuth (clockwise from true north).
    """
    geo_azimuth = (90 - math_angle) % 360
    return geo_azimuth