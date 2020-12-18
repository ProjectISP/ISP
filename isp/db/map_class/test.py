'''
    This code is a example for adding image on the top of map using cartopy.
    The generated image can be found here: https://i.imgur.com/aTY1rYY.png
    '''

import matplotlib.pyplot as plt
import cartopy.crs as crs
from matplotlib.offsetbox import AnnotationBbox, OffsetImage
from PIL import Image

# Read image
lat = 116
lon = 39
img = Image.open('/Users/robertocabieces/Documents/ISPshare/isp/db/map_class/foca_mec.png')

# Plot the map
fig = plt.figure(figsize=(10, 5))
ax = plt.axes(projection=crs.PlateCarree())
ax.coastlines()
ax.stock_img()
# Use `zoom` to control the size of the image
imagebox = OffsetImage(img, zoom=0.1)
imagebox.image.axes = ax
ab = AnnotationBbox(imagebox, [lat, lon], pad=0, frameon=False)
ax.add_artist(ab)

plt.show()



