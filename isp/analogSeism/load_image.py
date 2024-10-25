try:
	import cv2
except ImportError as error:
	print("OpenCV not install please activate your environment and type: pip install opencv-python")
import numpy as np
from PIL import Image
from PyQt5.QtWidgets import QMessageBox, QFileDialog, QInputDialog

# Load imagen and pick on it class
class Load_Pick:
	"""
	"""
	def __init__(self, path_to_file):
		"""
		parmeters:
			path_to_file (str): Path to image file
		"""
		self.imagefile = path_to_file
		
		pass

	def pick_on_image(self, img, directory):
		"""
		Pick over an image using openCV modules
			Parameters:
					img (opencv object)	: Object image from opneCV
					directory (list)	: Listo of strings of the path to image file split by directory
					imagefile (str)		: Path to image file
			Returns:
					points (list)		: Listo fo x,y coordinates of every point picked over the image
					img (openCV object) : Object image from opneCV with the picked points
		"""
		import time
		ro, co = img.shape[:2]
		check=0
		try:
			color = (255, 0, 225)
			img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
		except:
			# img = clone.copy()
			colors = [(255,0,0),(0,0,255),(0,255,0),(255,255,0),(255,0,255),(0,255,255),(255,136,0)]
			color = colors[np.random.choice(len(colors))]

		# mb = QMessageBox()
		# mb.setIcon(QMessageBox.Information)
		# mb.setWindowTitle('Instructions',)
		print('"DobleClick"	Mark the coordinate on the seismogram image \n'
			'"z" 	   Undo the last marked point\n'
			'"r" 	   restarts vectorization function \n'
			'"Esc" 	   ends vectorization function')

		points = []

		def mouseDrawing(event, x, y, flags, params):
			if event == cv2.EVENT_LBUTTONDBLCLK:
				points.append((x, y))
		cv2.namedWindow(f'Vectorize on {directory[-1]}', cv2.WINDOW_NORMAL)
		cv2.resizeWindow(f'Vectorize on {directory[-1]}',(int(600), int(400))) # This size can be adapated to the screen resolution
		cv2.setMouseCallback(f'Vectorize on {directory[-1]}', mouseDrawing)
		while True:
			clone = img.copy()
			for cp in points:
				cv2.circle(clone, cp, 3, (0, 0, 255), -1)
				if len(points) > 1:
					for i in range(len(points)):
						if i != 0:
							cv2.line(clone, points[i-1], points[i], color, 2)
			cv2.imshow(f'Vectorize on {directory[-1]}',clone)
			key = cv2.waitKey(delay=1) & 0xff
			if key == ord('z'):
				points = points[:len(points) - 1]
			elif key == ord('r'):
				points = []
			elif key == 27:
				break
		img = clone.copy()
		del clone
		cv2.destroyWindow(f'Vectorize on {directory[-1]}')
		return img, points

	def load_image(self):
		"""
		Loads an images annd open it with openCV
            Parameters:
                
			Returns:
				img (opencv object)	: Object image from opneCV
				ppi (int)			: Pixels per inch of teh image
				directory (list)	: Listo of strings of the path to image file split by directory
				imagefile (str)		: Path to image file
		"""

		# imagefile, _ = QFileDialog.getOpenFileName(self, 'Load Seismogram Raster Image',
		# path, "Image Files (*.jpg *.png *.jpeg *.tif);; All files (*)")

		imagefile = self.imagefile
		directory = imagefile.split('/')
		try:
			ii = Image.open(imagefile)
			self.ppi = ii.info['dpi'][0]
		except KeyError:
			(inp, okPressed) = QInputDialog.getInt(self, 'Raster Image pixels per inch',
						'No PPI on raster information. \n'
						'Input PPI: ',600, 0, 3000, 2)
			if okPressed:
				self.ppi = float(inp)

		except NameError:
			QMessageBox.Critical(self,'Error!',
				'Without a PPI value, the usage of some functions will be limited')
		img = cv2.imread(self.imagefile,0)

		img, points = self.pick_on_image(img, directory)

		return img, points
	
	
if __name__ == "__main__":
	print("")
	# tmp = Load_Pick('/Users/admin/Documents/desarrollo/tiitba_0/examples/1928.03.22.GDL.NS.125.jpg')
	# img, points = tmp.load_image()
	#
	# print(points)
	#
	# print('All set ')
# 	instance = Load_and_pick('')
    
	