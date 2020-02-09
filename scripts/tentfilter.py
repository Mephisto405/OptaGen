from skimage import filters
from skimage import img_as_float
import matplotlib.pyplot as plt
from scipy import misc

img = misc.imread('out.png')

blur_size = 0.5  # Standard deviation in pixels.

# Convert to float so that negatives don't cause problems
image = img_as_float(img)
blurred = filters.gaussian(image, blur_size)

misc.imsave('C:/Users/Dorian/data_scenes/mitsuba/bedroom/Figure_2.png', blurred)