from PIL import Image
import numpy as np

# loads in image into numpy array
imgArr = np.asarray(Image.open('panda.jpg'))

#img = Image.fromarray(np.uint8(imgArr))
Image.fromarray(np.uint8(imgArr)).show()
