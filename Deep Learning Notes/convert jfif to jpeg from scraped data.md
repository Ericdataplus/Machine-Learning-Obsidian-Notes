```python

""" looop through a folder of jfif and convert them to .jpeg """

import os import glob import cv2

def convert\_to\_jpeg(folder): for file in glob.glob(folder +
'/\*.jfif'): img = cv2.imread(file) cv2.imwrite(file[:-5] + '.jpeg',
img) os.remove(file)

convert\_to\_jpeg('images')

```