import cv2
import cvlib as cv
import matplotlib.pyplot as plt
from cvlib.object_detection import draw_bbox
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Input
# importamos la imagen a analizar
im = cv2.imread("*.jpg")
#con este tensor analizamos la imagen
bbox, label, conf = cv.detect_common_objects(im)
output_image = draw_bbox(im, bbox, label, conf)
plt.imshow(output_image)
plt.show()
print('* en la imagen: '+ str(label.count('*')))
# si la computadora que tengan no tiene gnu o no es lo suficientemente potente para realizar el analisis
# exite la opcion de utilizar Google Colab para eso solo hay que realizar pequeñas modificaciones en el codigo:
"""
from io import BytesIO
uploaded = files.upload()
from google.colab import files
from IPython.display import Image
"""
# despues de este codigo colocamos el de mas arriba