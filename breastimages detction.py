import cv2
import glob
import numpy as np
from matplotlib import pyplot as plt
from skimage import feature
import mahotas as mt

#numer of all images data normal and sample
#feature_all=np.zeros((602,13))
#i=0


for img in glob.glob("total_final/*.jpg"):
    image = cv2.imread(img)
    image1 = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)## 3D to 2d
    image2=cv2.equalizeHist(image1)
    image3 = cv2.Laplacian(image2, ddepth=-1, ksize=7, scale=3)
    image4 = cv2.medianBlur(image3,5)
    textures = mt.features.haralick(image4)
    feature_all[i,:]=textures.mean(axis=0)
    i=i+1


plt.imshow(image4)
X=feature_all
class1=np.ones((106,1))
class2=np.zeros((496,1))
Y=np.concatenate((class1,class2),axis=0)
#X = [[0, 0], [1, 1]]
#y = [0, 1]
X_sparse = coo_matrix(X)
X, X_sparse, Y = shuffle(X, X_sparse, Y, random_state=0)
