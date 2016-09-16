import skimage.morphology as skm
import numpy as np
from scipy import misc
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from skimage.color import rgb2gray

def dispersionratio(image, alpha_th=5):
#image = misc.imread("../data/cellI4.tif")
    [h,l]=image.shape

    sortim = np.sort(np.reshape(image,h*l))
    th = sortim[int(round(h*l-h*l*alpha_th/100.))]

    coordsy = np.array([[ j for i in range(h)] for j in range(l)]); ys = coordsy[image>=th]
    coordsx = np.transpose(coordsy); xs = coordsx[image>=th]
    positions = np.array([[i,j] for i,j in zip(xs,ys)])

    pca = PCA()
    pca.fit(positions)
    [lamb1, lamb2] = pca.explained_variance_ratio_
    return (lamb1/lamb2)
