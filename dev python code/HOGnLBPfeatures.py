import numpy as np
import skimage.feature as ft
from numpy.fft import fft2


def abs_pha_fft(image):
    imfft = fft2(image)
    return np.hstack([np.abs(imfft).reshape(1,-1), np.angle(imfft).reshape(1,-1)])[0]

def getHoG(image):
    HOG = ft.hog(image).reshape(1,-1)
    return HOG[0]

def getLBP(image):
    hLBP = np.array([]).reshape(1,-1)
    for P in range(4,20,2):
        for R in range(2,10,2):
            LBP = ft.local_binary_pattern(image, P, R, 'uniform').reshape(-1,1)
            hLBP = np.hstack((np.histogram(LBP,50,range=(0,50))[0].reshape(1,-1), hLBP))
    return hLBP[0]

def Get_all_features(image):
    return np.hstack([abs_pha_fft(image),getHoG(image), getLBP(image)])
