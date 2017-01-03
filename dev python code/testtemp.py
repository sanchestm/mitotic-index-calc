import pandas as pd
import numpy as np
import glob
from skimage.exposure import adjust_gamma
from skimage.color import rgb2gray
from scipy import misc
from skimage import transform
from texture import texture
from time import time
from dispersionratio import *
#from image_rec import getExtractors
from numpy import fft
from vectorize import vectorize
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score,StratifiedKFold
from sklearn.feature_selection import RFECV
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA, FastICA
from metriclearning import *


def rotate(df, degrees):
    result = df.copy()
    result.photo = result.photo.apply(lambda x: transform.rotate(x, degrees))
    result.photorgb = result.photorgb.apply(lambda x: transform.rotate(x, degrees))
    return result

def rotateAll(df, number_of_rotations):
    orig_df = df.copy()
    for i in [(360./number_of_rotations) * (i+1) for i in range(number_of_rotations)]:
        df = pd.concat((df, rotate(orig_df, i)))
    return df

def sizeMatch(array):
    ash=array.shape
    if ash==(100,100):
        return array
    else:
        new=np.zeros((100,100))
        new[50-round(ash[0]/2):50+round(ash[0]/2),50-round(ash[1]/2):50+round(ash[1]/2)]
        return new

def norm_rgb2gray(image):
    try:
        if image.shape[2] == 3: return rgb2gray(image)
    except: return rgb2gray(image)*0.0721



trainpercentage=.9
directories=[r"C:/Users/Estêvão/Documents/Scripts/Python Scripts/cellsData/ClassAna",r"C:/Users/Estêvão/Documents/Scripts/Python Scripts/cellsData/Células classificadas - ZeH/Curva de crescimento/0dias/08-07-2016 (Menk)",r"C:/Users/Estêvão/Documents/Scripts/Python Scripts/cellsData/Células classificadas - ZeH/IM/16-09",r"C:/Users/Estêvão/Documents/Scripts/Python Scripts/cellsData/Células classificadas - ZeH/Sincronização jul2016/IM/31-08-2016 (Menck)"]
#directories=[r"C:/Users/Estêvão/Documents/Scripts/Python Scripts/cellsData/ClassAna"]

training = pd.DataFrame()
testImg = pd.DataFrame()
for saveDir in directories:
    allFiles = glob.glob(saveDir+"/*.csv")
    for file_ in allFiles:
        df = pd.read_csv(file_,index_col=None, header=0)
        df["file"] = df.file.apply(lambda x: saveDir +'/'+ x)

        lastTrainEx = int(len(df)*trainpercentage)
        training = pd.concat([training,df.ix[:lastTrainEx]],ignore_index=True)
        testImg  = pd.concat([testImg,df.ix[lastTrainEx:]],ignore_index=True)

training = training[np.logical_or(training['class']=='mitose'  ,training['class']=='interfase' )]
testImg = testImg[np.logical_or(testImg['class']=='mitose'  ,testImg['class']=='interfase' )]

#fazer shuffle do dataframe TO DO

training["photo"] = training.file.apply(lambda x: misc.imread(x))
training["photo"] = training.photo.apply(norm_rgb2gray)
training["photo"]= training.photo.apply(sizeMatch)
testImg["photo"] = testImg.file.apply(lambda x: misc.imread(x))
testImg["photo"] = testImg.photo.apply(norm_rgb2gray)
testImg["photo"]= testImg.photo.apply(sizeMatch)
b4time = time()
training["FFT"] = training.photo.apply(fft.fft2)
training["Phase"] = training.FFT.apply(np.angle)
training["FFT"] = training.FFT.apply(abs)
testImg["FFT"] = testImg.photo.apply(fft.fft2)
testImg["Phase"] = testImg.FFT.apply(np.angle)
testImg["FFT"] = testImg.FFT.apply(abs)
training["FFT"] = training["FFT"].apply(lambda x: np.reshape(x,(1,-1) ) )
training["Phase"] = training["Phase"].apply(lambda x: np.reshape(x,(1,-1) ))
testImg["FFT"] = testImg["FFT"].apply(lambda x: np.reshape(x,(1,-1) ) )
testImg["Phase"] = testImg["Phase"].apply(lambda x: np.reshape(x,(1,-1) ))
aftertime = time()

#training["photo"] = training.photo.apply(exposure.equalize_adapthist)

Y_training = training["class"].values
Xabs = np.vstack([training.ix[i,"FFT"] for i in training.index])
Xpha = np.vstack([training.ix[i,"Phase"] for i in training.index])
gamma1 = suvrel(Xabs,Y_training)
gamma2 = suvrel(Xpha,Y_training)
print('suvrel calculado')
training["FFT"].apply(lambda x: gamma1*x)
training["Phase"].apply(lambda x: gamma2*x)
testImg["FFT"].apply(lambda x: gamma1*x)
testImg["Phase"].apply(lambda x: gamma2*x)
