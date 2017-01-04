import pandas as pd
import numpy as np
import glob
from skimage.exposure import adjust_gamma
from skimage.color import rgb2gray
from scipy import misc
from skimage import transform
#from texture import texture
from time import time
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
from sklearn.model_selection import train_test_split
from get_features import *
from sklearn.metrics import roc_curve
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import fbeta_score, make_scorer
from learning_curves import *

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

def AutoClassification(trainpercentage):
    #open images
    #directories=[r"C:/Users/Estêvão/Documents/Scripts/Python Scripts/cellsData/ClassAna",r"C:/Users/Estêvão/Documents/Scripts/Python Scripts/cellsData/Células classificadas - ZeH/Curva de crescimento/0dias/08-07-2016 (Menk)",r"C:/Users/Estêvão/Documents/Scripts/Python Scripts/cellsData/Células classificadas - ZeH/IM/16-09",r"C:/Users/Estêvão/Documents/Scripts/Python Scripts/cellsData/Células classificadas - ZeH/Sincronização jul2016/IM/31-08-2016 (Menck)"]
    directories=[r"C:/Users/Estêvão/Documents/Scripts/Python Scripts/cellsData/ClassAna"]

    features = pd.DataFrame()
    for saveDir in directories:
        allFiles = glob.glob(saveDir+"/*.csv")
        for file_ in allFiles:
            df = pd.read_csv(file_,index_col=None, header=0)
            df["file"] = df.file.apply(lambda x: saveDir +'/'+ x)
            features = pd.concat([features,df],ignore_index=True)
    features = features[np.logical_or(features['class']=='mitose'  ,features['class']=='interfase' )]


    features["photorgb"] = features.file.apply(lambda x: misc.imread(x))
    features["photo"] = features.photorgb.apply(norm_rgb2gray)
    features["photo"]= features.photo.apply(sizeMatch)

    y = features['class'].values
    X_train,X_test,y_train,y_test = train_test_split(features,y, train_size=trainpercentage, stratify=y)
    clf = cell_classifier(n_rotations=0)
    print('Fitando o modelo')
    clf.fit(X_train,y_train)
    ypred = clf.predict(X_test)
    ####################### GRID SEARCH
    scorer = make_scorer(lambda yt, yp: fbeta_score(yt, yp, beta=1,pos_label='mitose'))

    #plot_learning_curve(clf, 'Learning Curves', X_train, y_train, cv=5, train_sizes=np.linspace(.1, 1.0, 2),scoring=scorer)

    yproba = clf.predict_proba(X_test)[:,1]
    fpr,tpr, thr = roc_curve(y_test, yproba,pos_label='mitose')
    plt.plot(fpr,tpr)
    plt.show()
    Ypredict= yproba>0.192; y_test= y_test=='mitose'
    print(confusion_matrix(y_test, Ypredict))
    print(np.mean(Ypredict==y_test))
if __name__ == "__main__":
    AutoClassification(.8)
