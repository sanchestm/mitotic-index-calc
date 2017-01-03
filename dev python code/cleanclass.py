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
    directories=[r"C:/Users/Estêvão/Documents/Scripts/Python Scripts/cellsData/ClassAna",r"C:/Users/Estêvão/Documents/Scripts/Python Scripts/cellsData/Células classificadas - ZeH/Curva de crescimento/0dias/08-07-2016 (Menk)",r"C:/Users/Estêvão/Documents/Scripts/Python Scripts/cellsData/Células classificadas - ZeH/IM/16-09",r"C:/Users/Estêvão/Documents/Scripts/Python Scripts/cellsData/Células classificadas - ZeH/Sincronização jul2016/IM/31-08-2016 (Menck)"]
    #directories=[r"C:/Users/Estêvão/Documents/Scripts/Python Scripts/cellsData/ClassAna"]

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
    clf = cell_classifier(RandomForestClassifier(class_weight='balanced'))
    print('Fitando o modelo')
    clf.fit(X_train,y_train)
    Ypredict = clf.predict(X_test,y_test)
    print(confusion_matrix(y_test, Ypredict))
    print(np.mean(Ypredict==y_test))



    ## Add dispersion ratios to features
    y = features["class"].values
    b4time = time()
    features["dispersion"] = features.photo.apply(dispersionratio)
    aftertime = time()
    print("dispratio: "+ str(aftertime - b4time)+ "\n")

    ## Apply FFT to photos (does NOT add to features yet)
    b4time = time()
    features["FFT"] = features.photo.apply(fft.fft2)
    features["Phase"] = features.FFT.apply(np.angle)
    features["FFT"] = features.FFT.apply(abs)
    features["FFT"] = features["FFT"].apply(lambda x: np.reshape(x,(1,-1) ) )
    features["Phase"] = features["Phase"].apply(lambda x: np.reshape(x,(1,-1) ))
    features["frequency"] = np.hstack([row['FFT'], row['Phase']])
    aftertime = time()
    print("fft: "+ str(aftertime - b4time)+ "\n")






    #Suvrel multiplication
    Xabs = np.vstack([row["FFT"] for index,row in features.iterrows()])
    Xpha = np.vstack([row["Phase"] for index,row in features.iterrows()])
    X = np.hstack([Xabs,Xpha])
    gamma1 = suvrel(Xabs,y)
    gamma2 = suvrel(Xpha,y)
    print('suvrel calculado')
    mediaFFT = features['FFT'].mean()
    mediaPh = features['Phase'].mean()
    #features["FFT"].apply(lambda x: gamma1*x)#/mediaFFT)
    #features["Phase"].apply(lambda x: gamma2*x)#/mediaPh)

    ## Dimensionality reduction on the FFTs
    pca = PCA(n_components = 16)
    pcb = PCA(n_components = 16)
    fabsPCA = pca.fit(np.concatenate(features["FFT"].values,axis=0))
    fphiPCA = pcb.fit(np.concatenate(features["Phase"].values,axis=0))

    #Make features set
    X = np.array([x for x in features["dispersion"].values]).reshape(-1,1)
    X = np.hstack( ( X, fphiPCA.transform(np.concatenate(features["Phase"].values,axis=0) ), fabsPCA.transform(np.concatenate(features["FFT"].values,axis=0)) ) )

    X_train,X_test,y_train,y_test = train_test_split(X,y, train_size=trainpercentage, stratify=y)
#######################################################
    clf = RandomForestClassifier(n_estimators = 15,class_weight='balanced')
    rfecv = RFECV(estimator=clf, step=1, cv=StratifiedKFold(5))
    rfecv.fit(X_train,y_train)

    print("Optimal number of features : %d" % rfecv.n_features_)
    print("ranking: "+ str(rfecv.ranking_))
    print(rfecv.support_)
    # Plot number of features VS. cross-validation scores
    plt.figure()
    plt.xlabel("Number of features selected")
    plt.ylabel("Cross validation score (nb of correct classifications)")
    plt.plot(range(1, len(rfecv.grid_scores_) + 1), rfecv.grid_scores_)
    plt.show()
#########################################################
    print('Fitando o modelo')
    clf.fit(X_train[:,rfecv.support_],y_train)

    Ypredict = clf.predict(X_test[:,rfecv.support_])
    #Ypredict = clf.predict(testImg_feats_final)
    print(confusion_matrix(y_test, Ypredict))
    print(np.mean(Ypredict==y_test))

    #fazer o PCA nas features escolhidas e PLOTAR


if __name__ == "__main__":
    AutoClassification(.8)
