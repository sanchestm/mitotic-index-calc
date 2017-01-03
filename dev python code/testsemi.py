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
def AutoClassification(trainpercentage):
    #open images
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


    training["photorgb"] = training.file.apply(lambda x: misc.imread(x))
    training["photo"] = training.photorgb.apply(rgb2gray)
    training["photo"]= training.photo.apply(sizeMatch)

    #training["photo"] = training.photo.apply(exposure.equalize_adapthist)
    testImg["photorgb"] = testImg.file.apply(lambda x: misc.imread(x))
    testImg["photo"] = testImg.photorgb.apply(rgb2gray)
    testImg["photo"]= testImg.photo.apply(sizeMatch)
    #testImg["photo"] = testImg.photo.apply(exposure.equalize_adapthist)

    ## Rotate training images
    #number_of_rotations = 3
    #training= rotateAll(training,number_of_rotations)
    #testImg= rotateAll(testImg,number_of_rotations)

    ## Initialize features with texture values
    b4time = time()
    train_feats = np.array([x for x in training.photo.apply(texture).values])
    Y_training = training["class"].values
    Y_testing = testImg["class"].values
    testImg_feats = np.array([x for x in testImg.photo.apply(texture).values])
    aftertime = time()
    print("texture: "+ str(aftertime - b4time)+ "\n")
    print(train_feats.shape)
    ## Add dispersion ratios to features
    b4time = time()
    training["dispersion"] = training.photo.apply(dispersionratio)
    train_feats = np.hstack( ( train_feats, np.array([x for x in training["dispersion"].values]).reshape(-1,1) ) )
    testImg["dispersion"] = testImg.photo.apply(dispersionratio)
    testImg_feats  = np.hstack( ( testImg_feats, np.array([x for x in testImg["dispersion"].values]).reshape(-1,1) ) )
    aftertime = time()
    print("dispratio: "+ str(aftertime - b4time)+ "\n")

####################################################
    #Add SimpleCV features ( HueHistogramFeatureExtractor, HaarLikeFeatureExtractor, MorphologyFeatureExtractor)
    #b4time = time()
    #training["SCVfeatures"] = training.photorgb.apply(getExtractors)
    #train_feats = np.hstack( ( train_feats, np.array([x for x in training["SCVfeatures"].values]).reshape(-1,1) ) )
    #testImg["SCVfeatures"]  = training.photorgb.apply(getExtractors)
    #testImg_feats = np.hstack( ( train_feats, np.array([x for x in training["SCVfeatures"].values]).reshape(-1,1) ) )
    #aftertime = time()
    #print("SimpleCV "+ str(aftertime - b4time)+ "\n")
#######################################################

    #Add feature img_to_graph to features
    #b4time = time()
    #training["img2graph"] = training.photo.apply(lambda x: img_to_graph(x))
    #train_feats = np.hstack( ( train_feats, np.array([x for x in training["img2graph"].values]).reshape(-1,1) ) )
    #testImg["img2graph"] = testImg.photo.apply(lambda x: img_to_graph(x))
    #testImg_feats  = np.hstack( ( testImg_feats, np.array([x for x in testImg["img2graph"].values]).reshape(-1,1) ) )
    #aftertime = time()
    #print("img_to_graph: "+ str(aftertime - b4time)+ "\n")



    ## Apply FFT to photos (does NOT add to features yet)
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
    print("fft: "+ str(aftertime - b4time)+ "\n")


    ## Dimensionality reduction on the FFTs
    pca = PCA(n_components = 10)
    pcb = PCA(n_components = 10)
    fabsPCA = pca.fit(np.concatenate(training["FFT"].values,axis=0))
    fphiPCA = pcb.fit(np.concatenate(training["Phase"].values,axis=0))

    #Adding ffts to feature set
    train_feats_final = np.hstack( ( train_feats, fphiPCA.transform(np.concatenate(training["Phase"].values,axis=0) ), fabsPCA.transform(np.concatenate(training["FFT"].values,axis=0)) ) )
    testImg_feats_final = np.hstack( ( testImg_feats, fphiPCA.transform(np.concatenate(testImg["Phase"].values, axis=0)), fabsPCA.transform(np.concatenate(testImg["FFT"].values,axis=0)) ) )
    meanScores=[]
    stdScores=[]
    for nest in []:
        #train_feats_final = normalize_columns(train_feats_final)
        #testImg_feats_final = normalize_columns(testImg_feats_final)
        clf = RandomForestClassifier(n_estimators = nest,class_weight='balanced')
        #clf = SVC()
        scores = cross_val_score(clf, train_feats_final, Y_training, cv=10)
        meanScores.append(np.mean(scores))
        stdScores.append(np.std(scores))
        #sklearn.feature_selection.RFECV(estimator, step=1, cv=None, scoring=None, verbose=0, n_jobs=1)[source]
        clf.fit(train_feats_final,Y_training)
#######################################################
    clf = RandomForestClassifier(n_estimators = 20,class_weight='balanced')
    rfecv = RFECV(estimator=clf, step=1, cv=StratifiedKFold(5))
    rfecv.fit(train_feats_final,Y_training)

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
    clf.fit(train_feats_final[:,rfecv.support_],Y_training)
    #print(meanScores)
    #print(stdScores)
    #plt.errorbar(range(10,20), meanScores, yerr=stdScores, fmt='o')
    #plt.show()
    print(confusion_matrix(Y_testing, clf.predict(testImg_feats_final[:,rfecv.support_])))
    print( np.mean(clf.predict(testImg_feats_final[:,rfecv.support_])==Y_testing))
    print('NOW TESTING SEMISUP')


if __name__ == "__main__":
    AutoClassification(.5)
