from sklearn.preprocessing import StandardScaler
from skimage import transform
from sklearn.decomposition import PCA
from numpy import fft
import numpy as np
from metriclearning import *
from dispersionratio import *
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.base import BaseEstimator

def rotate(df, degrees):
    result = df.copy()
    result.photo = result.photo.apply(lambda x: transform.rotate(x, degrees))
    result.photorgb = result.photorgb.apply(lambda x: transform.rotate(x, degrees))
    return result

def rotateN_AndConcatenate(df, number_of_rotations):
    print('Rotating photos')
    orig_df = df.copy()
    for i in [(360./(number_of_rotations+1) ) * (i+1) for i in range(number_of_rotations)]:
        df = pd.concat((df, rotate(orig_df, i)))
    return df

def numToClass(classificatedValues):
    classes = []
    for num in classificatedValues:
        if num==1:
            classes.append('mitose')
        else:
            classes.append('interfase')
    return classes

class cell_classifier():
    def __init__(self, n_features=10, n_rotations=3, min_samples=2):
        self.n_features = n_features
        self.n_rotations = n_rotations
        self.min_samples = min_samples

    def get_features(self,x):
        dispersion = dispersionratio(x['photo'])/100
        freqfeats = self.pca_.transform(x['freq'])[0,:]
        return np.hstack([dispersion,freqfeats])

    def raw_to_freq(self,photo):
        freq = fft.fft2(photo)
        phase = np.angle(freq)
        power = abs(freq)
        power =  np.reshape(power,(1,-1))
        L=len(power[0]); l = int(L/2)
        power=power[0,0:l]
        phase = np.reshape(phase,(1,-1))[0,0:l]

        return np.hstack([power,phase])

    def process_df(self,df,newparams=False):
        print('Processing Dataframe')
        X=df.copy()
        X['freq'] = X['photo'].apply(self.raw_to_freq)
        if newparams:
            self.pca_ = PCA(n_components = self.n_features)
            self.clf_ = RandomForestClassifier(class_weight='balanced',min_samples_split =self.min_samples,n_jobs=-1,bootstrap =False  )
            self.scaler_ = StandardScaler()
        X['freq'] = X['freq'].apply( lambda x: self.gamma_*self.scaler_.transform(np.reshape(x,(1,-1) ) ) )

    def fit(self,Xv,y):
        self.pca_ = PCA(n_components = self.n_features)
        self.clf_ = RandomForestClassifier(class_weight='balanced',min_samples_split =self.min_samples,n_jobs=-1,bootstrap =False  )
        self.scaler_ = StandardScaler()
        X = Xv.copy()
        if self.n_rotations > 0:
            X= rotateN_AndConcatenate(X, self.n_rotations)
            y = X["class"].values
        X['freq'] = X['photo'].apply(self.raw_to_freq)

        freqs = np.vstack(X['freq'].values)
        self.scaler_.fit(freqs)
        self.gamma_ = suvrel(self.scaler_.transform(freqs),y)

        X['freq'] = X['freq'].apply( lambda x: self.gamma_*self.scaler_.transform(np.reshape(x,(1,-1) ) ) )

        freqs = np.vstack(X['freq'].values)

        self.pca_.fit(freqs)
        print('Generating Features')
        feats = np.vstack([self.get_features(row) for index,row in X.iterrows()])
        self.clf_.fit(feats,y)

    def predict(self,Xv):
        ProbsOfEachRotation=[]
        ListOfRotationDegrees = [(360./(self.n_rotations+1) ) * (i) for i in range(self.n_rotations+1)]
        print('Generating Prediction Features')
        for irotation in range(self.n_rotations+1):
            Xi = rotate(Xv, ListOfRotationDegrees[irotation])
            Xi['freq'] = Xi['photo'].apply(self.raw_to_freq)
            Xi['freq'] = Xi['freq'].apply( lambda x: self.gamma_*self.scaler_.transform(np.reshape(x,(1,-1) ) ) )

            feats = np.vstack([self.get_features(row) for index,row in Xi.iterrows()])
            probs = self.clf_.predict_proba(feats)[:,1]
            ProbsOfEachRotation.append(probs)
        ProbsOfEachRotation = np.vstack(ProbsOfEachRotation)
        return numToClass(np.round(np.median(ProbsOfEachRotation,0)))

    def predict_proba(self,Xv):
        X = Xv.copy()
        X['freq'] = X['photo'].apply(self.raw_to_freq)
        X['freq'] = X['photo'].apply(self.raw_to_freq)
        X['freq'] = X['freq'].apply( lambda x: self.gamma_*self.scaler_.transform(np.reshape(x,(1,-1) ) ) )
        feats = np.vstack([self.get_features(row) for index,row in X.iterrows()])
        return self.clf_.predict_proba(feats)

    def score(X, y):
        return fbeta_score(y=='mitose', self.predict(X), beta)

    def get_params(self,deep=True):
        return {"n_features":self.n_features, "n_rotations":self.n_rotations, "min_samples":self.min_samples}

    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self
