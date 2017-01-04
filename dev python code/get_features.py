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

def rotateAll(df, number_of_rotations):
    orig_df = df.copy()
    for i in [(360./number_of_rotations) * (i+1) for i in range(number_of_rotations)]:
        df = pd.concat((df, rotate(orig_df, i)))
    return df



class cell_classifier():
    def __init__(self, n_features=10, n_rotations=0, min_samples=2):
        self.n_features = n_features
        self.n_rotations = n_rotations
        self.min_samples = min_samples

    def get_features(self,x):
        dispersion = dispersionratio(x['photo'])/100
        freqfeats = self.pca_.transform(x['freq'])
        return np.hstack([dispersion,freqfeats[0,:]])

    def raw_to_freq(self,photo):
        freq = fft.fft2(photo)
        phase = np.angle(freq)
        power = abs(freq)
        power =  np.reshape(power,(1,-1))
        L=len(power[0]); l = int(L/2)
        power=power[0,0:l]
        phase = np.reshape(phase,(1,-1))[0,0:l]

        return np.hstack([power,phase])

    def fit(self,Xv,y):
        X = Xv.copy()
        X['freq'] = X['photo'].apply(self.raw_to_freq)
        self.pca_ = PCA(n_components = self.n_features)
        self.clf = RandomForestClassifier(class_weight='balanced',min_samples_split=self.min_samples)
        self.scaler = StandardScaler()
        freqs = np.vstack(X['freq'].values)

        self.scaler.fit(freqs)
        self.gamma = suvrel(self.scaler.transform(freqs),y)
        if self.n_rotations > 0:
            X= rotateAll(X, self.n_rotations)
            y = X["class"].values
        X['freq'] = X['photo'].apply(self.raw_to_freq)
        X['freq'] = X['freq'].apply( lambda x: self.gamma*self.scaler.transform(np.reshape(x,(1,-1) ) ) )
        freqs = np.vstack(X['freq'].values)

        self.pca_.fit(freqs)
        feats = np.vstack([self.get_features(row) for index,row in X.iterrows()])
        self.clf.fit(feats,y)

    def predict(self,Xv):
        X = Xv.copy()
        X['freq'] = X['photo'].apply(self.raw_to_freq)
        X['freq'] = X['photo'].apply(self.raw_to_freq)
        X['freq'] = X['freq'].apply( lambda x: self.gamma*self.scaler.transform(np.reshape(x,(1,-1) ) ) )
        feats = np.vstack([self.get_features(row) for index,row in X.iterrows()])
        return self.clf.predict(feats)

    def predict_proba(self,Xv):
        X = Xv.copy()
        X['freq'] = X['photo'].apply(self.raw_to_freq)
        X['freq'] = X['photo'].apply(self.raw_to_freq)
        X['freq'] = X['freq'].apply( lambda x: self.gamma*self.scaler.transform(np.reshape(x,(1,-1) ) ) )
        feats = np.vstack([self.get_features(row) for index,row in X.iterrows()])
        return self.clf.predict_proba(feats)

    def score(X, y):
        return fbeta_score(y, self.predict(X), beta)

    def get_params(self,deep=True):
        return {"n_features":self.n_features, "n_rotations":self.n_rotations, "min_samples":self.min_samples}

    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self
