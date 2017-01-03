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
    def __init__(self, clf,n_features=10, n_rotations=3):
        self.clf=clf#RandomForestClassifier()
        self.pca = PCA(n_components = n_features-1)
        self.n_rotations = n_rotations

    def get_features(self,x):
        dispersion = dispersionratio(x['photo'])/100
        freqfeats = self.pca.transform(x['freq'])
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
        self.pca.fit(freqs)
        feats = np.vstack([self.get_features(row) for index,row in X.iterrows()])
        print(feats.shape)
        self.clf.fit(feats,y)

    def predict(self,Xv,y):
        X = Xv.copy()
        X['freq'] = X['photo'].apply(self.raw_to_freq)
        X['freq'] = X['photo'].apply(self.raw_to_freq)
        X['freq'] = X['freq'].apply( lambda x: self.gamma*self.scaler.transform(np.reshape(x,(1,-1) ) ) )
        feats = np.vstack([self.get_features(row) for index,row in X.iterrows()])
        print(feats)
        return self.clf.predict_proba(feats)
