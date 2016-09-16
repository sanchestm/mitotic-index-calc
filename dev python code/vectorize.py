import numpy as np
import pandas as pd
def vectorize(df_field):
    L = np.prod(df_field.iloc[0].shape)
    return np.array([df_field.iloc[i].reshape(L) for i in range(len(df_field))])

def normalize_columns(X):
    newMean = 0
    newVar  = 1000
    normX = np.zeros(X.shape)
    for icol in range(len(X[0])):
        normX[:,icol] = newVar*( X[:,icol]-np.mean(X[:,icol]) )/np.sqrt(np.var(X[:,icol]))
    return normX
