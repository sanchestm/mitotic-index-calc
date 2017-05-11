from metriclearning import suvrel
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

class Suvrel():
    def __init__(self):
        pass

    def fit(self,X,y):
        self.gamma_ = suvrel(X,y)

    def transform(self,X,y = None):
        return self.gamma_*X

    def fit_transform(self,X,y=None):
        self.fit(X,y)
        return self.transform(X,y)

    def get_params(self,deep=True):
        return {}

    def set_params(self):
        return self
