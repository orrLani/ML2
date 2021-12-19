from sklearn.base import BaseEstimator,ClassifierMixin
from scipy.spatial.distance import  cdist
import numpy as np
import copy
# normalized_df=(df-df.min())/(df.max()-df.min())
import pandas as pd

class kNN(BaseEstimator, ClassifierMixin):
    def __init__(self, n_neighbors:int = 5):
        self.n_neighbors = n_neighbors

    # X is the dataframe
    # y is the labels of target values (names)
    def fit(self, X, y):
        self.data = copy.deepcopy(X)
        self.targets = self.data[y]
        #self.data = self.data.drop(axis=1,columns= y)
        # find closest examples
        # we need to do by continous values?

        # TODO: complete
        return self
    def predict(self, X):
        # use closest exmaples (majority)
        # Note: You can use self.n_neighbors here
        df = pd.DataFrame(X)
        for i in df.values:
            distlist = cdist(self.data,i)
            lables = np.argpartition(distlist, self.n_neighbors)
            tmp = df[lables]
            predict = tmp['covid'].mode()
        #idx = np.argpartition(A, k)
        predictions = None

        # TODO: compute the predicted labels (+1 or -1)
        return predictions


if __name__ == '__main__':
    data = pd.read_csv('virus_data.csv')
    data['covid'] = data['covid']*2-1
    f = kNN()
    f.fit(data,['spread','risk','covid'])
    f.predict(data)
    print("meow")