from sklearn.base import BaseEstimator,ClassifierMixin
from scipy.spatial.distance import  cdist
import numpy as np
import copy
# normalized_df=(df-df.min())/(df.max()-df.min())
import pandas as pd
from collections import Counter
from sklearn.datasets import make_classification
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import confusion_matrix

import seaborn as sns
import prepare
import matplotlib.pyplot as plt
from sklearn import preprocessing

or_id = 0
itay_id = 3

class kNN(BaseEstimator, ClassifierMixin):
    def __init__(self, n_neighbors:int = 5):
        self.n_neighbors = n_neighbors
        self.data = None
        self.targets = None

    def fit(self, X, y):
        # self.data = copy.deepcopy(X)
        self.data = pd.DataFrame(X)
        self.targets = pd.DataFrame(y)
            # self.targets = copy.deepcopy(y)

        # we need to do by continous values?

        # TODO: complete
        return self


    def predict(self, X):
        # use closest exmaples (majority)
        # Note: You can use self.n_neighbors here
        distlist = cdist(X, self.data)
        indexes = np.argpartition(distlist, self.n_neighbors-1)[:, :self.n_neighbors]
        tmp = self.targets.to_numpy()
        labels = np.array(list(map(tmp.__getitem__, indexes.flatten())))
        # labels = np.take(self.targets,indexes)
        labels = np.expand_dims(a= labels, axis=0 ).reshape(indexes.shape)

        # labels = self.targets.values
        predictions = np.array(list(map(lambda x: Counter(x).most_common(1)[0][0], labels)))
        # TODO: compute the predicted labels (+1 or -1)
        return predictions


def visualize_clf(clf, X, Y, title, marker_size=250):
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap

    plt.figure(figsize=(8, 8))

    # Parameters
    n_classes = 2
    markers = ["P", "v"]
    palette = sns.color_palette("hls", 2)
    custom_cmap = ListedColormap(palette.as_hex())
    plot_step = 0.02

    x_delta = np.abs(X[:, 0].max() - X[:, 0].min()) * 0.1
    y_delta = np.abs(X[:, 1].max() - X[:, 1].min()) * 0.1
    x_min, x_max = X[:, 0].min() - x_delta, X[:, 0].max() + x_delta
    y_min, y_max = X[:, 1].min() - y_delta, X[:, 1].max() + y_delta
    xx, yy = np.meshgrid(np.arange(x_min, x_max, plot_step),
                         np.arange(y_min, y_max, plot_step))
    plt.tight_layout(h_pad=0.5, w_pad=0.5, pad=2.5)

    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    cs = plt.contourf(xx, yy, Z, cmap=custom_cmap, alpha=0.35)

    plt.xlabel("$x_1$")
    plt.ylabel("$x_2$")
    plt.grid(alpha=0.75)

    # Plot the training points
    for i, color, marker in zip(range(n_classes), palette, markers):
        idx = np.where(Y == i)
        plt.scatter(X[idx, 0], X[idx, 1], color=color,
                    marker=marker,
                    edgecolor='white', s=marker_size)

    plt.title(title, fontsize=20)
    plt.axis("tight")
    plt.show()












def normalize(df,row,process):
    temp = df[row].values.reshape(-1, 1)
    df[row] = process.fit_transform(temp)

    return df


from sklearn.model_selection import validation_curve


import seaborn as sn




if __name__ == '__main__':

    # df = pd.read_csv('ido.csv')
    df =  pd.read_csv('train_clean.csv.csv')
    df['spread'] = df['spread'].map(dict(High=1 , Low = -1))
    # Q5(df)
    # df = Q7(df.copy())

    df_normalize = prepare.normalize_data(df.copy())

    # Q8(df,df_normalize)
   # k = Q9(df=df_normalize)
    k = 9
    # best k is 9
    # Q10(df=df_normalize,k=k)




