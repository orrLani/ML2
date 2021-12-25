from sklearn.base import BaseEstimator,ClassifierMixin
from scipy.spatial.distance import  cdist
import numpy as np
import copy
# normalized_df=(df-df.min())/(df.max()-df.min())
import pandas as pd
from collections import Counter
from sklearn.datasets import make_classification
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
    # X is the dataframe
    # y is the labels of target values (names)
    # def fit(self, X, y):
    #     # self.data = copy.deepcopy(X)
    #     self.data = X.copy()
    #     if y.ndim == 1:
    #         self.targets = np.expand_dims(a=y,axis=1)
    #     else:
    #         self.targets = y.copy()
    #         # self.targets = copy.deepcopy(y)
    #
    #     # we need to do by continous values?
    #
    #     # TODO: complete
    #     return self
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


def Q1(data):
    X_toy, y_toy = make_classification(n_samples=100, n_features=2,
                                       random_state=or_id+itay_id, flip_y=0.1,
                                       n_informative=2, n_redundant=0)


def Q2(df):
    s = df.corr().spread.abs()
    s.sort_values(kind="quicksort", ascending=False)


def Q3(df):
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')

    n = 100
    ax.tick_params(axis='x', colors='green')
    ax.tick_params(axis='y', colors='blue')
    ax.tick_params(axis='z', colors='black')
    # For each set of style and range settings, plot n random points in the box
    # defined by x in [23, 32], y in [0, 100], z in [zlow, zhigh].
    samples = df.sample(n= n )
    for m, zlow, zhigh in [('o', -50, -25), ('^', -30, -5)]:
        if (samples['spread'].dtypes == np.int64 ):
            samples1 = samples[samples['spread'] == 1]
            samples2 = samples[samples['spread'] == -1]
        else:
            samples1 = samples[samples['spread'] == 'High']
            samples2 = samples[samples['spread'] == 'Low']
        pcr03 = samples1['PCR_03'].values
        pcr07 = samples1['PCR_07'].values
        pcr10 = samples1['PCR_10'].values
        ax.scatter(pcr07, pcr10, pcr03, marker=m, c= 'green')
        pcr03 = samples2['PCR_03'].values
        pcr07 = samples2['PCR_07'].values
        pcr10 = samples2['PCR_10'].values
        ax.scatter(pcr07, pcr10, pcr03, marker=m, c='red')

    ax.set_zlabel('PCR_03')
    ax.set_ylabel('PCR_10')
    ax.set_xlabel('PCR_07')
    ax.view_init(30, 25)
    plt.show()

def Q5(df):
    # df['spread'] = df['spread'].map(dict(High=1 , Low = -1))
    # arr = df[['PCR_03','PCR_07','PCR_10','spread']].to_numpy()
    my_knn = kNN(n_neighbors=11)
    my_knn = my_knn.fit(df[['PCR_03','PCR_07','PCR_10','spread']], df['spread'])
    print(my_knn.score(my_knn.data, my_knn.targets))


def normalize(df,row,process):
    temp = df[row].values.reshape(-1, 1)
    df[row] = process.fit_transform(temp)

    return df

def Q7_normalize(df):


    df = normalize(df=df,row='PCR_03',process=preprocessing.StandardScaler())
    df = normalize(df=df,row='PCR_07',process=preprocessing.StandardScaler())
    df = normalize(df=df,row= 'PCR_10',process=preprocessing.MinMaxScaler())
    #df = normalize(df=df, row='PCR_10', process=preprocessing.StandardScaler())
    my_knn = kNN(n_neighbors=11)

    # print the knn score
    my_knn = my_knn.fit(df[['PCR_03','PCR_07','PCR_10','spread']], df['spread'])
    print(my_knn.score(my_knn.data, my_knn.targets))

    # normalize the rest features

def Q8(df_before,df_after):

    # df.hist(figsize=(10, 10))
    # df_normalize = prepare.normalize_data(df.copy())
    # df_normalize.hist(figsize=(10, 10))

    # print household_income
    df_before['household_income'].plot.hist(title="household_income before normalize")
    plt.show()
    df_after['household_income'].plot.hist(title="household_income after MinMaxScaler normalize")
    plt.show()
    # print sugar_levels
    df_before['sugar_levels'].plot.hist(title="sugar_levels before normalize")
    plt.show()
    df_after['sugar_levels'].plot.hist(title="sugar_levels after StandardScaler normalize")
    plt.show()
    # ax("Frequency")
    # ax.set_ylable('Standard deviation')


    # prepare to read the graths

    plt.show()




#############
############3
def numerical_subgradient(w, b, C, X, y, delta=1e-4):
    w_ = w.copy()
    g_w = np.zeros_like(w_)
    orig_objective = SoftSVM.loss(w_, b, C, X, y)
    for i in range(g_w.shape[0]):
        w_[i] += delta
        perturbed_objective = SoftSVM.loss(w_, b, C, X, y)
        w_[i] -= delta
        g_w[i] = (perturbed_objective - orig_objective) / delta

    g_b = (SoftSVM.loss(w_, b + delta, C, X, y) - orig_objective) / delta
    return g_w, g_b


def compare_gradients(X, y, deltas, C=1, REPEATS=100):
    residual_means = []

    for delta in deltas:
        residuals = []

        for _ in range(REPEATS):
            # Randomize vectors in which the gradient is computed
            w = np.random.randn(X.shape[1])
            b = np.random.randn(1)

            # Compute the two types of gradients
            analytic_grad = SoftSVM.subgradient(w, b, C, X, y)[0]
            numeric_grad = numerical_subgradient(w, b, C, X, y, delta=delta)[0]

            residual = np.linalg.norm(numeric_grad - analytic_grad)
            residuals.append(residual)

        residual_means.append(np.mean(residuals))

    plt.plot(deltas, residual_means, linewidth=3)
    plt.yscale('log')
    plt.xscale('log')
    plt.xlabel('$\delta$', fontsize=18)
    plt.ylabel('dist(analytic, numeric$_\delta$)', fontsize=18)
    plt.ylabel(r'$\left\Vert p_{C}\left(w,b\right) - u_{\delta} \left(w,b\right)\right\Vert$', fontsize=18)

    plt.grid(alpha=0.5)
    plt.show()

###############
#################################
class SoftSVM(BaseEstimator, ClassifierMixin):
    """
    Custom C-Support Vector Classification.
    """
    def __init__(self, C: float, lr: float = 1e-5):
        """
        Initialize an instance of this class.
        ** Do not edit this method **

        :param C: inverse strength of regularization. Must be strictly positive.
        :param lr: the SGD learning rate (step size)
        """
        self.C = C
        self.lr = lr
        self.batch_size = 32
        self.w = None
        self.b = 0.0

    # Initialize a random weight vector
    def init_solution(self, n_features: int):
        """
        Randomize an initial solution (weight vector)
        ** Do not edit this method **

        :param n_features:
        """
        self.w = np.random.randn(n_features)
        self.b = 0.0

    @staticmethod
    def loss(w, b: float, C: float, X, y):
        """
        Compute the SVM objective loss.

        :param w: weight vector for linear classification; array of shape (n_features,)
        :param b: bias scalar for linear classification
        :param C: inverse strength of regularization. Must be strictly positive.
        :param X: samples for loss computation; array of shape (n_samples, n_features)
        :param y: targets for loss computation; array of shape (n_samples,)
        :return: the Soft SVM objective loss (float scalar)
        """
        margins = (X.dot(w) + b).reshape(-1, 1)
        hinge_inputs = np.multiply(margins, y.reshape(-1, 1))

        norm = np.linalg.norm(w)
        np.power(norm,2)
        # TODO: complete the loss calculation
        loss = 0.0

        loss = np.power(norm,2) * C * np.sum( (max(0,1-hinge_inputs)))
        return loss

    @staticmethod
    def subgradient(w, b: float, C: float, X, y):
        """
        Compute the (analytical) SVM objective sub-gradient.

        :param w: weight vector for linear classification; array of shape (n_features,)
        :param b: bias scalar for linear classification
        :param C: inverse strength of regularization. Must be strictly positive.
        :param X: samples for loss computation; array of shape (n_samples, n_features)
        :param y: targets for loss computation; array of shape (n_samples,)
        :return: a tuple with (the gradient of the weights, the gradient of the bias)
        """
        # TODO: calculate the analytical sub-gradient of soft-SVM w.r.t w and b
        margins = (X.dot(w) + b).reshape(-1, 1)
        hinge_inputs = np.multiply(margins, y.reshape(-1, 1))

        f = lambda x : -1 if x < 1 else 1
        g_w =2*w + C * np.multiply(np.multiply(np.sum(f(hinge_inputs), y), X))

        g_b = np.multiply(np.sum(f(hinge_inputs), y))
        # g_w = None
        # g_b = 0.0

        return g_w, g_b

    def fit_with_logs(self, X, y, max_iter: int = 2000, keep_losses: bool = True):
        """
        Fit the model according to the given training data.

        :param X: training samples; array of shape (n_samples, n_features)
        :param y: training targets; array of shape (n_samples,)
        :param max_iter: number of SGD iterations
        :param keep_losses:
        :return: the training losses and accuracies during training
        """
        # Initialize learned parameters
        self.init_solution(X.shape[1])

        losses = []
        accuracies = []

        if keep_losses:
            losses.append(self.loss(self.w, self.b, self.C, X, y))
            accuracies.append(self.score(X, y))

        # Iterate over batches
        for iter in range(0, max_iter):
            start_idx = (iter * self.batch_size) % X.shape[0]
            end_idx = min(X.shape[0], start_idx + self.batch_size)
            batch_X = X[start_idx: end_idx]
            batch_y = y[start_idx: end_idx]

            # TODO: Compute the (sub)gradient of the current *batch*
            g_w, g_b = None, None

            # Perform a (sub)gradient step
            # TODO: update the learned parameters correctly
            self.w = None
            self.b = 0.0

            if keep_losses:
                losses.append(self.loss(self.w, self.b, self.C, X, y))
                accuracies.append(self.score(X, y))

        return losses, accuracies

    def fit(self, X, y, max_iter: int = 2000):
        """
        Fit the model according to the given training data.
        ** Do not edit this method **

        :param X: training samples; array of shape (n_samples, n_features)
        :param y: training targets; array of shape (n_samples,)
        :param max_iter: number of SGD iterations
        """
        self.fit_with_logs(X, y, max_iter=max_iter, keep_losses=False)

        return self

    def predict(self, X):
        """
        Perform classification on samples in X.

        :param X: samples for prediction; array of shape (n_samples, n_features)
        :return: Predicted class labels for samples in X; array of shape (n_samples,)
        """
        # TODO: compute the predicted labels (+1 or -1)
        y_pred = None

        return y_pred

#############


if __name__ == '__main__':

    df = pd.read_csv('train_clean.csv.csv')
    df['spread'] = df['spread'].map(dict(High=1 , Low = -1))
    Q5(df)
    # Q7_normalize(df)

    df_normalize = prepare.normalize_data(df.copy())

    compare_gradients(df_normalize, df , deltas=np.logspace(-5, -1, 9))

    # f = kNN(2)
    # X = np.array([[1,1,1],[1,2,3],[-1,-1,-1]])
    # y = np.array([1 , -1 ,1])
    #
    # f.fit(X,y)
    # f.predict(X)
    # print("meow")
    #
    # fig = plt.figure()
    # ax = fig.add_subplot(projection='3d')
    #
    # n = 100
    # ax.tick_params(axis='x', colors='green')
    # ax.tick_params(axis='y', colors='blue')
    # ax.tick_params(axis='z', colors='black')
    # # For each set of style and range settings, plot n random points in the box
    # # defined by x in [23, 32], y in [0, 100], z in [zlow, zhigh].
    # df = pd.read_csv("train_clean.csv.csv")
    # samples = df.sample(n= n )
    # for m, zlow, zhigh in [('o', -50, -25), ('^', -30, -5)]:
    #     samples1 = samples[samples['spread'] == 'High']
    #     pcr03 = samples1['PCR_03'].values
    #     pcr07 = samples1['PCR_07'].values
    #     pcr10 = samples1['PCR_10'].values
    #     ax.scatter(pcr07, pcr10, pcr03, marker=m, c= 'green')
    #     samples2 = samples[samples['spread'] == 'Low']
    #     pcr03 = samples2['PCR_03'].values
    #     pcr07 = samples2['PCR_07'].values
    #     pcr10 = samples2['PCR_10'].values
    #     ax.scatter(pcr07, pcr10, pcr03, marker=m, c='red')
    #
    #
    #
    # ax.set_zlabel('PCR_03')
    # ax.set_ylabel('PCR_10')
    # ax.set_xlabel('PCR_07')
    # # ax.view_init(30, 25)
    # plt.show()
