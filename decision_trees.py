import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import prepare
from sklearn import tree
import graphviz
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
def plot(data,COL_NAMES,COLS=3):
    # COL_NAME = ['age','num_of_siblings','conversations_per_day']
    ROWS = int(np.ceil(3/COLS))
    for i, column in enumerate(COL_NAMES, 1):
        ax = plt.subplot(ROWS, COLS, i)
        ax.title.set_text(column)
        sns.histplot(data=data, x=column, hue='risk', kde=True)
        plt.grid(alpha=0.5)
    plt.tight_layout()
    plt.show()


def get_histograms(data):
    # household_income sugar_levels sport_activity
    # COL_NAME = ['risk', 'covid', 'spread']
    COL_NAME = []
    COL_NAME_1 = ['age','num_of_siblings','conversations_per_day']
    COL_NAME_2 = ['household_income','sugar_levels','sport_activity']
                #','PCR_01','PCR_02','PCR_03',
                # 'PCR_04','PCR_05','PCR_06','PCR_07','PCR_08','PCR_09','PCR_10']
    COL_NAME_3 = ['PCR_01','PCR_02','PCR_03']

    COL_NAME_4 = ['PCR_04','PCR_05','PCR_06']

    COL_NAME_5 = ['PCR_04', 'PCR_05', 'PCR_06']

    COL_NAME_6 = ['PCR_07','PCR_08','PCR_09','PCR_10']

    plot(data, COL_NAME_1)
    plot(data, COL_NAME_2)
    plot(data, COL_NAME_3)
    plot(data, COL_NAME_4)
    plot(data, COL_NAME_5)
    plot(data, COL_NAME_6,4)





def Q11(data):
    pass



def Q12(data):
    # get_histograms(data)

    # Corr between the features to risk
    names = data.keys().to_list()
    corr_dict = {}
    for name in names:
        try:
            if name =='is_army' or name=='risk':
                raise Exception
            corr = abs(data['risk'].corr(data[name]))
            corr_dict[name] = corr

            # print("Correlation between risk and  " + name + " is: {:.3f}".format(corr))

        except Exception:
            pass
    corr_dict = {k: v for k, v in sorted(corr_dict.items(), key=lambda item: item[1],reverse=True)}
    print(corr_dict)
    # df['risk'] = df['risk'].map(dict(High=1, Low=-1))
    list = []

    for item in corr_dict:
        list.append(item)    # get_histograms(data)

    # Corr between the features to risk
    names = data.keys().to_list()
    corr_dict = {}
    for name in names:
        try:
            if name =='is_army' or name=='risk':
                raise Exception
            corr = abs(data['risk'].corr(data[name]))
            corr_dict[name] = corr

            print("Correlation between risk and  " + name + " is: {:.3f}".format(corr))

        except Exception:
            pass
    corr_dict = {k: v for k, v in sorted(corr_dict.items(), key=lambda item: item[1],reverse=True)}
    print(corr_dict)
    # df['risk'] = df['risk'].map(dict(High=1, Low=-1))
    list = []

    for item in corr_dict:
        list.append(item)

def Q13(data):
    model = tree.DecisionTreeClassifier(criterion="entropy",max_depth=4)
    # drop
    data = data.drop(['Unnamed: 0','covid','spread'], axis=1)
    X = data
    Y = data['risk']
    X= X.drop(['risk'], axis=1)
    # X = np.float32(X)
    # Y = np.float(Y)

    # X = np.nan_to_num(X, nan=-9999, posinf=33333333, neginf=33333333)
    # Y = np.nan_to_num(Y, nan=-9999, posinf=33333333, neginf=33333333)
    model.fit(X,Y)
    plt.figure(figsize=(19, 19))
    tree.plot_tree(model,filled=True,fontsize=7,feature_names=X.columns)
    plt.title('DecisionTreeClassifier with max death 4')
    plt.show()

    predict = model.predict(X)
    risk = accuracy_score(Y, predict)
    print(risk)


    # create train set
    # feature_cols =
    # X = pima[feature_cols]  # Features
    #y = pima.label  # Target variable
    #X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,
    #                                                    random_state=1)  # 70% training and 30% test


    pass

def Q15(data):
    parameters = {'max_depth': ('linear', 'rbf'), 'C': [1, 10]}
    return df


if __name__ == '__main__':
    df =  pd.read_csv('train_clean.csv.csv')

    # df['risk'] = df['risk'].map(dict(High=1, Low=-1))
    # df['sex'] = df['sex'].map(dict(F=1, M=-1))
    # df['is_army'] = df['is_army'].map(dict(FALSE=-1, TRUE=1)
    # )
    df = prepare.normalize_data(df)
    df = prepare.create_number_convention(df)
    # Q12(df)
    # Q13(data=df)
        df = Q15(df)


