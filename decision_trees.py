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


