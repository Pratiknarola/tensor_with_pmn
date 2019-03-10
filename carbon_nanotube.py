import pandas as pd
import numpy as np
import sklearn
from sklearn import linear_model
from sklearn.utils import shuffle
import matplotlib.pyplot as pyplt
import pickle
from matplotlib import style


data = pd.read_csv('last.csv', sep=',')
high = pd.read_csv('high.csv', sep=';')
low = pd.read_csv('low.csv')

predict = 'r'

x = np.array(high)
y = np.array(low)

x_train , x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.1)

best = 0
for _ in range(10000):
    x_train , x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.1)

    linear = linear_model.LinearRegression()

    linear.fit(x_train, y_train)
    acc = linear.score(x_test, y_test)
    print(acc)

    if acc > best:
        best = acc
        with open('carbonmodel.pickle', 'wb') as f:
            pickle.dump(linear, f)
            print("===================================================================")
            print('Dumped ' + str(acc))

pickle_in = open('carbonmodel.pickle', 'rb')
linear = pickle.load(pickle_in)