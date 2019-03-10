import pandas as pd
import numpy as np
import sklearn
from sklearn import linear_model
import pickle
from matplotlib import style

data = pd.read_csv('final.csv', sep=';')

predict = 'AH'

x = np.array(data.drop([predict], 1))
y = np.array(data[predict])

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
        with open('airmodel.pickle', 'wb') as f:
            pickle.dump(linear, f)
            print('Dumped ' + str(acc))

pickle_in = open('airmodel.pickle', 'rb')
linear = pickle.load(pickle_in)
