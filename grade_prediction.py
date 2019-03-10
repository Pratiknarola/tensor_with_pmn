import pandas as pd
import numpy as np
import sklearn
from sklearn import linear_model
from sklearn.utils import shuffle
import matplotlib.pyplot as pyplt
import pickle
from matplotlib import style


data = pd.read_csv('final.csv', sep=',')

#print(data.head())

#data = data[['G1', 'G2', 'G3', 'studytime', 'failures', 'absences']]

#print(data.head())

predict = "G3"

x = np.array(data.drop([predict], 1))
y = np.array(data[predict])

x_train , x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.1)

'''best = 0
for _ in range(10000):
    x_train , x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.1)

    linear = linear_model.LinearRegression()

    linear.fit(x_train, y_train)
    acc = linear.score(x_test, y_test)
    print(acc)

    if acc > best:
        best = acc
        with open('studentmodel.pickle', 'wb') as f:
            pickle.dump(linear, f)
            print('Dumped ' + str(acc))'''

pickle_in = open('trained_models/studentmodel.pickle', 'rb')
linear = pickle.load(pickle_in)

#print('Co: ' , linear.coef_)
#print('Intercept: ', linear.intercept_)
predictions = linear.predict(x_test)
for x in range(len(predictions)):
    print('predection:',predictions[x],'xtest:', x_test[x],'ytest:', y_test[x])
#print(acc)

p = 'famsize'
style.use('ggplot')
pyplt.scatter(data[p], data['G3'])
pyplt.xlabel(p)
pyplt.ylabel('Final grade')
pyplt.show()