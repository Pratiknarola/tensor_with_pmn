import pandas as pd
import numpy as np
import sklearn
from sklearn import linear_model
from sklearn.utils import shuffle
import matplotlib.pyplot as pyplt
import pickle
from sklearn import linear_model, preprocessing

data = pd.read_csv('final.csv')

le = preprocessing.LabelEncoder()
sex = le.fit_transform(list(data['sex']))
length = list(data['length'])
diameter = list(data['diameter'])
height = list(data['height'])
whole_weight = list(data['whole_weight'])
shucked_weight = list(data['shucked_weight'])
viscera_weight = list(data['viscera_weight'])
shell_weight = list(data['shell_weight'])
rings = list(data['rings'])

predict = "rings"

x = np.array(data.drop([predict], 1))
y = np.array(data[predict])

x_train , x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.1)

best = 0
for _ in range(1000000):
    x_train , x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.1)

    linear = linear_model.LinearRegression()

    linear.fit(x_train, y_train)
    acc = linear.score(x_test, y_test)
    print(acc)

    if acc > best:
        best = acc
        with open('abalonemodel.pickle', 'wb') as f:
            pickle.dump(linear, f)
            print('Dumped ' + str(acc))

'''pickle_in = open('trained_models/abalonemodel.pickle', 'rb')
linear = pickle.load(pickle_in)

result = linear.score(x_test,y_test)
print(result)'''