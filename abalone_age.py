import sklearn
from sklearn.utils import shuffle
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
import numpy as np
from sklearn import linear_model, preprocessing


data = pd.read_csv('datasets/abalone.data')

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

x = list(zip(sex,length,diameter,height,whole_weight,shucked_weight,viscera_weight,shell_weight))
y = list(rings)


x_train,x_test,y_train,y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.1)

model = KNeighborsClassifier(n_neighbors=9)

model.fit(x_train, y_train)
acc = model.score(x_test, y_test)
print(acc)
predicted = model.predict(x_test)
print('================================================================================================')
if acc > 0.3:
    abalone = open('abalone.txt', 'w+')
    for x in range(len(x_test)):
        print('Predicted:', predicted[x], 'data:', x_test[x], 'actual:', y_test[x])
        abalone.write('Predicted: ' + str(predicted[x]) + ' data: ' +  str(x_test[x]) +  ' actual: ' + str(y_test[x])+'\n')
print(acc)
