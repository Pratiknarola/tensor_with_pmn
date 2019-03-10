import pandas as pd
import sklearn
from sklearn.neighbors import KNeighborsClassifier
from sklearn import linear_model, preprocessing

data = pd.read_csv('final.csv')

le = preprocessing.LabelEncoder()
temp = list(data['temp'])
nausea = le.fit_transform(list(data['nausea']))
lumber_pain = le.fit_transform(list(data['lumber_pain']))
urine_pushing = le.fit_transform(list(data['urine_pushing']))
micturition = le.fit_transform(list(data['micturition']))
itch = le.fit_transform(list(data['itch']))
u_bladder = le.fit_transform(list(data['u_bladder']))
nephritis = le.fit_transform(list(data['nephritis']))


predict1 = 'u_bladder'
predict2 = 'nephritis'

x = list(zip(temp,nausea,lumber_pain,urine_pushing,micturition,itch))
y = list(u_bladder)
z = list(nephritis)

x_train,x_test,y_train,y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.1)

model = KNeighborsClassifier(n_neighbors=9)

model.fit(x_train, y_train)
acc1 = model.score(x_test, y_test)
print('Accuracy: ' + str(acc1))

predicted = model.predict(x_test)

for i in range(len(x_test)):
    print('Predicted: ',predicted[i], 'data:', x_test[i], 'actual:',y_test[i])

x_train,x_test,z_train,z_test = sklearn.model_selection.train_test_split(x, z, test_size=0.1)

model.fit(x_train, z_train)
acc2 = model.score(x_test, z_test)
print('Accuracy: ' + str(acc2))

predicted = model.predict(x_test)

for j in range(len(x_test)):
    print('Predicted: ',predicted[j], 'data:', x_test[j], 'actual:',z_test[j])