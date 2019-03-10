import pandas as pd
import sklearn
from sklearn.neighbors import KNeighborsClassifier
from sklearn import linear_model, preprocessing
from matplotlib import pyplot as plt

data = pd.read_csv(na_values='?', filepath_or_buffer= 'datasets/diabetic_data.csv')

le = preprocessing.LabelEncoder()
race = le.fit_transform(list(data['race']))
gender = le.fit_transform(list(data['gender']))
age = le.fit_transform(list(data['age']))
weight = le.fit_transform(list(data['weight']))
time_in_hospital = le.fit_transform(list(data['time_in_hospital']))
medical_specialty = le.fit_transform(list(data['medical_specialty']))
num_lab_procedures = le.fit_transform(list(data['num_lab_procedures']))
num_procedures = le.fit_transform(list(data['num_procedures']))
num_medications = le.fit_transform(list(data['num_medications']))
number_outpatient = le.fit_transform(list(data['number_outpatient']))
number_emergency = le.fit_transform(list(data['number_emergency']))
number_inpatient = le.fit_transform(list(data['number_inpatient']))
diag_1 = le.fit_transform(list(data['diag_1']))
diag_2 = le.fit_transform(list(data['diag_2']))
diag_3 = le.fit_transform(list(data['diag_3']))
number_diagnoses = le.fit_transform(list(data['number_diagnoses']))
max_glu_serum = le.fit_transform(list(data['max_glu_serum']))
A1Cresult = le.fit_transform(list(data['A1Cresult']))
metformin = le.fit_transform(list(data['metformin']))
repaglinide = le.fit_transform(list(data['repaglinide']))
nateglinide = le.fit_transform(list(data['nateglinide']))
chlorpropamide = le.fit_transform(list(data['chlorpropamide']))
glimepiride = le.fit_transform(list(data['glimepiride']))
acetohexamide = le.fit_transform(list(data['acetohexamide']))
glipizide = le.fit_transform(list(data['glipizide']))
glyburide = le.fit_transform(list(data['glyburide']))
tolbutamide = le.fit_transform(list(data['tolbutamide']))
pioglitazone = le.fit_transform(list(data['pioglitazone']))
rosiglitazone = le.fit_transform(list(data['rosiglitazone']))
acarbose = le.fit_transform(list(data['acarbose']))
miglitol = le.fit_transform(list(data['miglitol']))
troglitazone = le.fit_transform(list(data['troglitazone']))
tolazamide = le.fit_transform(list(data['tolazamide']))
examide = le.fit_transform(list(data['examide']))
citoglipton = le.fit_transform(list(data['citoglipton']))
insulin = le.fit_transform(list(data['insulin']))
glyburide_metformin = le.fit_transform(list(data['glyburide-metformin']))
glipizide_metformin = le.fit_transform(list(data['glipizide-metformin']))
glimepiride_pioglitazone = le.fit_transform(list(data['glimepiride-pioglitazone']))
metformin_rosiglitazone = le.fit_transform(list(data['metformin-rosiglitazone']))
metformin_pioglitazone = le.fit_transform(list(data['metformin-pioglitazone']))
change = le.fit_transform(list(data['change']))
diabetesMed = le.fit_transform(list(data['diabetesMed']))
readmitted = le.fit_transform(list(data['readmitted']))

predict1 = 'diabetesMed'
predict2 = 'readmitted'

plt.plot(diag_1, diabetesMed, 'ro')
plt.xlabel('gender')
plt.ylabel('diab')
plt.show()


'''
x = list(zip(race,gender,age,weight,time_in_hospital,medical_specialty,num_lab_procedures,diag_1,num_procedures,num_medications,
             number_outpatient,number_emergency,number_inpatient,diag_2,diag_3,number_diagnoses,max_glu_serum,A1Cresult,
             metformin,repaglinide,nateglinide,chlorpropamide,glimepiride,acetohexamide,glipizide,glyburide,tolbutamide,
             pioglitazone,rosiglitazone,acarbose,miglitol,troglitazone,tolazamide,examide,citoglipton,insulin,glyburide_metformin,
             glipizide_metformin,glimepiride_pioglitazone,metformin_rosiglitazone,metformin_pioglitazone,change))
y = list(diabetesMed)
z = list(readmitted)

x_train,x_test,y_train,y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.1)

model = KNeighborsClassifier(n_neighbors=23)

model.fit(x_train, y_train)
acc1 = model.score(x_test, y_test)
print('Accuracy: ' + str(acc1))

predicted = model.predict(x_test)

for i in range(len(x_test)):
    if i > 15:
        break
    print('Predicted: ',predicted[i], 'data:', x_test[i], 'actual:',y_test[i])

'''
'''
x_train,x_test,z_train,z_test = sklearn.model_selection.train_test_split(x, z, test_size=0.1)

model2 = KNeighborsClassifier(n_neighbors=13)
model2.fit(x_train, z_train)
acc2 = model2.score(x_test, z_test)
print('Accuracy: ' + str(acc2))
'''