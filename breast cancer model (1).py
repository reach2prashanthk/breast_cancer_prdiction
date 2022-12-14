#!/usr/bin/env python
# coding: utf-8



import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn.datasets
from sklearn.model_selection import train_test_split


breast_caner_dataset=sklearn.datasets.load_breast_cancer()
breast_caner_dataset

print(breast_caner_dataset)


data_frame=pd.DataFrame(breast_caner_dataset.data,columns=breast_caner_dataset.feature_names)

data_frame


data_frame['label']=breast_caner_dataset.target
data_frame

data_frame.tail()



data_frame.shape



data_frame.info()


data_frame.describe()


data_frame['label'].value_counts()
#1 beningn
# 0 malignant


data_frame.groupby('label').mean()

X=data_frame.drop(columns='label',axis=1)
Y=data_frame['label']

X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,random_state=2)

print(X.shape,X_train.shape,X_test.shape)

from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()
X_train_std=scaler.fit_transform(X_train)
X_test_std=scaler.fit_transform(X_test)
print(X_train_std)


import tensorflow as tf
tf.random.set_seed(3)
from tensorflow.keras import layers




model=tf.keras.Sequential([tf.keras.layers.Flatten(input_shape=(30,)),
                        tf.keras.layers.Dense(20,activation='relu'),
                        tf.keras.layers.Dense(2,activation='sigmoid')])


model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])



history=model.fit(X_train_std,Y_train,validation_split=0.1,epochs=10)


#visualising
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.xlabel('accuracy')
plt.ylabel('epoch')
plt.legend(['train','validation_data'],loc='lower right')


#accuracy of model on test data


loss,accuracy=model.evaluate(X_test_std,Y_test)
print(accuracy)


print(X_test_std.shape)
print(X_test_std[0])


Y_pred=model.predict(X_test_std)
print(Y_pred.shape)
print(Y_pred[0])


print(X_test_std)


print(Y_pred)


#model.predict() gives prediction probabailty for each class



my_list=[10,20,30]
index_of_max_value=np.argmax(my_list)
print(my_list)
print(index_of_max_value)
#index of max value



Y_pred_labels=[np.argmax(i) for i in Y_pred]
Y_pred_labels



#building predictive system
input_data=(17.99,10.38,122.8,1001,0.1184,0.2776,0.3001,0.1471,0.2419,0.07871,1.095,0.9053,8.589,153.4,0.006399,0.04904,0.05373,0.01587,0.03003,0.006193,25.38,17.33,184.6,2019,0.1622,0.6656,0.7119,0.2654,0.4601,0.1189)
input_data_as_numpy_array=np.array(input_data)
input_data_reshaped=input_data_as_numpy_array.reshape(1,-1)
#standardizing  input data
input_data_std=scaler.transform(input_data_reshaped)
prediction=model.predict(input_data_std)
print(prediction)
prediction_label=[np.argmax(prediction)]
print(prediction_label)
if (prediction_label[0]==0):
    print('tumor is malignant')
else:
    print("tumpr is benign")

