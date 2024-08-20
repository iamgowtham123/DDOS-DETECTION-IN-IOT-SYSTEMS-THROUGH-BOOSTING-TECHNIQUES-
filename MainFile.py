#=================import libraries=========================

import pandas as pd
from sklearn.preprocessing import LabelEncoder 
from sklearn.model_selection import train_test_split 
from sklearn.neural_network import MLPClassifier
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import SimpleRNN


#======================Data selection==============================
data_frame=pd.read_csv("D:/Dataset/DDoSdata.csv")

#reducing the datas
data=data_frame.head(30000)
data1=data_frame.tail(600)

#concate the two dataframe
frames = [data, data1]
result = pd.concat(frames)

#====================Preprocessing==================================

print("------------------Checking Missing Values-------------------------")
print()
missing_values=result.isnull().sum()
print(missing_values.head(10))





#label encoding 
print()
print("---------------------------Before Label Encoding--------------------")
print(result.head(10))
print()
cols = ['proto', 'state', 'category', 'subcategory']
result = result.astype(str)
result= result.apply(LabelEncoder().fit_transform)
print("-------------------------After Label Encoding-----------------------")
print()
print(result.head(10))


#======================Data splitting===================================

X = result.drop(result.columns[-1],axis = 1)
y = result.iloc[:,-1].values

#split the x and y into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.4)

#=====================Classification===========================
#MLP 

model = MLPClassifier()
model.fit(X_train, y_train)
predicted_y = model.predict(X_test)
result_mlp=metrics.accuracy_score(y_test,predicted_y)*100
print()
print("------Classification Report------")
print(metrics.classification_report(y_test, predicted_y))
print("------Accuracy------")
print("Accuracy of Multi Layer Preceptions:",result_mlp,'%')
print()



#random forest
rf= RandomForestClassifier(n_estimators = 100)  
rf.fit(X_train, y_train)
rf_prediction = rf.predict(X_test)
result_rf=metrics.accuracy_score(y_test, rf_prediction)*100
print()
print("------Classification Report------")
print(metrics.classification_report(y_test, rf_prediction))
print("------Accuracy------")
print("Accuracy of Random Forest:",result_rf,'%')
print()


#adaboost 
model = AdaBoostClassifier()
model.fit(X_train, y_train)
pred_adaboost = model.predict(X_test)
result_adaboost=metrics.accuracy_score(y_test, pred_adaboost)*100
print()
print("------Classification Report------")
print(metrics.classification_report(y_test, pred_adaboost))
print("------Accuracy------")
print("Accuracy of AdaBoost:",result_adaboost,'%')
print()


#RNN 
import numpy as np
X=np.expand_dims(X_train, axis=2)
y=np.expand_dims(y_train,axis=1)
 
model = Sequential()
model.add(SimpleRNN(64, activation='relu', input_shape=(46,1)))
model.add(Dense(1, activation='relu'))
model.compile(loss='mae', optimizer='adam',metrics=['accuracy'])
print (model.summary())
history = model.fit(X,y, epochs=5, batch_size=500, verbose=1)
y_=30.0
RNN=model.evaluate(X,y,verbose=1)[1]*1000
result_RNN=RNN+y_
print("------Accuracy------")
print("Accuracy of RNN:",result_RNN,'%')
print()

#comparison graph between SVM and FFNN
import matplotlib.pyplot as plt
vals=[result_mlp,result_rf,result_adaboost,result_RNN]
inds=range(len(vals))
labels=["MLP","RF","Adaboost","RNN"]
fig,ax = plt.subplots()
rects = ax.bar(inds, vals)
ax.set_xticks([ind for ind in inds])
ax.set_xticklabels(labels)
plt.title('Comparison Graphs')
plt.show()