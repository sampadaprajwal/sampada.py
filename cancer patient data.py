#import libraries
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import sklearn
#check the version of imported libraries
print(pd.__version__)
print(np.__version__)
print(sklearn.__version__)
print(matplotlib.__version__)
#sklearn builds and evalautes ML models so import some lib from sklearn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve, roc_auc_score
#Load the data from csv file
data = pd.read_csv(r"C:\Users\Prajwal\Documents\cancer data.csv")
#data inspection to check what is present in the data
print(data.head())
print(data.shape)
print(data.columns)
print(data.info())
#Feature- target seperation
#X= information about the patient
#y= the answer we want computer to predict
X = data.drop("Classification", axis=1)
y = data["Classification"]
print(X.columns)
print(y.name)
#data preprocessing, find the missing values
print(X.isnull().sum())
print(y.isnull().sum())
#split the data into train and test data
X_train, X_test, y_train, y_test= train_test_split(X,y, test_size=0.2, random_state=42)
#verify whether the data has divided properly or not , test data 20 percent, remaining train the model
print(X_train.shape)
print(X_test.shape)
#Feature scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
#verify whether all are on same scale or not
print(X_train.mean(axis=0))
print(X_train.std(axis=0))
#verify to check that scaling doesnot chnages the no of patients changes just the values
print(X_train.shape)
print(X_test.shape)
#feature selection
model= RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)
importance= model.feature_importances_
print(importance)
features = data.drop("Classification", axis=1).columns
for feature, score in zip(features, importance):
    print(feature, score)
    #prediction of data
    y_pred= model.predict(X_test)
    print(y_pred)
    #calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)
    print("Accuracy:", accuracy)
    #calculate confusion matrix
    cm= confusion_matrix(y_test, y_pred)
    print(cm)
    #classification report
    print(classification_report(y_test, y_pred))
    #ROC curve
    print(model.classes_)
    y_prob= model.predict_proba(X_test) [:, 1]
    fpr, tpr, thresholds = roc_curve(y_test, y_prob, pos_label=2)
plt.plot(fpr, tpr)
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.show()