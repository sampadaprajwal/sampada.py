import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    roc_curve,
    roc_auc_score
)
data = pd.read_excel("C:/Users/Prajwal/Downloads/gene_expression_cancer_classification.xlsx")
print(data.head()) 
print(data.columns)
print(data.shape)
print(data.info())
print(data.isnull().sum())
X = data.drop(columns=["Sample_ID", "Target"])
y = data["Target"]
print(X.columns)
print(y.name)
print(y.unique())
encoder= LabelEncoder()
y= encoder.fit_transform(y)
print(y)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(X_train.shape)
print(X_test.shape)
#Feature selection and model training
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier(n_estimators=100, random_state= 42)
model.fit(X_train, y_train)
importance= model.feature_importances_
features = X.columns
for feature, score in zip(features, importance):
    print(feature, score)
    # Prediction
y_pred = model.predict(X_test)
print(y_pred)
#Compare actual versus predicted
print("Actual:", y_test)
print("Predicted:", y_pred)
#predict the probability of each class (how confident the model is)
y_prob = model.predict_proba(X_test)[:, 1]
#confusion matrix
cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(cm)
# Accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
#classification report
print(classification_report(y_test, y_pred))
#ROC auc curve
fpr, tpr, thresholds = roc_curve(y_test, y_prob, pos_label=1)
plt.plot(fpr, tpr)
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.show()