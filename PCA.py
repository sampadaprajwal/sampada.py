import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
#Data loading
data = pd.read_excel(r"C:\Users\Prajwal\Downloads\gene_expression_cancer_classification.xlsx")
#Inspect the data
print(data.head())
print(data.shape)
print(data.columns)
print(data.info())
#Check missing values
print(data.isnull().sum())
#expression data
X = data.drop(columns=["Sample_ID", "Target"])
print(X.shape)
print(X.head())
#Standardization
scaler= StandardScaler()
X_scaled= scaler.fit_transform(X)
print(X_scaled.shape)
#Apply pca
pca= PCA(n_components=2)
X_pca= pca.fit_transform(X_scaled)
print(X_pca.shape)
#Check explaoned variance
print("Explained variance:", pca.explained_variance_ratio_)
print("Total variance:", sum(pca.explained_variance_ratio_))
# PCA visualization
plt.scatter(X_pca[:, 0], X_pca[:, 1])

plt.xlabel("PC1")
plt.ylabel("PC2")
plt.title("PCA of Gene Expression Data")

plt.show()