import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import sklearn
print(pd.__version__)
print(np.__version__)
print(sklearn.__version__)
print(matplotlib.__version__)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve, roc_auc_score