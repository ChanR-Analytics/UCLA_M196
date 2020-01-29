import pandas as pd
from os import getcwd, listdir
from hate_speech_detection.src.python.hate_speech_oop import HatebaseTwitter


# Getting Davidson's Hatbase Twitter Data
hb_path = getcwd() + "/hate_speech_detection/data/HatebaseTwitter"

# Initializing the HatebaseTwitter Class
hb = HatebaseTwitter(hb_path)

# Performing an Exploratory Data Analysis of the Hatebase Twitter Dataset
hb.eda()

# Extracting the Tweet TF-IDF, POS TF-IDF, and Other Tweet Features into a Multidimensional Data Matrix
features = hb.features()

from sklearn.model_selection import StratifiedKFold, GridSearchCV, train_test_split
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

X = pd.DataFrame(features)
X.shape

y = hb.df['class']

dim_reduce = SelectFromModel(LogisticRegression(solver='liblinear', class_weight='balanced', C=0.04, penalty='l1'))
X_ = dim_reduce.fit_transform(X, y)
X_.shape

X_train, X_test, Y_train, Y_test = train_test_split(X_, y, test_size=0.15, stratify=y)

model = RandomForestClassifier(n_estimators=200, criterion='entropy')

model.fit(X_train, Y_train)

Y_pred = model.predict(X_test)
Y_pred.shape
accuracy_score(Y_test, Y_pred)

print(classification_report(Y_test, Y_pred))

model_2 = XGBClassifier()
model_2.fit(X_train, Y_train)
Y_pred_2 = model_2.predict(X_test)

accuracy_score(Y_test, Y_pred_2)

print(classification_report(Y_test, Y_pred_2))
