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

from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from xgboost import XGBClassifier
from ngboost import ngboost
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

X = pd.DataFrame(features)
X.shape 
