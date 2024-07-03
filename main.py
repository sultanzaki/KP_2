import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import model_selection
from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import r2_score
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import BaggingClassifier
from sklearn.pipeline import make_pipeline

df = pd.read_csv('dataset.csv');
df = df.dropna()

x = df.drop(['label'], axis=1)
y = df['label']
encode = preprocessing.LabelEncoder()
y = encode.fit_transform(y)

x_train, x_test, y_train, y_test = model_selection.train_test_split(x, y)

pipe = make_pipeline(preprocessing.StandardScaler(), RandomForestClassifier(n_estimators=10))
bag_model = BaggingClassifier(estimator=pipe,
                              n_estimators=200,
                              oob_score=True,
                              random_state=10,
                              max_samples=0.7)
bag_model.fit(x_train, y_train)

y_pred = bag_model.predict(x_test)

print(r2_score(y_test, y_pred))