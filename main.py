#LIBRERIE
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import RandomOverSampler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report
#LETTURA FILE CSV
df = pd.read_csv("Dataset.csv",sep=";",header=0)
#SPLITTING IN TRAINING VALIDATION E TESTING DATASET
train, valid, test = np.split(df.sample(frac=1), [int(0.6*len(df)), int(0.8*len(df))])
#SCALING E OVERSAMPLE
def scale_dataset(dataframe, oversample=False):
  X = dataframe[dataframe.columns[:-1]].values
  y = dataframe[dataframe.columns[-1]].values

  scaler = StandardScaler()
  X = scaler.fit_transform(X)

  if oversample:
    ros = RandomOverSampler()
    X, y = ros.fit_resample(X, y)

  data = np.hstack((X, np.reshape(y, (-1, 1))))

  return data, X, y

train, X_train, y_train = scale_dataset(train, oversample=True)
valid, X_valid, y_valid = scale_dataset(valid, oversample=False)
test, X_test, y_test = scale_dataset(test, oversample=False)
#APPLICHIAMO MODELLO
knn_model = KNeighborsClassifier(n_neighbors=5)
knn_model.fit(X_train, y_train)
#CALCOLO DEL VALORE E CLASSIFICATION REPORT
y_pred = knn_model.predict(X_test)
print(classification_report(y_test, y_pred))