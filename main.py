# LIBRERIE
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import RandomOverSampler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.metrics import classification_report
import tensorflow as tf
from sklearn.model_selection import KFold

# LETTURA FILE CSV
df = pd.read_csv("Dataset.csv", sep=";", header=0)

# SCALING E OVERSAMPLE
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

data, X, y = scale_dataset(df, oversample=True)

# APPLICAZIONE CROSS-VALIDATION CON K-NN CON GRID SEARCH
knn_model = KNeighborsClassifier()

# Definizione dei valori degli iperparametri da esplorare per K-NN
param_grid_knn = {
    'n_neighbors': [3, 5, 7, 9, 11],
    'weights': ['uniform', 'distance'],
    'metric': ['euclidean', 'manhattan'],
}

# Creazione dell'oggetto GridSearchCV per K-NN
grid_search_knn = GridSearchCV(knn_model, param_grid_knn, cv=5)

# Esecuzione della ricerca della griglia per K-NN
grid_search_knn.fit(X, y)

# Stampa dei risultati della ricerca per K-NN
print("Best parameters found for K-NN: ", grid_search_knn.best_params_)
print("Best cross-validation score for K-NN: {:.2f}".format(grid_search_knn.best_score_))

# Utilizzo dei migliori iperparametri per costruire il modello K-NN ottimizzato
best_knn_model = grid_search_knn.best_estimator_

# APPLICAZIONE CROSS-VALIDATION CON IL MODELLO K-NN OTTIMIZZATO
knn_cv_scores = cross_val_score(best_knn_model, X, y, cv=5)

# Stampa delle prestazioni medie per K-NN
print("K-NN Cross-Validation Scores:", knn_cv_scores)
print("K-NN Mean Accuracy:", np.mean(knn_cv_scores))


#Creazioni Rete Neurali
nn_model = tf.keras.Sequential([
    tf.keras.layers.Dense(32, activation = "relu", input_shape=(41,)),
    tf.keras.layers.Dense(32, activation = "relu",),
    tf.keras.layers.Dense(1, activation = "sigmoid")

])

nn_model.compile(optimizer=tf.keras.optimizers.Adam(0.001), loss='binary_crossentropy',
                  metrics=['accuracy'])



num_folds = 5
kf = KFold(n_splits = num_folds)

fold_scores = []
for train_index, test_index in kf.split(X):
    X_train, X_val = X[train_index], X[test_index]
    y_train, y_val = y[train_index], y[test_index]

    nn_model.fit(X_train, y_train, epochs=100, batch_size=32, verbose=0)
    _, accuracy = nn_model.evaluate(X_val, y_val, verbose=0)
    fold_scores.append(accuracy)

mean_accuracy = np.mean(fold_scores)

print(f"Mean CV Accurcy: {mean_accuracy}")



