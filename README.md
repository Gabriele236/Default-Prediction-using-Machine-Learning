# Default-Prediction-using-Machine-Learning
Questo codice è un esempio di classificazione utilizzando l'algoritmo K-Nearest Neighbors (KNN) su un dataset CSV. KNN è un algoritmo di machine learning di tipo supervisionato utilizzato per la classificazione e la regressione.

Librerie utilizzate:

numpy: Libreria per supportare operazioni matematiche su array multidimensionali.
pandas: Libreria per la manipolazione e l'analisi dei dati.
sklearn.preprocessing.StandardScaler: Classe per eseguire la standardizzazione delle caratteristiche (feature scaling).
imblearn.over_sampling.RandomOverSampler: Classe per l'oversampling dei dati in caso di classi sbilanciate.
sklearn.neighbors.KNeighborsClassifier: Classe per l'implementazione dell'algoritmo KNN.
sklearn.metrics.classification_report: Funzione per calcolare il report di classificazione contenente diverse metriche.
Fase di preparazione dei dati:

Lettura del file CSV: Il codice legge il file CSV chiamato "Dataset.csv" utilizzando pandas e imposta ";" come separatore dei valori e la prima riga come header dei nomi delle colonne. Il dataset viene memorizzato nel DataFrame "df".
Splitting del dataset: Il dataset viene suddiviso in tre parti: training set (60% del dataset originale), validation set (20% del dataset originale) e testing set (20% del dataset originale). Questo viene fatto utilizzando la funzione np.split su un campione casuale del DataFrame "df".
Preparazione dei dati per l'addestramento e la valutazione:

Scaling e Oversampling: Viene definita la funzione scale_dataset per effettuare lo scaling dei dati e, se richiesto, l'oversampling. In particolare, la funzione esegue le seguenti operazioni:
Divide il DataFrame in matrice delle features "X" e vettore dei target "y".
Utilizza la classe StandardScaler di scikit-learn per standardizzare le features nell'intervallo (media = 0, deviazione standard = 1).
Se oversample=True, utilizza la classe RandomOverSampler di imbalanced-learn per bilanciare le classi dell'insieme di addestramento, aumentando il numero di campioni delle classi minoritarie fino a raggiungere il numero di campioni della classe maggioritaria.
Restituisce i dataset scalati e, se applicato, l'oversampling.
Addestramento del modello:

Viene creato un modello di classificazione KNN utilizzando la classe KNeighborsClassifier di scikit-learn, con n_neighbors=5, che indica che il modello considererà i 5 vicini più prossimi per effettuare una previsione.
Il modello viene addestrato sul training set utilizzando il metodo fit, passando le matrici di feature "X_train" e il vettore di target "y_train".
Valutazione del modello:

Viene utilizzato il modello addestrato per effettuare previsioni sul testing set utilizzando il metodo predict, ottenendo il vettore di previsioni "y_pred".
Viene calcolato e stampato un report di classificazione utilizzando la funzione classification_report di scikit-learn, che mostra diverse metriche di valutazione (come precision, recall, f1-score e support) per ogni classe presente nel testing set.
Nota importante: Nel caso in cui si voglia utilizzare questo codice su un altro dataset, assicurarsi di sostituire il nome del file "Dataset.csv" con il nome del proprio dataset. Inoltre, è possibile personalizzare alcuni iperparametri dell'algoritmo KNN come il numero di vicini "n_neighbors" o applicare altre tecniche di preprocessing dei dati, a seconda delle esigenze specifiche del problema.
