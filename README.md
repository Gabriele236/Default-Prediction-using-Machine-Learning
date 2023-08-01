# K-Nearest Neighbors (K-NN) Algorithm with Hyperparameter Optimization
This Python code demonstrates the application of the K-Nearest Neighbors (K-NN) algorithm for classification using GridSearchCV for hyperparameter tuning and cross-validation. The dataset is read from a CSV file named "Dataset.csv," where the target variable is located in the last column.

# Requirements
numpy
pandas
scikit-learn (including KNeighborsClassifier, StandardScaler, GridSearchCV, and cross_val_score)
imbalanced-learn (imblearn) for oversampling
Dataset Format
The dataset should be in CSV format with the target variable in the last column and all other columns as features. The delimiter used in the CSV file should be a semicolon (";").

# Steps
Import Libraries: The necessary libraries, including numpy, pandas, scikit-learn, and imbalanced-learn, are imported.

Read Dataset: The code reads the dataset from "Dataset.csv" using pandas and stores it in the DataFrame "df."

Data Preprocessing: The function scale_dataset is defined to perform data preprocessing. It scales the feature data using StandardScaler and, if specified, applies RandomOverSampler to address class imbalance.

Hyperparameter Tuning for K-NN: GridSearchCV is used to find the best hyperparameters for the K-NN classifier. The hyperparameters to be explored are n_neighbors, weights, and metric.

Print Best Parameters and Cross-Validation Score for K-NN: The results of the hyperparameter tuning are printed, including the best parameters and the corresponding cross-validation score for the K-NN model.

Build Optimized K-NN Model: The best hyperparameters found from GridSearchCV are used to create an optimized K-NN model.

Cross-Validation with Optimized K-NN Model: The optimized K-NN model is evaluated using cross-validation, and the cross-validation scores as well as the mean accuracy are printed.

# Additional Notes
The code assumes that the dataset file "Dataset.csv" exists in the same directory as the code file.
The code includes the option to perform oversampling using RandomOverSampler. If the dataset is imbalanced, it is recommended to set oversample=True to improve model performance.
