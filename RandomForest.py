# Importing the libraries
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn import metrics
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

# Importing the dataset
dataset = pd.read_csv('heartdiseases.csv')
X = dataset.iloc[:,:-1].values
y = dataset.iloc[:, 13].values

#handling missing data
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
imputer.fit(X[:, 11:13])
X[:, 11:13] = imputer.transform(X[:, 11:13])

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 101)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

#EXPLORING THE DATASET
dataset.num.value_counts()

# Fitting Random Forest Classifier to the Training set with hyperparameter tuning using grid search
# Define the hyperparameters to tune
param_grid = {'n_estimators': [10, 50, 100, 200, 300], 'max_depth': [3, 5, 7, 9, 11], 'max_features': ['sqrt', 'log2', None]}

# Create a Random Forest Classifier object
classifier = RandomForestClassifier(random_state=0)

# Create a grid search object
grid_search = GridSearchCV(estimator=classifier, param_grid=param_grid, cv=5)

# Fit the grid search object to the training data
grid_search.fit(X_train, y_train)

# Print the best parameters and score
print("Best parameters: ", grid_search.best_params_)
print("Best score: ", grid_search.best_score_)

# Predicting the Test set results using the best estimator
y_pred = grid_search.best_estimator_.predict(X_test)

# Accuracy Score
RF_acc = metrics.accuracy_score(y_test, y_pred)
print("Accuracy of RF :",metrics.accuracy_score(y_test, y_pred))
# Confusion Matrix
tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
# Model Specificity: true negative rate
specificity = tn/(tn + fp)
print('Specificity of RF : ', specificity)
# Model Sensitivity: true positive rate
sensitivity = tp/(tp + fn)
print('Sensitivity of RF : ', sensitivity)
