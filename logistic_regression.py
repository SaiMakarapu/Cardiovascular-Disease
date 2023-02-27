# Importing the libraries
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn import metrics
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

# Importing the dataset
dataset = pd.read_csv('heartdiseases.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 13].values

# Handling missing data
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
imputer.fit(X[:, 11:13])
X[:, 11:13] = imputer.transform(X[:, 11:13])

# Splitting dataset into training set and test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 101)

# Feature scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)

# Define the hyperparameters to tune
param_grid = {'penalty': ['l1', 'l2'], 'C': [0.001, 0.01, 0.1, 1, 10, 100]}

# Create a logistic regression object
classifier = LogisticRegression(random_state=0, solver='liblinear')

# Create a grid search object
grid_search = GridSearchCV(estimator=classifier, param_grid=param_grid, cv=5)

# Fit the grid search object to the training data
grid_search.fit(X_train, y_train)

# Print the best parameters and score
print("Best parameters: ", grid_search.best_params_)
print("Best score: ", grid_search.best_score_)

# Predict the test set results using the best estimator
y_pred = grid_search.best_estimator_.predict(X_test)

# Accuracy
logreg_acc = metrics.accuracy_score(y_test, y_pred)
print("Accuracy of LR :",metrics.accuracy_score(y_test, y_pred))
# Confusion Matrix
tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
# Model Specificity: true negative rate
specificity = tn/(tn + fp)
print('Specificity of LR : ', specificity)
# Model Sensitivity: true positive rate
sensitivity = tp/(tp + fn)
print('Sensitivity of LR : ', sensitivity)
