# Importing the libraries
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from sklearn import metrics
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

# Importing the dataset
dataset = pd.read_csv('heartdiseases.csv')
X = dataset.iloc[:,:-1].values
y = dataset.iloc[:, 13].values

# handling missing data
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
X[:,11:13] = imputer.fit_transform(X[:,11:13])

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 101)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Fitting Decision Tree Classification to the Training set
classifier = DecisionTreeClassifier(random_state=8)

# Hyperparameter Tuning
params = {'criterion': ['gini', 'entropy'], 'max_depth': range(1, 11)}
grid_search = GridSearchCV(estimator=classifier, param_grid=params, cv=5, n_jobs=-1)
grid_search.fit(X_train, y_train)
best_params = grid_search.best_params_

# Fitting Decision Tree Classification with the best hyperparameters
classifier = DecisionTreeClassifier(random_state=8, **best_params)
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Accuracy
DT_acc = metrics.accuracy_score(y_test, y_pred)
print("Accuracy of DT :",metrics.accuracy_score(y_test, y_pred))
# Confusion Matrix
tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
# Model Specificity: true negative rate
specificity = tn/(tn + fp)
print('Specificity of DT : ', specificity)
# Model Sensitivity: true positive rate
sensitivity = tp/(tp + fn)
print('Sensitivity of DT : ', sensitivity)
