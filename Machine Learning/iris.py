# Importing libraries
import pandas as pd
import sklearn
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Printing out the dummy datasets in sklearn
print(dir(sklearn.datasets), "\n")

# loading the iris dataset
iris = datasets.load_iris()
print(iris, "\n")

# converting the iris dataset into a dataframe
data = pd.DataFrame(iris.data, columns = iris.feature_names)
data["target"] = iris.target

# creating a new feature that includes the name of the flower
tn = {0:"setosa", 1:"versicolor", 2:"virginica"}
data["target_name"] = data.target.map(tn)

# Data Cleaning
data.drop("target_name", axis = 1, inplace = True)
print(data, "\n")

# Splitting the data into feature matrix (input data) and target variables (output variables)
m = data.drop("target", axis = 1)   # feature matrix
n = data.target

print(m, "\n")
print(n, "\n")

# Splitting data into train test split
m_train, m_test, n_train, n_test = train_test_split(m, n, test_size = 0.2, random_state = 42)

# Model Training
# Creating a AdaBoost Model
from sklearn.ensemble import AdaBoostClassifier
ab_model = AdaBoostClassifier(n_estimators = 200, algorithm = 'SAMME')

# Training the model
ab_model.fit(m_train, n_train)

# Evaluating the training set
print(f"Training Score: {ab_model.score(m_train, n_train) * 100:.1f}", "\n")

# Make Predictions
ab_pred = ab_model.predict(m_test)

# Evaluating the training set
print(f"Accuracy: {accuracy_score(n_test, ab_pred) * 100:.1f}", "\n")
print(f"Confusion matrix: {confusion_matrix(n_test, ab_pred)}", "\n")
print(classification_report(n_test, ab_pred))