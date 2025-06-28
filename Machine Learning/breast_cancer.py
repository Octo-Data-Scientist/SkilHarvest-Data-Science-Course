# Importing libraries
import numpy as np
import pandas as pd
import sklearn
from sklearn import datasets
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import StandardScaler

# Selecting a datasets
print(dir(sklearn.datasets), "\n")
data = datasets.load_breast_cancer()
print(data, "\n")

# Splitting the datasets in feature matrix and target variable
m = data.data # feature matrix
n = data.target # target variable
print(m, "\n")
print(n, "\n")

# Splitting into train-test data
m_train, m_test, n_train, n_test = train_test_split(m, n, test_size = 0.3, random_state = 42)

# Model Training
# Classification Algorithms
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

# Define Classifiers
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "SVM": SVC(),
    "KNN": KNeighborsClassifier( n_neighbors = 5),
    "Naive Bayes": GaussianNB(),
    "Decision Tree": DecisionTreeClassifier(),
    "Random Forest": RandomForestClassifier(),
    "Gradient Boosting": GradientBoostingClassifier()
}

for name, model in models.items():
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", model)
    ])

    # Training the Model
    pipeline.fit(m_train, n_train)

    # Evaluating training score
    train_accuracy = pipeline.score(m_train, n_train)

    # Make Predictions
    predictions = pipeline.predict(m_test)
    accuracy = accuracy_score(n_test, predictions)
    cv_scores = cross_val_score(pipeline, m, n, cv = 5).mean()
    print(f"{name:<20s} | Train Accuracy: {train_accuracy: .2f} | Test Accuracy: {accuracy:.2f} | CV Accuracy: {cv_scores:.2f}", "\n")