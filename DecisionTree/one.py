# Import necessary libraries
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
import pandas as pd

# Download the Iris dataset from a URL
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
names = ["sepal_length", "sepal_width", "petal_length", "petal_width", "class"]
dataset = pd.read_csv(url, names=names)

# Separate features (X) from target variable (y)
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a decision tree classifier model
clf = DecisionTreeClassifier(max_depth=3, random_state=42)  # Limit depth for better visualization

# Train the model on the training data
clf.fit(X_train, y_train)

# Make predictions on the test data
y_pred = clf.predict(X_test)
print(y_pred)

# Evaluate model performance (accuracy in this case)
from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# Visualize the decision tree (optional)
from sklearn import tree
import matplotlib.pyplot as plt
plt.figure(figsize=(10, 7))
tree.plot_tree(clf, feature_names=names[:-1], class_names=dataset["class"].unique(), rounded=True, filled=True)
plt.show()
