# Import necessary libraries
from sklearn.datasets import load_iris
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Download the Iris dataset from a URL
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
names = ["sepal_length", "sepal_width", "petal_length", "petal_width", "class"]
dataset = pd.read_csv(url, names=names)

# Separate features (X) from target variable (y)
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a KNN classifier model with 5 neighbors
knn = KNeighborsClassifier(n_neighbors=5)

# Train the model
knn.fit(X_train, y_train)

# Choose a specific test variable from the dataset
test_variable = [7.1, 3.3, 5.8, 2.2]  # Example test variable

# Make a prediction for the test variable
prediction = knn.predict([test_variable])
print("Predicted class for test variable:", prediction[0])

# Define a color map based on flower species
cmap = plt.colormaps["viridis"]  # Access colormap using recommended method

# Create a numerical mapping for flower species (optional)
# This step is optional, but it allows for color coding based on categories.
# You can define a dictionary or use other mapping techniques.
unique_classes = np.unique(y)  # Get unique flower species categories
color_map = dict(zip(unique_classes, range(len(unique_classes))))  # Map categories to numerical values

# Create a scatter plot with color mapping
plt.figure(figsize=(8, 6))
if color_map is not None:  # Use color mapping if defined
    colors = [color_map[cl] for cl in y]  # Convert categories to numerical values for color mapping
else:
    colors = np.random.rand(len(y))  # Use random colors if no mapping defined
plt.scatter(X[:, 0], X[:, 1], c=colors, edgecolors="black")  # Sepal length vs Sepal width
plt.xlabel("Sepal length (cm)")
plt.ylabel("Sepal width (cm)")
plt.title("Iris Sepal Length vs Sepal Width")

# Plot the test variable (optional)
plt.scatter(test_variable[0], test_variable[1], marker='o', color='red', label='Test Variable')
plt.legend()

plt.show()
