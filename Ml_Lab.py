# Step 1: Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, ConfusionMatrixDisplay, classification_report

# Step 2: Load the Iris dataset
iris = datasets.load_iris()

# Step 3: Explore the dataset (optional but good for understanding)
print("Dataset type:", type(iris))  # <class 'sklearn.utils._bunch.Bunch'>
print("Available keys:", iris.keys())
print("\nFeature names:", iris.feature_names)
print("Target names (classes):", iris.target_names)

# Step 4: View data samples
print("\nFirst 5 rows of features:\n", iris.data[:5])
print("First 5 target labels:\n", iris.target[:5])
print("Shape of data:", iris.data.shape)
print("Shape of target:", iris.target.shape)

# Step 5: Convert data to DataFrame for easier viewing
X = iris.data
y = iris.target
df = pd.DataFrame(X, columns=iris.feature_names)
df['target'] = y
print("\nFirst 5 rows of full DataFrame:\n", df.head())

# Step 6: Split the dataset into training and testing sets
Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size=0.2, shuffle=True)
print("\nShapes after train-test split:")
print("Xtrain:", Xtrain.shape, "| Xtest:", Xtest.shape)
print("ytrain:", ytrain.shape, "| ytest:", ytest.shape)

# Step 7: Create and train the KNN model with 6 neighbors
model = KNeighborsClassifier(n_neighbors=6)
model.fit(Xtrain, ytrain)

# Step 8: Predict on test data
results = model.predict(Xtest)
print("\nPredicted classes on test set:\n", results)

# Step 9: Calculate accuracy
accuracy = accuracy_score(ytest, results)
print("\nAccuracy of model:", accuracy)

# Step 10: Display the confusion matrix
ConfusionMatrixDisplay.from_estimator(model, Xtest, ytest)
plt.title("Confusion Matrix")
plt.show()

# Step 11: Print classification report
print("\nClassification Report:\n")
print(classification_report(ytest, results))

# Step 12: Test on new sample data
sample = np.array([[5.1, 3.5, 1.4, 0.2],
                   [4.9, 3.0, 1.4, 0.2],
                   [4.7, 3.2, 1.3, 0.2]])
predicted_classes = model.predict(sample)
print("\nPrediction on new sample data:", predicted_classes)
