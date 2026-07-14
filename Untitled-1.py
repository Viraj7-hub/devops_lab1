from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
# Import the Decision Tree Classifier tool from scikit-learn
from sklearn.tree import DecisionTreeClassifier
# Import the Linear Regression tool from scikit-learn
from sklearn.linear_model import LinearRegression
# Import numpy to help with rounding the linear regression predictions
import numpy as np

# Generate a synthetic binary classification dataset
# n_samples=200 creates 200 data points
# n_features=2 gives each point 2 variables (X and Y coordinates)
# random_state=42 ensures the data generated is the exact same every time you run the script
X, y = make_classification(n_samples=200, n_features=2, n_informative=2, n_redundant=0, random_state=42)

# Split the generated data into training sets (80%) and testing sets (20%)
# X represents the features, y represents the target labels (0 or 1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the K-Nearest Neighbors classifier
# n_neighbors=5 tells the algorithm to look at the 5 closest data points to make a decision
knn = KNeighborsClassifier(n_neighbors=5)

# Train the KNN model using the training features and their corresponding labels
knn.fit(X_train, y_train)

# Use the trained model to predict the labels for the unseen testing data
y_pred = knn.predict(X_test)

# Calculate the accuracy by comparing the model's predictions (y_pred) against the true labels (y_test)
accuracy = accuracy_score(y_test, y_pred)
print(y_pred)
# Print the final accuracy score formatted as a percentage
print(f"Model Accuracy: {accuracy * 100:.2f}%")

# Initialize the Decision Tree classifier
# random_state=42 ensures the tree splits the data the exact same way every time you run it
dtree = DecisionTreeClassifier(random_state=42)

# Train the Decision Tree model using the same training data as the KNN model
dtree.fit(X_train, y_train)

# Use the trained Decision Tree model to predict labels for the testing data
y_pred_dtree = dtree.predict(X_test)

# Calculate the accuracy by comparing the Decision Tree's predictions against the true labels
accuracy_dtree = accuracy_score(y_test, y_pred_dtree)

# Print the Decision Tree predictions array
print(y_pred_dtree)

# Print the final accuracy score for the Decision Tree formatted as a percentage
print(f"Decision Tree Model Accuracy: {accuracy_dtree * 100:.2f}%")

# --- Linear Regression Addition Below ---

# Initialize the Linear Regression model
lin_reg = LinearRegression()

# Train the Linear Regression model using the same training data
lin_reg.fit(X_train, y_train)

# Use the trained Linear Regression model to predict values for the testing data
# Note: Because this is linear regression, it outputs continuous fractions (e.g., 0.45, 0.82) instead of exact 0s and 1s
y_pred_lin_continuous = lin_reg.predict(X_test)

# To calculate classification accuracy, we convert the continuous fractions into solid 0s and 1s
# We use numpy (np.where) to say: if the prediction is 0.5 or higher, make it a 1; otherwise, make it a 0
y_pred_lin = np.where(y_pred_lin_continuous >= 0.5, 1, 0)

# Calculate the accuracy by comparing the rounded predictions against the true labels
accuracy_lin = accuracy_score(y_test, y_pred_lin)

# Print the rounded Linear Regression predictions array
print(y_pred_lin)

# Print the final accuracy score for the Linear Regression model formatted as a percentage
print(f"Linear Regression Model Accuracy: {accuracy_lin * 100:.2f}%")

# --- K-Means Clustering Addition Below ---

# Import the KMeans clustering tool from scikit-learn
from sklearn.cluster import KMeans

# Initialize the KMeans clustering model
# n_clusters=2 tells the algorithm to group the data into 2 distinct clusters
# random_state=42 ensures the initial cluster centers are placed identically each time
# n_init='auto' suppresses a common scikit-learn warning about default behavior changes
kmeans = KMeans(n_clusters=2, random_state=42, n_init='auto')

# Train (fit) the KMeans model using ONLY the features (X_train)
# Note: Because clustering is unsupervised, we do NOT pass the true labels (y_train) to it
kmeans.fit(X_train)

# Use the trained model to assign each test data point to one of the 2 clusters
clusters_pred = kmeans.predict(X_test)

# Print the assigned cluster labels for the testing data
# Note: These 0s and 1s are just arbitrary names for 'Group A' and 'Group B', not necessarily the true class labels
print("\nK-Means Cluster Assignments:")
print(clusters_pred)

print("\nAdded Trial Operation by Tanmay")
print("Thank You")


print("thankyou")
print("Thankyou by Aryan")

print("Thank you by Viraj")
