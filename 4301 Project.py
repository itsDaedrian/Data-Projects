import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans
from mlxtend.frequent_patterns import apriori, association_rules
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load the dataset
data = pd.read_csv('ds_salaries.csv')


# Print the data attributes and attributes' types
print("Data Attributes and Types:\n")
print(data.dtypes)

# Display 10 objects of data
print("\nFirst 10 Rows of Data:\n")
print(data.head(10))

# Conduct Linear Regression
# Using 'work_year' as the independent variable and 'salaryinusd' as the dependent variable
X = data[['work_year']]
y = data['salary_in_usd'].apply(pd.to_numeric, errors='coerce', downcast='float')
reg = LinearRegression().fit(X, y)
print("\nLinear Regression Results:")
print("Slope Coefficient (B1):", reg.coef_)
print("Intercept (B0):", reg.intercept_)



# Conduct Classification
# Using Decision Tree Classifier to predict whether the employee works in the US or not based on their years of experience and salary
X = data[['work_year', 'salary_in_usd']]
y = data['employee_residence']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Fit a decision tree classifier to the training data
clf = DecisionTreeClassifier(random_state=42)
clf.fit(X_train, y_train)

# Predict the test data and calculate accuracy score
y_pred = clf.predict(X_test)
print("\nClassification Results (Decision Tree Classifier):")
print("Accuracy Score:", accuracy_score(y_test, y_pred))

# Conduct Clustering
# Using K-Means clustering to group employees based on their years of experience and salary
X = data[['work_year', 'salary_in_usd']]
kmeans = KMeans(n_clusters=3, random_state=42).fit(X)
cluster_labels = kmeans.labels_
data['Cluster'] = cluster_labels
print("\nClustering Results (K-Means, 3 Clusters):\n")
print(data[['work_year', 'salary_in_usd', 'Cluster']])
