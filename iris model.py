# Import Libraries
import numpy as np
import pandas as pd

# Import visualization tools
import matplotlib.pyplot as plt
import seaborn as sns

# Import evaluation metric libraries
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split, cross_val_score

# Import ML models
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB

# Ignore warnings
import warnings
warnings.filterwarnings('ignore')

# Load dataset from sklearn
from sklearn.datasets import load_iris
iris = load_iris()
df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
df['Species'] = iris.target

# Display first five rows
print(df.head())

# Check for null values
print("\nMissing Values:\n", df.isnull().sum())

# Splitting data into features and target variable
X = df.drop(columns=["Species"])
y = df["Species"]

# Splitting dataset into training and testing set with shuffle enabled
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, shuffle=True
)

# Check if training and testing sets have duplicate rows
print(f"\nTraining set size: {X_train.shape}, Test set size: {X_test.shape}")
print("Duplicate rows in train and test:", pd.merge(X_train, X_test, how='inner').shape[0])

# Define models (excluding XGBoost)
models = {
    "Logistic Regression": LogisticRegression(),
    "Decision Tree": DecisionTreeClassifier(max_depth=5),  # Limited depth to avoid overfitting
    "Random Forest": RandomForestClassifier(n_estimators=100, max_depth=5),
    "Support Vector Machine": SVC(),
    "Neural Network": MLPClassifier(max_iter=500),
    "Naive Bayes": GaussianNB()
}

# Dictionary to store model performance
model_performance = {}

# Evaluate each model
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    model_performance[name] = accuracy
    print(f"{name} Test Accuracy: {accuracy:.4f}")

    # Cross-validation for better evaluation
    cv_scores = cross_val_score(model, X, y, cv=5)
    print(f"{name} Cross-Validation Accuracy: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

    # Check for overfitting
    y_train_pred = model.predict(X_train)
    train_acc = accuracy_score(y_train, y_train_pred)
    print(f"{name} Training Accuracy: {train_acc:.4f}\n")

# Visualizing model performance
plt.figure(figsize=(10, 5))
sns.barplot(x=list(model_performance.keys()), y=list(model_performance.values()))
plt.xlabel("Model")
plt.ylabel("Accuracy")
plt.xticks(rotation=45)
plt.title("Model Performance Comparison")
plt.show()
