
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix

# Load dataset
data = pd.read_csv("creditcard.csv")

# Show first rows and class counts
print(data.head())
print(data['Class'].value_counts())

# Visualize class imbalance
sns.countplot(x='Class', data=data)
plt.title('Class Distribution')
plt.show()

# Scale 'Amount'
scaler = StandardScaler()
data['Amount'] = scaler.fit_transform(data[['Amount']])

# Drop 'Time'
data = data.drop(['Time'], axis=1)

# Features and labels
X = data.drop('Class', axis=1)
y = data['Class']

# Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# Train model
model = LogisticRegression()
model.fit(X_train, y_train)

# Predict and evaluate
y_pred = model.predict(X_test)
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))
