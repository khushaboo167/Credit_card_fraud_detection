# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns

# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import StandardScaler
# from sklearn.linear_model import LogisticRegression
# from sklearn.metrics import classification_report, confusion_matrix

# # 1. Load the Dataset
# data = pd.read_csv("C:/Users/asus/Desktop/credit_card_fraud_detection/creditcard.csv")

# # 2. Basic Info
# print("Dataset Head:\n", data.head())
# print("\nClass Distribution:\n", data['Class'].value_counts())

# # 3. Visualize Class Distribution
# sns.countplot(x='Class', data=data)
# plt.title('Class Distribution (0: Not Fraud, 1: Fraud)')
# plt.show()

# # 4. Preprocessing
# scaler = StandardScaler()
# data['Amount'] = scaler.fit_transform(data[['Amount']])
# data = data.drop(['Time'], axis=1)  # Drop time column

# X = data.drop('Class', axis=1)
# y = data['Class']

# # 5. Train-Test Split
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# # 6. Model Training
# model = LogisticRegression()
# model.fit(X_train, y_train)

# # 7. Evaluation
# y_pred = model.predict(X_test)
# print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
# print("\nClassification Report:\n", classification_report(y_test, y_pred))

# # 8. Predict a New Transaction
# # (Fake sample: replace with real values as needed)
# new_transaction = np.array([[-1.35980713, -0.07278117, 2.53634674, 1.37815522, -0.33832077,
#                              0.46238778, 0.23959855, 0.0986979, 0.36378697, 0.09079417,
#                              -0.55159953, -0.61780086, -0.99138985, -0.31116935, 1.46817697,
#                              -0.47040053, 0.20797124, 0.02579058, 0.40399296, 0.2514121,
#                              -0.01830678, 0.27783758, -0.11047391, 0.06692807, 0.12853936,
#                              -0.18911484, 0.13355838, -0.02105305, 149.62]])  # Last is 'Amount'

# # Scale Amount
# new_transaction[:, -1] = scaler.transform(new_transaction[:, -1].reshape(-1, 1)).flatten()

# # Predict
# prediction = model.predict(new_transaction)

# # Output result
# if prediction[0] == 1:
#     print(" The transaction is **Fraudulent**.")
# else:
#     print(" The transaction is **Not Fraudulent**.")









# correct code
# import numpy as np
# import pandas as pd
# from sklearn.datasets import make_classification
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import StandardScaler
# from sklearn.linear_model import LogisticRegression
# from sklearn.metrics import classification_report, confusion_matrix

# # 1. Generate Synthetic Credit Card Data
# # 1000 samples, 30 features (like the original dataset), with imbalance (fraud = 1)
# X, y = make_classification(n_samples=1000, n_features=30, n_informative=10, n_redundant=10,
#                            n_classes=2, weights=[0.9, 0.1], random_state=42)

# # 2. Create a DataFrame
# columns = [f'V{i}' for i in range(1, 29)] + ['Amount', 'Time']
# df = pd.DataFrame(X, columns=columns)
# df['Class'] = y

# print(" Sample of synthetic dataset:")
# print(df.head())

# # 3. Preprocessing
# scaler = StandardScaler()
# df['Amount'] = scaler.fit_transform(df[['Amount']])  # Only scale 'Amount'

# X = df.drop('Class', axis=1)
# y = df['Class']

# # 4. Train/Test Split
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# # 5. Train the Model
# model = LogisticRegression()
# model.fit(X_train, y_train)

# # 6. Evaluate
# y_pred = model.predict(X_test)
# print("\n Model Evaluation:")
# print(confusion_matrix(y_test, y_pred))
# print(classification_report(y_test, y_pred))

# # 7. Predict a New Transaction (fake)
# new_transaction = np.array([X.iloc[0]])  # Let's just reuse one real sample for example

# # Predict
# prediction = model.predict(new_transaction)

# # 8. Show Prediction
# print("\n New Transaction Prediction:")
# print(" Fraudulent" if prediction[0] == 1 else " Not Fraudulent")








# correct code
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix

# 1. Generate Synthetic Credit Card Data
X, y = make_classification(n_samples=1000, n_features=30, n_informative=10, n_redundant=10,
                           n_classes=2, weights=[0.9, 0.1], random_state=42)

# 2. Create a DataFrame
columns = [f'V{i}' for i in range(1, 29)] + ['Amount', 'Time']
df = pd.DataFrame(X, columns=columns)
df['Class'] = y

# 3. Preprocessing
scaler = StandardScaler()
df['Amount'] = scaler.fit_transform(df[['Amount']])  # Only scale 'Amount'

X = df.drop('Class', axis=1)
y = df['Class']

# 4. Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# 5. Train the Model
model = LogisticRegression()
model.fit(X_train, y_train)

# 6. Evaluate
y_pred = model.predict(X_test)
print("\n Model Evaluation:")
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

# 7. Predict a New Transaction
# Manually create a fake new transaction with 30 features
new_transaction = np.array([[
    0.1, -1.2, 0.3, -0.4, 1.1, -0.2, 0.5, -1.5, 0.6, -0.8,
    0.4, 0.3, -0.7, 1.5, -0.6, 0.9, -1.1, 0.8, -0.3, 0.2,
    -0.5, 0.7, -1.0, 0.6, 0.3, -0.2, 0.1, 150.0, 50000.0,  # Amount and Time (unscaled)
    0.0  # dummy extra (to adjust, if needed)
]])[:, :30]  # Ensure exactly 30 features

# Make sure Amount column is scaled (index -2 for 2nd last col = 'Amount')
new_transaction[0][-2] = scaler.transform([[new_transaction[0][-2]]])[0][0]

# Predict
prediction = model.predict(new_transaction)

# 8. Show Prediction
print("\n New Transaction Prediction:")
print(" Fraudulent" if prediction[0] == 1 else " Not Fraudulent")











# Here's a **step-by-step guide to confidently explain your credit card fraud detection project** to an interviewer, from concept to implementation and improvements:

# ---

# ## 🔷 1. **Project Overview**

# **Start with a concise summary:**

# > “This project focuses on detecting fraudulent credit card transactions using a machine learning model. I trained a Logistic Regression classifier on a dataset (synthetic or real) with imbalanced class distribution, where fraud cases are rare. The goal was to classify new transactions as either legitimate or fraudulent.”

# ---

# ## 🔷 2. **Motivation / Why This Project**

# > “Credit card fraud is a major concern in the financial industry. Since fraudulent cases are rare, it's a good real-world example of an imbalanced classification problem. I chose this project to apply ML techniques to solve such practical challenges.”

# ---

# ## 🔷 3. **Dataset Details**

# If you used a real dataset:

# > “The dataset contains transactions made with credit cards in September 2013 by European cardholders. It has 284,807 transactions and 30 input features — anonymized as V1 to V28, with `Amount` and `Time`, and a target label `Class` (1 for fraud, 0 for non-fraud).”

# If you used a **synthetic dataset**:

# > “I generated synthetic data using `make_classification` to simulate an imbalanced credit card fraud scenario with 30 features. This allowed me to prototype the fraud detection pipeline end-to-end.”

# ---

# ## 🔷 4. **Data Preprocessing**

# > “I scaled the `Amount` feature using `StandardScaler` to bring it in line with the other features. I dropped the `Time` column as it was not predictive. I also handled class imbalance by stratified sampling during train-test split.”

# *(If asked more:)* You can mention oversampling with SMOTE, undersampling, or using metrics like precision/recall instead of just accuracy.

# ---

# ## 🔷 5. **Model Choice: Logistic Regression**

# > “I chose Logistic Regression because it’s interpretable, fast to train, and works well for binary classification. It gives me direct probability outputs, which helps in threshold tuning for fraud detection.”

# ---

# ## 🔷 6. **Model Evaluation**

# > “I split the data into training and test sets with an 80:20 ratio using stratified sampling. After training, I evaluated the model using confusion matrix, precision, recall, and F1-score. These metrics are more meaningful than accuracy due to class imbalance.”

# ✔️ Also mention:

# * High recall (sensitivity) is desirable in fraud detection.
# * You looked at the false negatives carefully.

# ---

# ## 🔷 7. **Prediction on New Data**

# > “I manually created a new transaction with 30 features, scaled the `Amount` value using the same scaler, and passed it to the trained model to predict whether it is fraudulent or not.”

# You can show the part of your code that does this if needed.

# ---

# ## 🔷 8. **Challenges Faced**

# * Class imbalance (fraud cases are <1%)
# * Choosing the right evaluation metrics
# * Scaling features consistently

# ---

# ## 🔷 9. **What You’d Improve**

# > “In future iterations, I’d use SMOTE for oversampling, compare other models like Random Forests or XGBoost, and use grid search for hyperparameter tuning. I’d also implement a ROC-AUC curve for threshold tuning.”

# ---

# ## 🔷 10. **Project Tools**

# > “Python, scikit-learn, pandas, NumPy, matplotlib, seaborn. I used VS Code for development.”

# ---

# ## ✅ Final One-Liner Summary

# > “I built and trained a logistic regression model to detect fraudulent transactions, evaluated its performance, and tested it with synthetic inputs to simulate real-time fraud detection.”

# ---

# Let me know if you want a **spoken mock answer** or **slides/script** for your viva/interview.
