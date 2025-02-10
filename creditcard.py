# Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# For machine learning
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import SMOTE

# Step 2: Load the dataset
df = pd.read_csv('creditcard.csv')

# Check for null values
print(df.isnull().sum())

# Preview the dataset
print(df.head())

# Step 3: Data Preprocessing
# Separate features and the target variable
X = df.drop('Class', axis=1)  # Features
y = df['Class']  # Target

# Apply standard scaling (important for models like SVM or when using distance-based algorithms)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Since the data is highly imbalanced, we can either use under-sampling or SMOTE
# Here we'll use RandomUnderSampler for balancing
rus = RandomUnderSampler(random_state=42)
X_res, y_res = rus.fit_resample(X_scaled, y)

print(f"Balanced dataset size: {X_res.shape}")

# Step 4: Train/Test Split
# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, test_size=0.3, random_state=42)

# Step 5: Build and Train the Model (Random Forest)
# Create a Random Forest Classifier
rfc = RandomForestClassifier(n_estimators=100, random_state=42)

# Train the model
rfc.fit(X_train, y_train)

# Make predictions
y_pred = rfc.predict(X_test)

# Step 6: Evaluate the Model
# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.title("Confusion Matrix")
plt.show()

# Classification Report
print(classification_report(y_test, y_pred))

# Accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.4f}")

# Step 7: Optional - SMOTE for Over-Sampling
# Alternative method using SMOTE to oversample the minority class
sm = SMOTE(random_state=42)
X_smote, y_smote = sm.fit_resample(X_scaled, y)

# Splitting the data
X_train_smote, X_test_smote, y_train_smote, y_test_smote = train_test_split(X_smote, y_smote, test_size=0.3, random_state=42)

# Train and evaluate the model with SMOTE
rfc.fit(X_train_smote, y_train_smote)
y_pred_smote = rfc.predict(X_test_smote)

# Evaluation
print("Evaluation with SMOTE:")
print(classification_report(y_test_smote, y_pred_smote))
