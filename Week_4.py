import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import load_model

# 1. Data Loading & Preprocessing for evaluation
df = pd.read_csv('credit_card_fraud_2025.csv')
le = LabelEncoder()
for col in df.select_dtypes(include=['object']).columns:
    df[col] = le.fit_transform(df[col])

x = df.drop('Fraud_Flag', axis=1)
y = df['Fraud_Flag']
scaler = StandardScaler()
x_scaled = scaler.fit_transform(x)

# Data Split
_, x_test, _, y_test = train_test_split(x_scaled, y, test_size=0.2, stratify=y, random_state=42)

# Week_3 Optimized Model load 
print("Loading Optimized Model...")
model = load_model('Fraud_model_v2_tuned.h5')
# Model Evaluation
print("\nEvaluating Model on Test Data...")
results = model.evaluate(x_test, y_test, verbose=1)
print(f"Test Accuracy: {results[1]*100:.2f}%")

# WEEK 4 Task: Visualizing Results
# Note:prediction results chart for presentation
y_pred = (model.predict(x_test) > 0.5).astype("int32")

#A mini Comparison Graph
plt.figure(figsize=(8, 5))
plt.hist(y_test, alpha=0.5, label='Actual Fraud', color='blue')
plt.hist(y_pred, alpha=0.5, label='Predicted Fraud', color='red')
plt.title('Week 4: Actual vs Predicted Fraud Cases')
plt.xlabel('Fraud Flag (0 = No, 1 = Yes)')
plt.ylabel('Count')
plt.legend()
plt.show()

print("\nWeek 4: Final Evaluation and Graphs Completed!")