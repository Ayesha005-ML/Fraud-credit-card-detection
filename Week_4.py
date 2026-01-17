import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout

# 1. Data Loading & Preprocessing
df = pd.read_csv('credit_card_fraud_2025.csv')
le = LabelEncoder()
for col in df.select_dtypes(include=['object']).columns:
    df[col] = le.fit_transform(df[col])
X = df.drop('Fraud_Flag', axis=1) 
y = df['Fraud_Flag'] 
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
x_train, x_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, stratify=y, random_state=42)

model = Sequential([
    Dense(32, activation='relu', input_shape=(x_train.shape[1],)), 
    Dropout(0.2), 
    Dense(16, activation='relu'),
    Dense(1, activation='sigmoid') 
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])


print("\n--- Starting Live Training from Scratch ---")
history = model.fit(x_train, y_train, epochs=20, batch_size=32, validation_data=(x_test, y_test))

print("\n--- Evaluating Model on Test Data ---")
results = model.evaluate(x_test, y_test, verbose=1)
print(f"Test Accuracy: {results[1]*100:.2f}%")

sample_input = x_test[0].reshape(1, -1)
prediction = model.predict(sample_input)
print(f"\nDemo Transaction Result: {prediction[0][0]}")
if prediction > 0.5:
    print("FINAL STATUS:[1]-> ALERT: FRAUD DETECTED!")
else:
    print("FINAL STATUS:[0]-> Transaction is Normal.")
    #--- Random Transaction Checker ---
import random
print("\n--- Testing Model on a RANDOM New Transaction ---")
random_index = random.randint(0, len(x_test)-1)
new_sample = x_test[random_index].reshape(1, -1)
raw_pred = model.predict(new_sample)
status = (raw_pred > 0.5).astype("int32")
print(f"Transaction Index: {random_index}")
print(f"Model Score: {raw_pred[0][0]:.4f}")
if status == 1:
    print("RESULT: [1] -> ALERT: Potential Fraud!")
else:
    print("RESULT: [0] -> Transaction is Legitimate.")
    
 from sklearn.metrics import classification_report
# Model evaluation with detailed metrics
y_pred_bool = (model.predict(x_test) > 0.5).astype("int32")
print(classification_report(y_test, y_pred_bool))