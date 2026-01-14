import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import EarlyStopping
#1.Data Load
df = pd.read_csv('credit_card_fraud_2025.csv')

#2.Preprocessing (change labels into numbers)
le = LabelEncoder()
for col in df.select_dtypes(include=['object']).columns:
    df[col] = le.fit_transform(df[col])

x = df.drop('Fraud_Flag', axis=1)
y = df['Fraud_Flag']
#3.Scaling
scaler = StandardScaler()
x_scaled = scaler.fit_transform(x)

#4.Data Splitting
x_train, x_test, y_train, y_test = train_test_split(x_scaled, y, test_size=0.2, stratify=y, random_state=42)

#5.load previous model
model = load_model('Fraud_model_v1.h5')
#Fix model & Compile
model.compile(optimizer = "adam", loss= "binary_crossentropy", metrics = ["Accuracy"])
#6.WEEK 3: Hyperparameter Tuning (Early Stopping)
early_stop = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
print("\nWeek 3: Starting hyperparameter tuning...")
history = model.fit(
    x_train, y_train,
    epochs=10,
    batch_size=512,
    validation_split=0.2,
    callbacks=[early_stop]
) 
# 7. Optimized Model Saved
model.save('Fraud_model_v2_tuned.h5')
print("\nWeek 3 Task Completed: Model optimized and saved!")
