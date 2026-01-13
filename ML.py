import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
##STEP1 Data Ingestion
df = pd.read_csv("credit_card_fraud_2025.csv")
print("Dataset Loaded Successfully")
print(df.head())
#STEP2 Handle Missing Values 
if df.isnull().values.any():
    df= df.fillna(df.median())
    print("Missing Values Imputed")
else:
    print("Missing Values not found")
#STEP3 Data Encoding
le = LabelEncoder()
for col in df.select_dtypes(include=["object"]).columns:
    df[col]= le.fit_transform(df[col])
print("Encoding Complete")
print(df.columns)
#STEP4 Data Normalization
target = "Fraud_Flag"
x = df.drop(target, axis=1)
y = df[target]
scaler = StandardScaler()
x_scaled = scaler.fit_transform(x)
#STEP5 Data split and use class weight according to instruction
x_train,x_test,y_train,y_test= train_test_split(x_scaled,y,test_size=0.2, stratify=y, random_state=42)
from sklearn.utils.class_weight import compute_class_weight
weight= compute_class_weight(class_weight="balanced", classes= np.unique(y_train),y= y_train)
class_weights = {0:weight[0], 1:weight[1]}
print(f"Total Rows:{len(df)} | Training Rows: {len(x_train)}")
print(f"Class Weights Forb Imbalance: {class_weights}")

print("WEEK ONE TASK COMPLETED")



