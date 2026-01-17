import tensorflow as tf
from tensorflow.keras import layers, models
# Finalized Model Architecture for Week 2 


# 1. Building the Model from Scratch
model = models.Sequential([
    # Input layer based on feature count
    layers.Dense(64, activation='relu', input_shape=(x_train.shape[1],)),
    layers.Dropout(0.2), 
    
    # Hidden layers
    layers.Dense(32, activation='relu'),
    layers.Dense(16, activation='relu'),
    
    # Output layer (Sigmoid for binary classification)
    layers.Dense(1, activation='sigmoid')
])

# 2. Compile Model with specialized metrics
model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=[
        'accuracy', 
        tf.keras.metrics.Precision(name='precision'), 
        tf.keras.metrics.Recall(name='recall')
    ]
)

# 3. Initial Baseline Training
print("\nStarting Week 2 Baseline Training...")
history = model.fit(
    x_train, y_train,
    epochs=10, 
    batch_size=1024, # Baray dataset ke liye batch size bara rakha hai
    validation_split=0.2,
    class_weight=class_weights, # Sir's Point 11: Class weights for imbalance
    verbose=1
)
print("\nWeek 2 Baseline Model is ready!")
model.save("Fraud_model_v1.h5")
print("Model Saved Successfully")
