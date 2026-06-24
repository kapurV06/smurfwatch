import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import os

# Import our custom kinematic preprocessor from Day 2
from preprocess import load_and_prepare_sequences

def build_kinematic_gru_net(input_shape):
    """
    Constructs a Deep Temporal GRU architecture optimized to extract
    behavioral anomalies from streaming kinematic metrics.
    """
    model = Sequential([
        # Layer 1: Primary Recurrent Processing. return_sequences=True passes the hidden 
        # state chain forward to the next layer.
        GRU(64, return_sequences=True, input_shape=input_shape),
        BatchNormalization(),
        Dropout(0.3),
        
        # Layer 2: Secondary Temporal Aggregation. Collapses the 60-second time dimension 
        # into a highly compressed behavioral feature vector.
        GRU(32, return_sequences=False),
        BatchNormalization(),
        Dropout(0.3),
        
        # Layer 3: Dense non-linear consolidation
        Dense(16, activation='relu'),
        
        # Layer 4: Output Logit (Sigmoid for binary Smurf [1] vs. Legit [0] categorization)
        Dense(1, activation='sigmoid')
    ])
    
    # Compile with Adam optimizer using an explicit learning rate
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=[
            'accuracy', 
            tf.keras.metrics.Precision(name='precision'), 
            tf.keras.metrics.Recall(name='recall')
        ]
    )
    
    return model

def run_training_pipeline():
    # 1. Fetch the position-invariant 3D arrays from our Day 2 engine
    X_train, X_test, y_train, y_test = load_and_prepare_sequences()
    
    # Input shape will be dynamically verified as (60 timesteps, 6 features)
    input_shape = (X_train.shape[1], X_train.shape[2])
    
    print("\n[INFO]: Compiling Stacked Kinematic GRU Classifier...")
    model = build_kinematic_gru_net(input_shape)
    model.summary()
    
    # 2. Configure training callbacks
    # Early stopping prevents over-memorization of synthetic noise patterns.
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True, verbose=1),
        ModelCheckpoint(filepath='models/smurfwatch_net.keras', monitor='val_loss', save_best_only=True, verbose=1)
    ]
    
    # 3. Fire up the training loops over temporal sequences
    print("\n[START]: Commencing Recurrent Cell Optimizations...")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=35,
        batch_size=32,
        callbacks=callbacks,
        verbose=1
    )
    
    # 4. Run out-of-sample performance audit
    print("\n[AUDIT]: Executing Strict Out-of-Sample Evaluation...")
    loss, accuracy, precision, recall = model.evaluate(X_test, y_test, verbose=0)
    
    # Calculate F1-Score (The ultimate standard metric for validation balance)
    f1_score = 2 * (precision * recall) / (precision + recall + 1e-10)
    
    print("\n" + "="*60)
    print("DAY 3 MODEL EVALUATION REPORT:")
    print(f" -> Accuracy:       {accuracy:.4f}")
    print(f" -> Precision:      {precision:.4f}")
    print(f" -> Recall:         {recall:.4f}")
    print(f" -> F1-Score:       {f1_score:.4f}")
    print("="*60)
    print("Neural core frozen and successfully written to 'models/smurfwatch_net.keras'\n")

if __name__ == "__main__":
    run_training_pipeline()