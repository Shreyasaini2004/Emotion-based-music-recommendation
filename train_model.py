import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import pickle

def load_and_train_model():
    print("Loading emotion data...")
    
    # Load all .npy files in current directory
    X_list = []
    y_list = []
    emotions = []
    
    for filename in os.listdir('.'):
        if filename.endswith('_data.npy'):
            emotion_name = filename.replace('_data.npy', '')
            data = np.load(filename)
            
            print(f"Loaded {emotion_name}: {data.shape[0]} samples")
            
            X_list.append(data)
            y_list.extend([emotion_name] * data.shape[0])
            emotions.append(emotion_name)
    
    if not X_list:
        print("No data files found! Please run data_collection.py first.")
        return
    
    # Combine all data
    X = np.vstack(X_list)
    y = np.array(y_list)
    
    print(f"\nTotal data shape: {X.shape}")
    print(f"Emotions found: {emotions}")
    print(f"Samples per emotion:")
    for emotion in emotions:
        count = np.sum(y == emotion)
        print(f"  {emotion}: {count}")
    
    # Encode labels
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    y_categorical = to_categorical(y_encoded)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_categorical, test_size=0.2, random_state=42, stratify=y_encoded
    )
    
    print(f"\nTraining set: {X_train.shape}")
    print(f"Test set: {X_test.shape}")
    
    # Build model
    model = Sequential([
        Dense(512, activation='relu', input_shape=(X.shape[1],)),
        BatchNormalization(),
        Dropout(0.3),
        
        Dense(256, activation='relu'),
        BatchNormalization(),
        Dropout(0.3),
        
        Dense(128, activation='relu'),
        BatchNormalization(),
        Dropout(0.2),
        
        Dense(64, activation='relu'),
        Dropout(0.2),
        
        Dense(len(emotions), activation='softmax')
    ])
    
    # Compile model
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    print("\nModel architecture:")
    model.summary()
    
    # Callbacks
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True
    )
    
    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=5,
        min_lr=1e-6
    )
    
    # Train model
    print("\nTraining model...")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=100,
        batch_size=32,
        callbacks=[early_stopping, reduce_lr],
        verbose=1
    )
    
    # Evaluate model
    train_loss, train_acc = model.evaluate(X_train, y_train, verbose=0)
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
    
    print(f"\nFinal Results:")
    print(f"Training Accuracy: {train_acc:.4f}")
    print(f"Test Accuracy: {test_acc:.4f}")
    
    # Save model and label encoder
    model.save('emotion_model.h5')
    
    with open('label_encoder.pkl', 'wb') as f:
        pickle.dump(label_encoder, f)
    
    # Save emotion labels
    np.save('emotions.npy', label_encoder.classes_)
    
    print(f"\nModel saved as 'emotion_model.h5'")
    print(f"Label encoder saved as 'label_encoder.pkl'")
    print(f"Emotions saved as 'emotions.npy'")

if __name__ == "__main__":
    load_and_train_model()