# train_sign_language.py
"""
Sign Language Recognition Training System
Supports single sign and multi-sign video processing
"""

import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import json
import argparse
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import mediapipe as mp
from tqdm import tqdm
import mlflow
import mlflow.tensorflow
import warnings
warnings.filterwarnings('ignore')

# Initialize MediaPipe
mp_holistic = mp.solutions.holistic

class SignLanguageDataProcessor:
    """Process video files and extract keypoints for sign language recognition"""
    
    def __init__(self, sequence_length=64, frame_size=(224, 224)):
        self.sequence_length = sequence_length
        self.frame_size = frame_size
        self.mp_holistic = mp_holistic.Holistic(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
    def extract_keypoints(self, results):
        """Extract keypoints from MediaPipe holistic results"""
        # Pose landmarks (33 points * 4 values each)
        pose = np.array([[res.x, res.y, res.z, res.visibility] 
                        for res in results.pose_landmarks.landmark]).flatten() \
               if results.pose_landmarks else np.zeros(33*4)
        
        # Face landmarks (468 points * 3 values each)
        face = np.array([[res.x, res.y, res.z] 
                        for res in results.face_landmarks.landmark]).flatten() \
               if results.face_landmarks else np.zeros(468*3)
        
        # Left hand landmarks (21 points * 3 values each)
        lh = np.array([[res.x, res.y, res.z] 
                      for res in results.left_hand_landmarks.landmark]).flatten() \
             if results.left_hand_landmarks else np.zeros(21*3)
        
        # Right hand landmarks (21 points * 3 values each)
        rh = np.array([[res.x, res.y, res.z] 
                      for res in results.right_hand_landmarks.landmark]).flatten() \
             if results.right_hand_landmarks else np.zeros(21*3)
        
        return np.concatenate([pose, face, lh, rh])
    
    def preprocess_video(self, video_path):
        """Preprocess video file and extract keypoints sequence"""
        cap = cv2.VideoCapture(video_path)
        frames = []
        
        # Get total frame count
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Calculate step size for uniform sampling
        if total_frames > self.sequence_length:
            step = total_frames // self.sequence_length
        else:
            step = 1
            
        frame_indices = np.arange(0, min(total_frames, self.sequence_length * step), step)
        frame_indices = frame_indices[:self.sequence_length]  # Ensure exact length
        
        keypoints_sequence = []
        
        for i in range(total_frames):
            ret, frame = cap.read()
            if not ret:
                break
                
            if i in frame_indices:
                # Convert BGR to RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Process frame with MediaPipe
                results = self.mp_holistic.process(frame_rgb)
                
                # Extract keypoints
                keypoints = self.extract_keypoints(results)
                keypoints_sequence.append(keypoints)
        
        cap.release()
        
        # Pad sequence if needed
        while len(keypoints_sequence) < self.sequence_length:
            keypoints_sequence.append(np.zeros_like(keypoints_sequence[0]) 
                                    if keypoints_sequence else np.zeros(1662))
        
        # Truncate if needed
        keypoints_sequence = keypoints_sequence[:self.sequence_length]
        
        return np.array(keypoints_sequence)
    
    def create_dataset(self, data_dir):
        """Create dataset from directory of video files"""
        sequences = []
        labels = []
        label_map = {}
        
        classes = sorted(os.listdir(data_dir))
        for i, class_name in enumerate(classes):
            label_map[i] = class_name
            class_path = os.path.join(data_dir, class_name)
            
            if not os.path.isdir(class_path):
                continue
                
            print(f"Processing class: {class_name}")
            for video_file in tqdm(os.listdir(class_path)):
                if video_file.lower().endswith(('.mp4', '.mov')):
                    video_path = os.path.join(class_path, video_file)
                    try:
                        keypoints = self.preprocess_video(video_path)
                        sequences.append(keypoints)
                        labels.append(i)
                    except Exception as e:
                        print(f"Error processing {video_path}: {e}")
        
        return np.array(sequences), np.array(labels), label_map

class VideoMAETransformer:
    """VideoMAE Transformer model for sign language recognition"""
    
    def __init__(self, input_shape, num_classes, d_model=512, num_heads=8, num_layers=6):
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.d_model = d_model
        self.num_heads = num_heads
        self.num_layers = num_layers
        
    def build_model(self):
        """Build VideoMAE transformer model"""
        # Input layer
        inputs = layers.Input(shape=self.input_shape)
        
        # Feature embedding
        x = layers.Dense(self.d_model, activation='relu')(inputs)
        
        # Positional encoding
        positions = tf.range(start=0, limit=self.input_shape[0], delta=1)
        pos_encoding = layers.Embedding(
            input_dim=self.input_shape[0],
            output_dim=self.d_model
        )(positions)
        x = x + pos_encoding
        
        # Transformer blocks
        for _ in range(self.num_layers):
            # Multi-head attention
            attn_output = layers.MultiHeadAttention(
                num_heads=self.num_heads,
                key_dim=self.d_model
            )(x, x)
            x = layers.LayerNormalization()(x + attn_output)
            
            # Feed forward
            ffn = keras.Sequential([
                layers.Dense(self.d_model * 4, activation='relu'),
                layers.Dense(self.d_model)
            ])
            ffn_output = ffn(x)
            x = layers.LayerNormalization()(x + ffn_output)
        
        # Global average pooling
        x = layers.GlobalAveragePooling1D()(x)
        
        # Classification head
        outputs = layers.Dense(self.num_classes, activation='softmax')(x)
        
        model = keras.Model(inputs, outputs)
        return model

def train_model(data_dir, model_save_path, epochs=50, batch_size=8):
    """Train the sign language recognition model"""
    
    # Initialize MLflow
    mlflow.set_experiment("Sign Language Recognition")
    
    with mlflow.start_run():
        # Log parameters
        mlflow.log_param("epochs", epochs)
        mlflow.log_param("batch_size", batch_size)
        mlflow.log_param("sequence_length", 64)
        
        # Process data
        print("Processing video data...")
        processor = SignLanguageDataProcessor(sequence_length=64)
        X, y, label_map = processor.create_dataset(data_dir)
        
        # Save label map
        with open('label_map.json', 'w') as f:
            json.dump(label_map, f)
        mlflow.log_artifact('label_map.json')
        
        print(f"Dataset shape: {X.shape}")
        print(f"Number of classes: {len(label_map)}")
        print(f"Classes: {list(label_map.values())}")
        
        # Log dataset info
        mlflow.log_param("num_classes", len(label_map))
        mlflow.log_param("dataset_size", len(X))
        
        # Convert labels to categorical
        y_cat = keras.utils.to_categorical(y, num_classes=len(label_map))
        
        # Split data
        X_train, X_temp, y_train, y_temp = train_test_split(
            X, y_cat, test_size=0.4, random_state=42, stratify=y
        )
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp, test_size=0.5, random_state=42, stratify=np.argmax(y_temp, axis=1)
        )
        
        print(f"Training set: {X_train.shape}")
        print(f"Validation set: {X_val.shape}")
        print(f"Test set: {X_test.shape}")
        
        # Build model
        model_builder = VideoMAETransformer(
            input_shape=(X.shape[1], X.shape[2]),
            num_classes=len(label_map)
        )
        model = model_builder.build_model()
        
        # Compile model
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=1e-4),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        # Log model summary
        with open('model_summary.txt', 'w') as f:
            model.summary(print_fn=lambda x: f.write(x + '\n'))
        mlflow.log_artifact('model_summary.txt')
        
        # Callbacks
        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor='val_accuracy',
                patience=10,
                restore_best_weights=True
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-7
            )
        ]
        
        # Train model
        print("Training model...")
        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1
        )
        
        # Log metrics
        for epoch, (loss, acc, val_loss, val_acc) in enumerate(zip(
            history.history['loss'],
            history.history['accuracy'],
            history.history['val_loss'],
            history.history['val_accuracy']
        )):
            mlflow.log_metric("train_loss", loss, step=epoch)
            mlflow.log_metric("train_accuracy", acc, step=epoch)
            mlflow.log_metric("val_loss", val_loss, step=epoch)
            mlflow.log_metric("val_accuracy", val_acc, step=epoch)
        
        # Evaluate model
        test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
        print(f"Test Accuracy: {test_accuracy:.4f}")
        mlflow.log_metric("test_accuracy", test_accuracy)
        
        # Save model
        model.save(model_save_path)
        mlflow.tensorflow.log_model(model, "model")
        
        print("Training completed successfully!")
        return model, history

def main():
    parser = argparse.ArgumentParser(description='Sign Language Recognition Training')
    parser.add_argument('--data_dir', type=str, required=True,
                       help='Path to training data directory')
    parser.add_argument('--model_path', type=str, default='sign_language_model.h5',
                       help='Path to save trained model')
    parser.add_argument('--epochs', type=int, default=50,
                       help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=8,
                       help='Batch size for training')
    
    args = parser.parse_args()
    
    train_model(
        data_dir=args.data_dir,
        model_save_path=args.model_path,
        epochs=args.epochs,
        batch_size=args.batch_size
    )

if __name__ == "__main__":
    main()
