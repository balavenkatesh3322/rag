# inference_sign_language.py
"""
Sign Language Recognition Inference System
Supports single sign recognition and multi-sign processing
"""

import cv2
import numpy as np
import tensorflow as tf
from collections import deque
import json
import argparse
import mediapipe as mp
import time

# Initialize MediaPipe
mp_holistic = mp.solutions.holistic

class SignRecognizer:
    """Base sign recognizer for single sign recognition"""
    
    def __init__(self, model_path, label_map_path):
        # Load trained model
        self.model = tf.keras.models.load_model(model_path)
        
        # Load label map
        with open(label_map_path, 'r') as f:
            self.label_map = json.load(f)
            self.label_map = {int(k): v for k, v in self.label_map.items()}
        
        # Initialize MediaPipe
        self.mp_holistic = mp_holistic.Holistic(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
    def extract_keypoints(self, frame):
        """Extract keypoints from frame using MediaPipe"""
        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.mp_holistic.process(frame_rgb)
        
        # Extract keypoints
        pose = np.array([[res.x, res.y, res.z, res.visibility] 
                        for res in results.pose_landmarks.landmark]).flatten() \
               if results.pose_landmarks else np.zeros(33*4)
        
        face = np.array([[res.x, res.y, res.z] 
                        for res in results.face_landmarks.landmark]).flatten() \
               if results.face_landmarks else np.zeros(468*3)
        
        lh = np.array([[res.x, res.y, res.z] 
                      for res in results.left_hand_landmarks.landmark]).flatten() \
             if results.left_hand_landmarks else np.zeros(21*3)
        
        rh = np.array([[res.x, res.y, res.z] 
                      for res in results.right_hand_landmarks.landmark]).flatten() \
             if results.right_hand_landmarks else np.zeros(21*3)
        
        return np.concatenate([pose, face, lh, rh])
    
    def predict_sign(self, keypoints_sequence):
        """Predict sign from keypoints sequence"""
        if len(keypoints_sequence) != 64:
            # Pad or truncate to 64 frames
            if len(keypoints_sequence) < 64:
                while len(keypoints_sequence) < 64:
                    keypoints_sequence.append(np.zeros(1662))
            else:
                keypoints_sequence = keypoints_sequence[:64]
        
        sequence = np.array(keypoints_sequence)
        sequence = np.expand_dims(sequence, axis=0)
        
        predictions = self.model.predict(sequence, verbose=0)
        confidence = np.max(predictions[0])
        predicted_class = np.argmax(predictions[0])
        
        return self.label_map[predicted_class], confidence

class SingleSignRecognizer(SignRecognizer):
    """Recognizer for single sign videos"""
    
    def __init__(self, model_path, label_map_path, sequence_length=64):
        super().__init__(model_path, label_map_path)
        self.sequence_length = sequence_length
        
    def process_video(self, video_path):
        """Process single sign video and return prediction"""
        cap = cv2.VideoCapture(video_path)
        keypoints_sequence = []
        
        # Get total frame count
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Calculate step size for uniform sampling
        if total_frames > self.sequence_length:
            step = total_frames // self.sequence_length
        else:
            step = 1
            
        frame_indices = np.arange(0, min(total_frames, self.sequence_length * step), step)
        frame_indices = frame_indices[:self.sequence_length]
        
        for i in range(total_frames):
            ret, frame = cap.read()
            if not ret:
                break
                
            if i in frame_indices:
                keypoints = self.extract_keypoints(frame)
                keypoints_sequence.append(keypoints)
        
        cap.release()
        
        # Predict sign
        sign, confidence = self.predict_sign(keypoints_sequence)
        return sign, confidence

class MultiSignRecognizer(SignRecognizer):
    """Recognizer for multi-sign videos with segmentation"""
    
    def __init__(self, model_path, label_map_path, sequence_length=64, 
                 min_sign_duration=0.5, confidence_threshold=0.7):
        super().__init__(model_path, label_map_path)
        self.sequence_length = sequence_length
        self.min_sign_duration = min_sign_duration
        self.confidence_threshold = confidence_threshold
        self.frame_buffer = deque(maxlen=sequence_length)
        self.recognized_signs = []
        
    def detect_activity(self, keypoints):
        """Detect sign activity based on hand movement"""
        # Extract hand keypoints (last 126 values are hands: 21*3*2)
        hand_keypoints = keypoints[-126:]
        
        # Calculate movement (simplified detection)
        movement = np.std(hand_keypoints)
        return movement > 0.01  # Threshold for activity
    
    def process_stream(self, frame_stream):
        """Process continuous frame stream for multi-sign recognition"""
        self.frame_buffer.clear()
        self.recognized_signs = []
        activity_detected = False
        frames_since_activity = 0
        
        for frame in frame_stream:
            keypoints = self.extract_keypoints(frame)
            activity = self.detect_activity(keypoints)
            
            if activity:
                activity_detected = True
                frames_since_activity = 0
                self.frame_buffer.append(keypoints)
                
                # When buffer is full, make prediction
                if len(self.frame_buffer) == self.sequence_length:
                    sign, confidence = self.predict_sign(list(self.frame_buffer))
                    if confidence > self.confidence_threshold:
                        self.recognized_signs.append((sign, confidence))
                    self.frame_buffer.clear()
            elif activity_detected:
                frames_since_activity += 1
                self.frame_buffer.append(keypoints)
                
                # If inactivity for too long, finalize current sign
                if frames_since_activity > 30:  # About 1 second at 30fps
                    if len(self.frame_buffer) >= 32:  # Minimum frames for prediction
                        sign, confidence = self.predict_sign(list(self.frame_buffer))
                        if confidence > self.confidence_threshold:
                            self.recognized_signs.append((sign, confidence))
                    self.frame_buffer.clear()
                    activity_detected = False
                    frames_since_activity = 0
            else:
                # Keep a small buffer for context
                if len(self.frame_buffer) < 10:
                    self.frame_buffer.append(keypoints)
        
        return self.recognized_signs
    
    def process_video_file(self, video_path):
        """Process multi-sign video file"""
        cap = cv2.VideoCapture(video_path)
        frames = []
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)
        
        cap.release()
        
        # Process frames as stream
        recognized_signs = self.process_stream(frames)
        return recognized_signs

class RealTimeRecognizer(MultiSignRecognizer):
    """Real-time sign recognition from camera stream"""
    
    def __init__(self, model_path, label_map_path):
        super().__init__(model_path, label_map_path)
        
    def start_real_time_recognition(self):
        """Start real-time recognition from webcam"""
        cap = cv2.VideoCapture(0)
        
        print("Starting real-time sign language recognition...")
        print("Press 'q' to quit")
        
        # For displaying recognized text
        display_text = ""
        last_sign_time = time.time()
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Process frame
            keypoints = self.extract_keypoints(frame)
            activity = self.detect_activity(keypoints)
            
            if activity:
                self.frame_buffer.append(keypoints)
                
                # When buffer is full, make prediction
                if len(self.frame_buffer) == self.sequence_length:
                    sign, confidence = self.predict_sign(list(self.frame_buffer))
                    if confidence > self.confidence_threshold:
                        # Add to recognized signs
                        current_time = time.time()
                        if current_time - last_sign_time > 1.0:  # 1 second gap
                            self.recognized_signs.append((sign, confidence))
                            display_text = " ".join([s[0] for s in self.recognized_signs[-5:]])
                            last_sign_time = current_time
                    self.frame_buffer.clear()
            
            # Display results
            display_frame = frame.copy()
            cv2.putText(display_frame, f"Recognized: {display_text}", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Show frame
            cv2.imshow('Sign Language Recognition', display_frame)
            
            # Break on 'q' key
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        # Cleanup
        cap.release()
        cv2.destroyAllWindows()
        
        return self.recognized_signs

def main():
    parser = argparse.ArgumentParser(description='Sign Language Recognition Inference')
    parser.add_argument('--mode', choices=['single', 'multi', 'realtime'], required=True,
                       help='Recognition mode: single sign, multi-sign, or real-time')
    parser.add_argument('--model_path', type=str, default='sign_language_model.h5',
                       help='Path to trained model')
    parser.add_argument('--label_map', type=str, default='label_map.json',
                       help='Path to label map JSON file')
    parser.add_argument('--video_path', type=str,
                       help='Path to video file (for single/multi modes)')
    
    args = parser.parse_args()
    
    if args.mode == 'single':
        if not args.video_path:
            raise ValueError("video_path is required for single mode")
        
        recognizer = SingleSignRecognizer(args.model_path, args.label_map)
        sign, confidence = recognizer.process_video(args.video_path)
        print(f"Recognized sign: {sign}")
        print(f"Confidence: {confidence:.4f}")
        
    elif args.mode == 'multi':
        if not args.video_path:
            raise ValueError("video_path is required for multi mode")
        
        recognizer = MultiSignRecognizer(args.model_path, args.label_map)
        signs = recognizer.process_video_file(args.video_path)
        
        print("Recognized signs:")
        for i, (sign, confidence) in enumerate(signs):
            print(f"{i+1}. {sign} (confidence: {confidence:.4f})")
        
        # Output as text
        text_output = " ".join([sign for sign, _ in signs])
        print(f"\nText output: {text_output}")
        
    elif args.mode == 'realtime':
        recognizer = RealTimeRecognizer(args.model_path, args.label_map)
        signs = recognizer.start_real_time_recognition()
        
        print("Recognized signs:")
        for i, (sign, confidence) in enumerate(signs):
            print(f"{i+1}. {sign} (confidence: {confidence:.4f})")
        
        # Output as text
        text_output = " ".join([sign for sign, _ in signs])
        print(f"\nText output: {text_output}")

if __name__ == "__main__":
    main()
