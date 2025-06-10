# rl_feedback_system.py
"""
Reinforcement Learning-based Feedback System for Emotion Recognition
This module implements a feedback learning system that improves emotion recognition
and stress estimation accuracy using user corrections.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import json
import sqlite3
import datetime
from typing import Dict, List, Tuple, Optional
from collections import deque
import random
from dataclasses import dataclass
from pathlib import Path
from PIL import Image
import cv2
from torchvision import transforms

# Define constants directly to avoid import issues
STANDARD_EMOTIONS = ['angry', 'contempt', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

# Define image transformations
data_transforms = {
    'train': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(20),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
        transforms.RandomPerspective(distortion_scale=0.05, p=0.3),
        transforms.RandomAffine(degrees=0, translate=(0.05, 0.05), scale=(0.95, 1.05)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
}

# Import emotion configuration if available, otherwise use fallbacks
try:
    from emotion_config import ( # type: ignore
        EMOTION_TO_INDEX, 
        FEEDBACK_CONFIG,
        DATABASE_CONFIG
    )
except ImportError:
    # Fallback if emotion_config is not available
    EMOTION_TO_INDEX = {emotion: idx for idx, emotion in enumerate(STANDARD_EMOTIONS)}
    FEEDBACK_CONFIG = {
        'update_frequency': 50,
        'min_confidence_for_correction': 0.8,
        'correction_weight': 0.3,
        'batch_size': 32,
        'learning_rate': 1e-4,
        'experience_buffer_size': 10000
    }
    DATABASE_CONFIG = {
        'path': 'feedback.db',
        'backup_frequency': 1000,
        'cleanup_days': 90
    }

@dataclass
class FeedbackEntry:
    """Structure for storing user feedback data"""
    image_features: np.ndarray
    predicted_emotion: str
    predicted_stress: float
    correct_emotion: str
    correct_stress: float
    timestamp: datetime.datetime
    confidence: float
    user_id: Optional[str] = None
    
class FeedbackDatabase:
    """Database manager for storing and retrieving user feedback"""
    
    def __init__(self, db_path: str = None):
        self.db_path = db_path or DATABASE_CONFIG.get('path', 'feedback.db')
        self.init_database()
    
    def init_database(self):
        """Initialize the feedback database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS feedback (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                image_features BLOB,
                predicted_emotion TEXT,
                predicted_stress REAL,
                correct_emotion TEXT,
                correct_stress REAL,
                confidence REAL,
                user_id TEXT,
                timestamp DATETIME,
                used_for_training BOOLEAN DEFAULT FALSE
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS model_performance (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                epoch INTEGER,
                accuracy REAL,
                avg_stress_error REAL,
                feedback_count INTEGER,
                timestamp DATETIME
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def store_feedback(self, feedback: FeedbackEntry):
        """Store user feedback in the database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Convert numpy array to bytes
        features_blob = feedback.image_features.tobytes()
        
        cursor.execute('''
            INSERT INTO feedback 
            (image_features, predicted_emotion, predicted_stress, correct_emotion, 
             correct_stress, confidence, user_id, timestamp)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            features_blob,
            feedback.predicted_emotion,
            feedback.predicted_stress,
            feedback.correct_emotion,
            feedback.correct_stress,
            feedback.confidence,
            feedback.user_id,
            feedback.timestamp
        ))
        
        conn.commit()
        conn.close()
    
    def get_feedback_batch(self, batch_size: int = 32, unused_only: bool = True) -> List[FeedbackEntry]:
        """Retrieve a batch of feedback entries for training"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        where_clause = "WHERE used_for_training = FALSE" if unused_only else ""
        cursor.execute(f'''
            SELECT image_features, predicted_emotion, predicted_stress, 
                   correct_emotion, correct_stress, confidence, user_id, timestamp
            FROM feedback {where_clause}
            ORDER BY timestamp DESC
            LIMIT ?
        ''', (batch_size,))
        
        rows = cursor.fetchall()
        conn.close()
        
        feedback_entries = []
        for row in rows:
            features = np.frombuffer(row[0], dtype=np.float32).reshape(-1)
            feedback_entries.append(FeedbackEntry(
                image_features=features,
                predicted_emotion=row[1],
                predicted_stress=row[2],
                correct_emotion=row[3],
                correct_stress=row[4],
                confidence=row[5],
                user_id=row[6],
                timestamp=datetime.datetime.fromisoformat(row[7])
            ))
        
        return feedback_entries
    
    def mark_feedback_used(self, feedback_ids: List[int]):
        """Mark feedback entries as used for training"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        placeholders = ','.join('?' * len(feedback_ids))
        cursor.execute(f'''
            UPDATE feedback 
            SET used_for_training = TRUE 
            WHERE id IN ({placeholders})
        ''', feedback_ids)
        
        conn.commit()
        conn.close()

class RewardCalculator:
    """Calculate rewards for reinforcement learning based on feedback"""
    
    @staticmethod
    def calculate_emotion_reward(predicted: str, correct: str, confidence: float) -> float:
        """Calculate reward for emotion prediction"""
        if predicted == correct:
            # Positive reward scaled by confidence
            return min(confidence * 2.0, 1.0)
        else:
            # Negative reward, more severe for high confidence wrong predictions
            return -min(confidence * 1.5, 1.0)
    
    @staticmethod
    def calculate_stress_reward(predicted: float, correct: float, threshold: float = 1.0) -> float:
        """Calculate reward for stress level prediction"""
        error = abs(predicted - correct)
        if error <= threshold:
            # Good prediction
            return 1.0 - (error / threshold) * 0.5
        else:
            # Poor prediction
            return -min(error / 5.0, 1.0)
    
    @staticmethod
    def calculate_combined_reward(emotion_reward: float, stress_reward: float, 
                                 emotion_weight: float = 0.6) -> float:
        """Combine emotion and stress rewards"""
        return emotion_weight * emotion_reward + (1 - emotion_weight) * stress_reward

class FeedbackLearner(nn.Module):
    """Neural network for learning from user feedback"""
    
    def __init__(self, feature_dim: int = 1280, emotion_classes: int = 8):
        super(FeedbackLearner, self).__init__()
        self.feature_dim = feature_dim
        self.emotion_classes = emotion_classes
        
        # Feature processing layers
        self.feature_processor = nn.Sequential(
            nn.Linear(feature_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        # Emotion correction head
        self.emotion_corrector = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, emotion_classes)
        )
        
        # Stress correction head
        self.stress_corrector = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
        
        # Value network for reward prediction
        self.value_network = nn.Sequential(
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
    
    def forward(self, features):
        """Forward pass through the network"""
        processed_features = self.feature_processor(features)
        
        emotion_logits = self.emotion_corrector(processed_features)
        stress_pred = self.stress_corrector(processed_features)
        value = self.value_network(processed_features)
        
        return emotion_logits, stress_pred.squeeze(), value.squeeze()

class ReinforcementFeedbackTrainer:
    """Main trainer class for reinforcement learning with user feedback"""
    
    def __init__(self, base_model, emotion_classes: List[str], 
                 feature_dim: int = 1280, learning_rate: float = 1e-4):
        self.base_model = base_model
        self.emotion_classes = emotion_classes
        self.emotion_to_idx = {emotion: idx for idx, emotion in enumerate(emotion_classes)}
        
        # Initialize feedback components
        self.feedback_db = FeedbackDatabase()
        self.reward_calculator = RewardCalculator()
        self.feedback_learner = FeedbackLearner(feature_dim, len(emotion_classes))
        
        # Training components
        self.optimizer = optim.Adam(self.feedback_learner.parameters(), lr=learning_rate)
        self.experience_buffer = deque(maxlen=10000)
        
        # Training parameters
        self.batch_size = FEEDBACK_CONFIG.get('batch_size', 32)
        self.update_frequency = FEEDBACK_CONFIG.get('update_frequency', 50)
        self.feedback_count = 0
        
    def extract_features(self, image_tensor: torch.Tensor) -> np.ndarray:
        """Extract features from the base model for feedback learning"""
        self.base_model.eval()
        with torch.no_grad():
            # Get features from the penultimate layer
            if hasattr(self.base_model, 'efficientnet'):
                features = self.base_model.efficientnet.features(image_tensor)
                features = self.base_model.efficientnet.avgpool(features)
                features = torch.flatten(features, 1)
            else:
                # Fallback for other architectures
                features = self.base_model(image_tensor)
            
            return features.cpu().numpy().flatten()
    
    def process_feedback(self, image_tensor: torch.Tensor, predicted_emotion: str, 
                        predicted_stress: float, correct_emotion: str, 
                        correct_stress: float, confidence: float, user_id: str = None):
        """Process user feedback and store for learning"""
        
        # Extract features for learning
        features = self.extract_features(image_tensor)
        
        # Create feedback entry
        feedback = FeedbackEntry(
            image_features=features,
            predicted_emotion=predicted_emotion,
            predicted_stress=predicted_stress,
            correct_emotion=correct_emotion,
            correct_stress=correct_stress,
            confidence=confidence,
            user_id=user_id,
            timestamp=datetime.datetime.now()
        )
        
        # Store feedback
        self.feedback_db.store_feedback(feedback)
        self.feedback_count += 1
        
        # Calculate rewards
        emotion_reward = self.reward_calculator.calculate_emotion_reward(
            predicted_emotion, correct_emotion, confidence
        )
        stress_reward = self.reward_calculator.calculate_stress_reward(
            predicted_stress, correct_stress
        )
        combined_reward = self.reward_calculator.calculate_combined_reward(
            emotion_reward, stress_reward
        )
        
        # Add to experience buffer
        self.experience_buffer.append({
            'features': features,
            'predicted_emotion_idx': self.emotion_to_idx[predicted_emotion],
            'correct_emotion_idx': self.emotion_to_idx[correct_emotion],
            'predicted_stress': predicted_stress,
            'correct_stress': correct_stress,
            'reward': combined_reward,
            'confidence': confidence
        })
        
        # Trigger training if enough feedback collected
        if self.feedback_count % self.update_frequency == 0:
            self.train_step()
        
        return combined_reward
    
    def train_step(self):
        """Perform one training step using collected feedback"""
        if len(self.experience_buffer) < self.batch_size:
            return
        
        # Sample batch from experience buffer
        batch = random.sample(list(self.experience_buffer), self.batch_size)
        
        # Prepare tensors
        features = torch.FloatTensor([exp['features'] for exp in batch])
        correct_emotions = torch.LongTensor([exp['correct_emotion_idx'] for exp in batch])
        correct_stress = torch.FloatTensor([exp['correct_stress'] for exp in batch])
        rewards = torch.FloatTensor([exp['reward'] for exp in batch])
        
        # Forward pass
        emotion_logits, stress_pred, values = self.feedback_learner(features)
        
        # Calculate losses
        emotion_loss = nn.CrossEntropyLoss()(emotion_logits, correct_emotions)
        stress_loss = nn.MSELoss()(stress_pred, correct_stress)
        
        # Policy gradient loss (REINFORCE-style)
        emotion_probs = torch.softmax(emotion_logits, dim=1)
        selected_probs = emotion_probs.gather(1, correct_emotions.unsqueeze(1)).squeeze()
        policy_loss = -torch.mean(torch.log(selected_probs) * rewards)
        
        # Value loss
        value_loss = nn.MSELoss()(values, rewards)
        
        # Combined loss
        total_loss = emotion_loss + stress_loss + 0.1 * policy_loss + 0.1 * value_loss
        
        # Backward pass
        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()
        
        print(f"Training step - Total Loss: {total_loss.item():.4f}, "
              f"Emotion Loss: {emotion_loss.item():.4f}, "
              f"Stress Loss: {stress_loss.item():.4f}")
    
    def get_corrected_prediction(self, image_tensor: torch.Tensor, 
                                original_emotion: str, original_stress: float) -> Tuple[str, float]:
        """Get corrected prediction using the feedback learner"""
        features = self.extract_features(image_tensor)
        features_tensor = torch.FloatTensor(features).unsqueeze(0)
        
        self.feedback_learner.eval()
        with torch.no_grad():
            emotion_logits, stress_correction, _ = self.feedback_learner(features_tensor)
            
            # Apply correction
            emotion_probs = torch.softmax(emotion_logits, dim=1)
            corrected_emotion_idx = torch.argmax(emotion_probs, dim=1).item()
            corrected_emotion = self.emotion_classes[corrected_emotion_idx]
            
            # Blend original and corrected stress
            alpha = 0.3  # Correction weight
            corrected_stress = (1 - alpha) * original_stress + alpha * stress_correction.item()
            corrected_stress = max(0, min(10, corrected_stress))  # Clamp to valid range
        
        return corrected_emotion, corrected_stress
    
    def save_model(self, path: str):
        """Save the feedback learner model"""
        torch.save({
            'model_state_dict': self.feedback_learner.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'emotion_classes': self.emotion_classes,
            'feedback_count': self.feedback_count
        }, path)
    
    def load_model(self, path: str):
        """Load the feedback learner model"""
        checkpoint = torch.load(path, map_location='cpu')
        self.feedback_learner.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.feedback_count = checkpoint.get('feedback_count', 0)
        
    def get_statistics(self) -> Dict:
        """Get training and feedback statistics"""
        return {
            'total_feedback_entries': self.feedback_count,
            'experience_buffer_size': len(self.experience_buffer),
            'model_parameters': sum(p.numel() for p in self.feedback_learner.parameters()),
            'last_update': self.feedback_count // self.update_frequency
        }

# Example usage and integration
def integrate_with_existing_system():
    """Example of how to integrate with your existing system"""
    
    # Initialize the feedback trainer with your existing model
    # feedback_trainer = ReinforcementFeedbackTrainer(
    #     base_model=your_existing_model,
    #     emotion_classes=STANDARD_EMOTIONS
    # )
    
    # Example prediction with feedback correction
    def enhanced_predict_emotion(model, feedback_trainer, image_path):
        """Enhanced prediction function with feedback learning"""
        
        # Your existing prediction logic
        transform = data_transforms['val']
        image = Image.open(image_path).convert('RGB')
        image_tensor = transform(image).unsqueeze(0)
        
        # Get original prediction
        with torch.no_grad():
            outputs = model(image_tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)[0]
            confidence, prediction = torch.max(probabilities, 0)
            
        original_emotion = STANDARD_EMOTIONS[prediction.item()]
        original_confidence = confidence.item()
        
        # Calculate original stress level (your existing logic)
        original_stress = calculate_stress_level_from_emotion(original_emotion, original_confidence)
        
        # Get corrected prediction if feedback learner is trained
        if feedback_trainer.feedback_count > 0:
            corrected_emotion, corrected_stress = feedback_trainer.get_corrected_prediction(
                image_tensor, original_emotion, original_stress
            )
        else:
            corrected_emotion, corrected_stress = original_emotion, original_stress
        
        return {
            'original_emotion': original_emotion,
            'original_stress': original_stress,
            'corrected_emotion': corrected_emotion,
            'corrected_stress': corrected_stress,
            'confidence': original_confidence,
            'image_tensor': image_tensor  # For feedback processing
        }
    
    return enhanced_predict_emotion

def calculate_stress_level_from_emotion(emotion: str, confidence: float) -> float:
    """Helper function to calculate stress level from emotion"""
    stress_mapping = {
        'angry': 8.0, 'fear': 9.0, 'sad': 7.0, 'disgust': 6.5,
        'contempt': 5.5, 'surprise': 4.0, 'neutral': 2.0, 'happy': 1.0
    }
    return stress_mapping.get(emotion, 3.0) * confidence

if __name__ == "__main__":
    # Example usage
    print("Reinforcement Learning Feedback System for Emotion Recognition")
    print("This system learns from user corrections to improve accuracy over time.")