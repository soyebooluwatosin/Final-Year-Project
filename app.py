from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import cv2
import onnxruntime
import base64
import os
import json
import traceback
import sqlite3
import datetime
from typing import Dict, List, Optional
import threading
import pickle

app = Flask(__name__)
CORS(app)  # This allows cross-origin requests

# Configuration
MODEL_PATH = os.environ.get("MODEL_PATH", "emotion_model.onnx")
FEEDBACK_DB_PATH = "feedback.db"

# Global variables
session = None
input_name = None
output_name = None
model_loaded = False

# Define emotion classes
EMOTIONS = ['angry', 'contempt', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

# Feedback learning variables
feedback_data = []
feedback_lock = threading.Lock()

class SimpleFeedbackLearner:
    """Simple feedback learning system for emotion correction"""
    
    def __init__(self):
        self.feedback_count = 0
        self.emotion_corrections = {}  # Track correction patterns
        self.confidence_adjustments = {}  # Track confidence adjustments
        
    def add_feedback(self, predicted_emotion: str, predicted_confidence: float, 
                    correct_emotion: str, predicted_stress: float, correct_stress: float):
        """Add user feedback to the learning system"""
        self.feedback_count += 1
        
        # Track emotion correction patterns
        correction_key = f"{predicted_emotion}->{correct_emotion}"
        if correction_key not in self.emotion_corrections:
            self.emotion_corrections[correction_key] = 0
        self.emotion_corrections[correction_key] += 1
        
        # Track confidence patterns for wrong predictions
        if predicted_emotion != correct_emotion:
            conf_range = self._get_confidence_range(predicted_confidence)
            if conf_range not in self.confidence_adjustments:
                self.confidence_adjustments[conf_range] = []
            self.confidence_adjustments[conf_range].append({
                'predicted': predicted_emotion,
                'correct': correct_emotion,
                'confidence': predicted_confidence
            })
    
    def _get_confidence_range(self, confidence: float) -> str:
        """Get confidence range bucket"""
        if confidence < 0.3:
            return "low"
        elif confidence < 0.7:
            return "medium"
        else:
            return "high"
    
    def should_apply_correction(self, emotion: str, confidence: float) -> tuple[str, float]:
        """Check if we should apply a correction based on learned patterns"""
        conf_range = self._get_confidence_range(confidence)
        
        # Look for common correction patterns
        for correction_key, count in self.emotion_corrections.items():
            predicted, correct = correction_key.split('->')
            if predicted == emotion and count >= 3:  # At least 3 similar corrections
                # Check if this confidence range has issues
                if conf_range in self.confidence_adjustments:
                    similar_corrections = [
                        adj for adj in self.confidence_adjustments[conf_range]
                        if adj['predicted'] == emotion and adj['correct'] == correct
                    ]
                    if len(similar_corrections) >= 2:
                        # Apply correction with reduced confidence
                        new_confidence = confidence * 0.8  # Reduce confidence
                        return correct, new_confidence
        
        return emotion, confidence
    
    def get_stats(self) -> dict:
        """Get feedback statistics"""
        return {
            'feedback_count': self.feedback_count,
            'correction_patterns': self.emotion_corrections,
            'confidence_adjustments': len(self.confidence_adjustments)
        }

# Initialize feedback learner
feedback_learner = SimpleFeedbackLearner()

def init_feedback_database():
    """Initialize the feedback database"""
    conn = sqlite3.connect(FEEDBACK_DB_PATH)
    cursor = conn.cursor()
    
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS feedback (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            predicted_emotion TEXT,
            predicted_confidence REAL,
            predicted_stress REAL,
            correct_emotion TEXT,
            correct_stress REAL,
            user_id TEXT,
            timestamp DATETIME,
            image_data TEXT
        )
    ''')
    
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS model_stats (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            total_predictions INTEGER,
            total_feedback INTEGER,
            accuracy_rate REAL,
            timestamp DATETIME
        )
    ''')
    
    conn.commit()
    conn.close()
    print("✅ Feedback database initialized")

def load_model():
    """Load the ONNX model with proper error handling"""
    global session, input_name, output_name, model_loaded
    
    try:
        print(f"Attempting to load model from: {MODEL_PATH}")
        print(f"Model file exists: {os.path.exists(MODEL_PATH)}")
        
        if not os.path.exists(MODEL_PATH):
            print(f"ERROR: Model file not found at {MODEL_PATH}")
            return False
            
        session = onnxruntime.InferenceSession(MODEL_PATH)
        input_name = session.get_inputs()[0].name
        output_name = session.get_outputs()[0].name
        model_loaded = True
        
        print(f"✅ Model loaded successfully from {MODEL_PATH}")
        print(f"Input name: {input_name}")
        print(f"Output name: {output_name}")
        return True
        
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        print(f"Full traceback: {traceback.format_exc()}")
        model_loaded = False
        return False

def preprocess_image(image):
    """Preprocess image for model input"""
    try:
        print(f"Preprocessing image - Original shape: {image.shape}")
        
        # Resize to model input size
        image = cv2.resize(image, (224, 224))
        
        # Convert to RGB if needed (OpenCV uses BGR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Normalize with same values as in training script
        image = image.astype(np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        image = (image - mean) / std
        
        # Transpose from HWC to CHW format (PyTorch style)
        image = image.transpose(2, 0, 1)
        
        # Add batch dimension
        image = np.expand_dims(image, axis=0).astype(np.float32)
        
        print(f"Final preprocessed shape: {image.shape}")
        return image
        
    except Exception as e:
        print(f"❌ Error in preprocess_image: {e}")
        raise

def image_from_base64(base64_str):
    """Convert base64 string to OpenCV image"""
    try:
        if ',' in base64_str:
            base64_str = base64_str.split(',', 1)[1]
            
        img_bytes = base64.b64decode(base64_str)
        img_arr = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(img_arr, cv2.IMREAD_COLOR)
        
        if img is None:
            raise ValueError("Could not decode image from base64 string")
            
        return img
        
    except Exception as e:
        print(f"❌ Error in image_from_base64: {e}")
        raise

def apply_softmax(scores):
    """Apply softmax to convert logits to probabilities"""
    exp_scores = np.exp(scores - np.max(scores))
    return exp_scores / exp_scores.sum()

def calculate_stress_level(emotion, confidence):
    """Calculate stress level based on emotion and confidence"""
    stress_mapping = {
        'angry': 8.5, 'fear': 9.0, 'disgust': 7.0, 'sad': 7.0,
        'contempt': 6.0, 'surprise': 5.0, 'neutral': 3.0, 'happy': 1.0
    }
    base_stress = stress_mapping.get(emotion, 3.0)
    return min(base_stress * confidence, 10.0)

def get_suggestions(emotion, stress_level):
    """Generate personalized suggestions"""
    suggestions = []
    
    if stress_level < 4.0:
        suggestions.append("Practice gratitude and maintain healthy routines.")
    elif stress_level < 7.0:
        suggestions.append("Try structured exercise and social connection.")
    else:
        suggestions.append("Consider professional help and stress management techniques.")
    
    # Emotion-specific suggestions
    emotion_suggestions = {
        'angry': "Practice deep breathing before responding to triggers.",
        'fear': "Use grounding techniques like the 5-4-3-2-1 method.",
        'sad': "Listen to uplifting music or engage in joyful activities.",
        'happy': "Savor this positive feeling and note what contributed to it."
    }
    
    if emotion in emotion_suggestions:
        suggestions.append(emotion_suggestions[emotion])
    
    return suggestions[:3]

def adjust_emotion_confidence(emotion, confidence):
    # Reduce contempt confidence if it's often wrong
    if emotion == 'contempt' and confidence < 0.4:
        return confidence * 0.8  # Reduce by 20%
    return confidence

def store_feedback(predicted_emotion, predicted_confidence, predicted_stress, 
                  correct_emotion, correct_stress, user_id=None, image_data=None):
    """Store feedback in database"""
    try:
        conn = sqlite3.connect(FEEDBACK_DB_PATH)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO feedback 
            (predicted_emotion, predicted_confidence, predicted_stress, 
             correct_emotion, correct_stress, user_id, timestamp, image_data)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            predicted_emotion, predicted_confidence, predicted_stress,
            correct_emotion, correct_stress, user_id, 
            datetime.datetime.now().isoformat(), image_data
        ))
        
        conn.commit()
        conn.close()
        
        # Add to feedback learner
        feedback_learner.add_feedback(
            predicted_emotion, predicted_confidence, correct_emotion,
            predicted_stress, correct_stress
        )
        
        print(f"✅ Feedback stored: {predicted_emotion} -> {correct_emotion}")
        return True
        
    except Exception as e:
        print(f"❌ Error storing feedback: {e}")
        return False

# Initialize on startup
print("🚀 Starting application with reinforcement learning...")
init_feedback_database()
load_model()

@app.route('/')
def home():
    """Health check endpoint"""
    model_status = "loaded" if model_loaded else "not loaded"
    stats = feedback_learner.get_stats()
    
    return jsonify({
        'status': 'ok',
        'model': model_status,
        'message': 'Facial Emotion Analysis API with Reinforcement Learning',
        'model_path': MODEL_PATH,
        'model_exists': os.path.exists(MODEL_PATH),
        'feedback_enabled': True,
        'feedback_stats': stats
    })

@app.route('/analyze', methods=['POST'])
@app.route('/predict', methods=['POST'])
def analyze():
    """Analyze facial emotion with reinforcement learning corrections"""
    print("\n" + "="*50)
    print("🔍 NEW ANALYSIS REQUEST")
    print("="*50)
    
    try:
        if not model_loaded:
            return jsonify({
                'error': 'Model not loaded',
                'success': False,
                'emotion': 'unknown',
                'confidence': 0,
                'stressLevel': 0,
                'stressCategory': 'unknown',
                'allEmotions': {},
                'suggestions': ["Model not available"],
                'correction_applied': False,
                'feedback_count': feedback_learner.feedback_count
            }), 500
            
        # Get image data
        data = request.json
        if not data or 'image' not in data:
            return jsonify({'error': 'No image data provided', 'success': False}), 400
        
        # Process image
        img = image_from_base64(data['image'])
        processed_img = preprocess_image(img)
        
        # Run inference
        outputs = session.run([output_name], {input_name: processed_img})
        scores = outputs[0][0]
        probabilities = apply_softmax(scores)
        
        # Get prediction
        emotion_idx = np.argmax(probabilities)
        original_emotion = EMOTIONS[emotion_idx]
        original_confidence = float(probabilities[emotion_idx])
        
        print(f"🎯 Original prediction: {original_emotion} ({original_confidence:.3f})")
        
        # Apply reinforcement learning correction
        corrected_emotion, corrected_confidence = feedback_learner.should_apply_correction(
            original_emotion, original_confidence
        )
        
        correction_applied = (corrected_emotion != original_emotion)
        if correction_applied:
            print(f"🔄 Applied correction: {original_emotion} -> {corrected_emotion}")
        
        # Use corrected values
        final_emotion = corrected_emotion
        final_confidence = corrected_confidence
        
        # Calculate stress level
        stress_level = calculate_stress_level(final_emotion, final_confidence)
        
        # Get stress category
        if stress_level < 3.0:
            stress_category = "Low"
        elif stress_level < 6.0:
            stress_category = "Moderate"
        elif stress_level < 8.0:
            stress_category = "High"
        else:
            stress_category = "Severe"
        
        # Get suggestions
        suggestions = get_suggestions(final_emotion, stress_level)
        
        # Create response
        all_probs = {emotion: float(prob) for emotion, prob in zip(EMOTIONS, probabilities)}
        
        result = {
            'emotion': final_emotion,
            'confidence': final_confidence * 100,
            'stressLevel': stress_level,
            'stressCategory': stress_category,
            'allEmotions': all_probs,
            'suggestions': suggestions,
            'success': True,
            'correction_applied': correction_applied,
            'feedback_count': feedback_learner.feedback_count
        }
        
        print("✅ Analysis completed successfully!")
        return jsonify(result)
        
    except Exception as e:
        error_msg = str(e)
        print(f"❌ Analysis failed: {error_msg}")
        print(f"Traceback:\n{traceback.format_exc()}")
        
        return jsonify({
            'error': f"Analysis failed: {error_msg}",
            'success': False,
            'emotion': 'unknown',
            'confidence': 0,
            'stressLevel': 0,
            'stressCategory': 'unknown',
            'allEmotions': {},
            'suggestions': ["Error analyzing image. Please try again."],
            'correction_applied': False,
            'feedback_count': feedback_learner.feedback_count
        }), 500

# Add this to your app.py - replace the existing submit_feedback route

@app.route('/feedback', methods=['POST'])
def submit_feedback():
    """Submit user feedback for reinforcement learning"""
    print("\n🔄 FEEDBACK SUBMISSION")
    
    try:
        data = request.json
        if not data:
            return jsonify({'error': 'No feedback data provided', 'success': False}), 400
        
        # Extract feedback data
        predicted_emotion = data.get('predicted_emotion')
        predicted_confidence = data.get('predicted_confidence', 0.0)
        predicted_stress = data.get('predicted_stress', 0.0)
        correct_emotion = data.get('correct_emotion')
        correct_stress = data.get('correct_stress', 0.0)
        user_id = data.get('user_id', 'anonymous')
        
        print(f"📝 Feedback: {predicted_emotion} -> {correct_emotion}")
        print(f"📊 Stress: {predicted_stress} -> {correct_stress}")
        print(f"👤 User: {user_id}")
        
        # Store feedback
        success = store_feedback(
            predicted_emotion, predicted_confidence, predicted_stress,
            correct_emotion, correct_stress, user_id
        )
        
        if success:
            # Calculate reward (simple implementation)
            emotion_correct = predicted_emotion == correct_emotion
            stress_diff = abs(predicted_stress - correct_stress)
            
            if emotion_correct and stress_diff < 1.0:
                reward = 1.0  # Perfect match
            elif emotion_correct:
                reward = 0.8  # Emotion correct, stress close
            elif stress_diff < 2.0:
                reward = 0.5  # Stress close, emotion wrong
            else:
                reward = 0.1  # Both wrong
            
            print(f"✅ Feedback processed successfully!")
            print(f"📈 Current feedback count: {feedback_learner.feedback_count}")
            print(f"🎯 Calculated reward: {reward}")
            
            return jsonify({
                'success': True,
                'message': 'Feedback received and processed',
                'reward': reward,
                'total_feedback': feedback_learner.feedback_count
            })
        else:
            print("❌ Failed to store feedback in database")
            return jsonify({
                'success': False,
                'message': 'Failed to store feedback',
                'reward': 0.0,
                'total_feedback': feedback_learner.feedback_count
            }), 500
            
    except Exception as e:
        print(f"❌ Feedback submission failed: {e}")
        print(f"Full traceback:\n{traceback.format_exc()}")
        return jsonify({
            'success': False,
            'message': f'Feedback submission failed: {str(e)}',
            'reward': 0.0,
            'total_feedback': feedback_learner.feedback_count
        }), 500

# Also update the store_feedback function to ensure proper counting:

def store_feedback(predicted_emotion, predicted_confidence, predicted_stress, 
                  correct_emotion, correct_stress, user_id=None, image_data=None):
    """Store feedback in database and update learner"""
    try:
        # Store in database first
        conn = sqlite3.connect(FEEDBACK_DB_PATH)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO feedback 
            (predicted_emotion, predicted_confidence, predicted_stress, 
             correct_emotion, correct_stress, user_id, timestamp, image_data)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            predicted_emotion, predicted_confidence, predicted_stress,
            correct_emotion, correct_stress, user_id, 
            datetime.datetime.now().isoformat(), image_data
        ))
        
        feedback_id = cursor.lastrowid  # Get the ID of inserted record
        conn.commit()
        conn.close()
        
        print(f"✅ Stored feedback in database with ID: {feedback_id}")
        
        # Add to feedback learner (this increments the count)
        feedback_learner.add_feedback(
            predicted_emotion, predicted_confidence, correct_emotion,
            predicted_stress, correct_stress
        )
        
        print(f"✅ Updated feedback learner. New count: {feedback_learner.feedback_count}")
        print(f"📊 Correction patterns: {feedback_learner.emotion_corrections}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error storing feedback: {e}")
        print(f"Full traceback:\n{traceback.format_exc()}")
        return False

# Update the SimpleFeedbackLearner.add_feedback method to ensure counting:

class SimpleFeedbackLearner:
    """Simple feedback learning system for emotion correction"""
    
    def __init__(self):
        self.feedback_count = 0
        self.emotion_corrections = {}  # Track correction patterns
        self.confidence_adjustments = {}  # Track confidence adjustments
        
    def add_feedback(self, predicted_emotion: str, predicted_confidence: float, 
                    correct_emotion: str, predicted_stress: float, correct_stress: float):
        """Add user feedback to the learning system"""
        # INCREMENT FIRST - this is crucial!
        self.feedback_count += 1
        
        print(f"🔢 Incremented feedback count to: {self.feedback_count}")
        
        # Track emotion correction patterns
        correction_key = f"{predicted_emotion}->{correct_emotion}"
        if correction_key not in self.emotion_corrections:
            self.emotion_corrections[correction_key] = 0
        self.emotion_corrections[correction_key] += 1
        
        print(f"📝 Added correction pattern: {correction_key} (count: {self.emotion_corrections[correction_key]})")
        
        # Track confidence patterns for wrong predictions
        if predicted_emotion != correct_emotion:
            conf_range = self._get_confidence_range(predicted_confidence)
            if conf_range not in self.confidence_adjustments:
                self.confidence_adjustments[conf_range] = []
            self.confidence_adjustments[conf_range].append({
                'predicted': predicted_emotion,
                'correct': correct_emotion,
                'confidence': predicted_confidence
            })
            
            print(f"🎯 Added confidence adjustment for {conf_range} confidence range")
        
        return self.feedback_count  # Return current count
    
    def _get_confidence_range(self, confidence: float) -> str:
        """Get confidence range bucket"""
        if confidence < 0.3:
            return "low"
        elif confidence < 0.7:
            return "medium"
        else:
            return "high"
    
    def should_apply_correction(self, emotion: str, confidence: float) -> tuple[str, float]:
        """Check if we should apply a correction based on learned patterns"""
        conf_range = self._get_confidence_range(confidence)
        
        # Look for common correction patterns
        for correction_key, count in self.emotion_corrections.items():
            predicted, correct = correction_key.split('->')
            if predicted == emotion and count >= 3:  # At least 3 similar corrections
                # Check if this confidence range has issues
                if conf_range in self.confidence_adjustments:
                    similar_corrections = [
                        adj for adj in self.confidence_adjustments[conf_range]
                        if adj['predicted'] == emotion and adj['correct'] == correct
                    ]
                    if len(similar_corrections) >= 2:
                        # Apply correction with reduced confidence
                        new_confidence = confidence * 0.8  # Reduce confidence
                        print(f"🔄 Applied learned correction: {emotion} -> {correct} (confidence: {confidence:.3f} -> {new_confidence:.3f})")
                        return correct, new_confidence
        
        return emotion, confidence
    
    def get_stats(self) -> dict:
        """Get feedback statistics"""
        return {
            'feedback_count': self.feedback_count,
            'correction_patterns': self.emotion_corrections,
            'confidence_adjustments': len(self.confidence_adjustments),
            'total_corrections': sum(self.emotion_corrections.values())
        }

@app.route('/feedback/stats', methods=['GET'])
def get_feedback_stats():
    """Get feedback statistics"""
    try:
        stats = feedback_learner.get_stats()
        
        # Get database stats
        conn = sqlite3.connect(FEEDBACK_DB_PATH)
        cursor = conn.cursor()
        
        cursor.execute("SELECT COUNT(*) FROM feedback")
        total_feedback = cursor.fetchone()[0]
        
        cursor.execute("""
            SELECT correct_emotion, COUNT(*) as count 
            FROM feedback 
            GROUP BY correct_emotion 
            ORDER BY count DESC
        """)
        emotion_distribution = dict(cursor.fetchall())
        
        conn.close()
        
        stats.update({
            'database_entries': total_feedback,
            'emotion_distribution': emotion_distribution
        })
        
        return jsonify(stats)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    debug = os.environ.get('DEBUG', 'True').lower() == 'true'
    print(f"🚀 Starting Flask app with RL on port {port}")
    app.run(host='0.0.0.0', port=port, debug=debug)