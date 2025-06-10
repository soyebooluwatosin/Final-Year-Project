# enhanced_main.py
"""
Enhanced FastAPI backend with reinforcement learning feedback system
"""
from fastapi import FastAPI, HTTPException, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import numpy as np
import cv2
import onnxruntime
import base64
from typing import Dict, List, Optional
import io
from PIL import Image
import torch
import os
from datetime import datetime

# Import your existing modules
from stress_analysis import calculate_stress_level, generate_suggestions
from rl_feedback_system import ReinforcementFeedbackTrainer, FeedbackDatabase

app = FastAPI(
    title="Enhanced Facial Emotion Recognition API with Feedback Learning",
    description="API for analyzing facial expressions with reinforcement learning from user feedback",
    version="2.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables for model and feedback system
model_loaded = False
feedback_trainer = None
session = None
input_name = None
output_name = None

# Define emotion classes
EMOTIONS = ['angry', 'contempt', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

# Initialize models and feedback system
def initialize_system():
    global model_loaded, feedback_trainer, session, input_name, output_name
    
    try:
        # Load ONNX model
        session = onnxruntime.InferenceSession("emotion_model.onnx")
        input_name = session.get_inputs()[0].name
        output_name = session.get_outputs()[0].name
        model_loaded = True
        print("ONNX model loaded successfully")
        
        # Initialize feedback trainer (without base model for now)
        feedback_trainer = ReinforcementFeedbackTrainer(
            base_model=None,  # We'll handle features differently
            emotion_classes=EMOTIONS,
            feature_dim=1280
        )
        
        # Try to load existing feedback model
        if os.path.exists("feedback_model.pth"):
            feedback_trainer.load_model("feedback_model.pth")
            print("Loaded existing feedback model")
        
        print("Feedback learning system initialized")
        
    except Exception as e:
        print(f"Error initializing system: {e}")
        print("Server will start with limited functionality")

# Initialize on startup
initialize_system()

# Request/Response models
class PredictionRequest(BaseModel):
    image: str  # Base64 encoded image
    user_id: Optional[str] = None

class FeedbackRequest(BaseModel):
    image: str  # Base64 encoded image (same as in prediction)
    predicted_emotion: str
    predicted_stress: float
    predicted_confidence: float
    correct_emotion: str
    correct_stress: float
    user_id: Optional[str] = None

class PredictionResponse(BaseModel):
    emotion: str
    confidence: float
    stressLevel: float
    stressCategory: str
    allEmotions: Dict[str, float]
    suggestions: List[str]
    success: bool
    message: Optional[str] = None
    correction_applied: bool = False
    feedback_count: int = 0

class FeedbackResponse(BaseModel):
    success: bool
    message: str
    reward: float
    total_feedback: int

# Helper functions
def preprocess_image(image):
    """Preprocess the image for model input."""
    image = cv2.resize(image, (224, 224))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = image.astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = (image - mean) / std
    image = image.transpose(2, 0, 1)
    image = np.expand_dims(image, axis=0)
    return image

def image_from_base64(base64_str):
    """Convert base64 string to OpenCV image."""
    if ',' in base64_str:
        base64_str = base64_str.split(',', 1)[1]
    img_bytes = base64.b64decode(base64_str)
    img_arr = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(img_arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Could not decode image from base64 string")
    return img

def apply_softmax(scores):
    """Apply softmax to convert logits to probabilities."""
    exp_scores = np.exp(scores - np.max(scores))
    return exp_scores / exp_scores.sum()

def extract_features_from_onnx(image):
    """Extract features for feedback learning from preprocessed image."""
    # For now, we'll use the final layer before classification
    # In a real implementation, you'd modify your ONNX model to output features
    processed_img = preprocess_image(image)
    outputs = session.run([output_name], {input_name: processed_img})
    
    # Use the logits as features (simplified approach)
    # In practice, you'd want to extract from an earlier layer
    features = outputs[0][0]
    return features

# API Endpoints
@app.get("/")
async def root():
    """Health check endpoint."""
    return {
        "status": "ok", 
        "message": "Enhanced Facial Emotion Recognition API with Feedback Learning is running",
        "model_loaded": model_loaded,
        "feedback_enabled": feedback_trainer is not None
    }

@app.post("/predict", response_model=PredictionResponse)
async def predict_emotion(request: PredictionRequest):
    """Analyze facial emotion from a base64 encoded image with feedback correction."""
    
    if not model_loaded:
        return PredictionResponse(
            emotion="happy",
            confidence=0.75,
            stressLevel=3.5,
            stressCategory="Low",
            allEmotions={emotion: 0.125 for emotion in EMOTIONS},
            suggestions=["This is a mock response. Please upload the model file."],
            success=True,
            message="Using mock data - model not loaded",
            correction_applied=False,
            feedback_count=0
        )
    
    try:
        # Decode and preprocess the image
        img = image_from_base64(request.image)
        processed_img = preprocess_image(img)
        
        # Run inference
        outputs = session.run([output_name], {input_name: processed_img})
        scores = outputs[0][0]
        
        # Convert scores to probabilities
        probabilities = apply_softmax(scores)
        
        # Create emotion probabilities dictionary
        emotion_probs = {emotion: float(prob) for emotion, prob in zip(EMOTIONS, probabilities)}
        
        # Get the main emotion
        emotion_idx = np.argmax(probabilities)
        original_emotion = EMOTIONS[emotion_idx]
        confidence = float(probabilities[emotion_idx])
        
        # Calculate stress level using existing logic
        stress_level, detected_emotion = calculate_stress_level(emotion_probs)
        
        # Apply feedback correction if available
        correction_applied = False
        final_emotion = original_emotion
        final_stress = stress_level
        
        if feedback_trainer and feedback_trainer.feedback_count > 50:  # Only after sufficient feedback
            try:
                # Extract features for correction
                features = extract_features_from_onnx(img)
                
                # Get corrected prediction
                corrected_emotion, corrected_stress = feedback_trainer.get_corrected_prediction(
                    torch.FloatTensor(features).unsqueeze(0), original_emotion, stress_level
                )
                
                # Apply correction with confidence threshold
                if confidence < 0.8:  # Only correct when original prediction is uncertain
                    final_emotion = corrected_emotion
                    final_stress = corrected_stress
                    correction_applied = True
                    
            except Exception as e:
                print(f"Error applying feedback correction: {e}")
        
        # Get stress category
        stress_category = "Low"
        if final_stress > 3.0 and final_stress <= 6.0:
            stress_category = "Moderate"
        elif final_stress > 6.0 and final_stress <= 8.0:
            stress_category = "High"
        elif final_stress > 8.0:
            stress_category = "Severe"
        
        # Generate suggestions
        suggestions = generate_suggestions(final_emotion, final_stress)
        
        return PredictionResponse(
            emotion=final_emotion,
            confidence=confidence,
            stressLevel=final_stress,
            stressCategory=stress_category,
            allEmotions=emotion_probs,
            suggestions=suggestions,
            success=True,
            correction_applied=correction_applied,
            feedback_count=feedback_trainer.feedback_count if feedback_trainer else 0
        )
        
    except Exception as e:
        print(f"Error during prediction: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={
                "success": False,
                "message": f"Error analyzing image: {str(e)}",
                "emotion": "unknown",
                "confidence": 0.0,
                "stressLevel": 0.0,
                "stressCategory": "unknown",
                "allEmotions": {},
                "suggestions": ["Try again with a clearer image of a face"],
                "correction_applied": False,
                "feedback_count": 0
            }
        )

@app.post("/feedback", response_model=FeedbackResponse)
async def submit_feedback(request: FeedbackRequest):
    """Submit user feedback to improve the model."""
    
    if not feedback_trainer:
        return FeedbackResponse(
            success=False,
            message="Feedback system not initialized",
            reward=0.0,
            total_feedback=0
        )
    
    try:
        # Decode image and extract features
        img = image_from_base64(request.image)
        features = extract_features_from_onnx(img)
        
        # Create a dummy tensor for the feedback system
        # In practice, you'd pass the actual image tensor
        dummy_tensor = torch.FloatTensor(features).unsqueeze(0)
        
        # Process the feedback
        reward = feedback_trainer.process_feedback(
            image_tensor=dummy_tensor,
            predicted_emotion=request.predicted_emotion,
            predicted_stress=request.predicted_stress,
            correct_emotion=request.correct_emotion,
            correct_stress=request.correct_stress,
            confidence=request.predicted_confidence,
            user_id=request.user_id
        )
        
        # Save feedback model periodically
        if feedback_trainer.feedback_count % 100 == 0:
            feedback_trainer.save_model("feedback_model.pth")
            print(f"Saved feedback model at {feedback_trainer.feedback_count} entries")
        
        return FeedbackResponse(
            success=True,
            message="Feedback processed successfully",
            reward=reward,
            total_feedback=feedback_trainer.feedback_count
        )
        
    except Exception as e:
        print(f"Error processing feedback: {str(e)}")
        return FeedbackResponse(
            success=False,
            message=f"Error processing feedback: {str(e)}",
            reward=0.0,
            total_feedback=feedback_trainer.feedback_count if feedback_trainer else 0
        )

@app.get("/feedback/stats")
async def get_feedback_stats():
    """Get feedback and learning statistics."""
    
    if not feedback_trainer:
        return {"error": "Feedback system not initialized"}
    
    stats = feedback_trainer.get_statistics()
    
    # Add database statistics
    db = FeedbackDatabase()
    try:
        import sqlite3
        conn = sqlite3.connect(db.db_path)
        cursor = conn.cursor()
        
        cursor.execute("SELECT COUNT(*) FROM feedback")
        total_entries = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(*) FROM feedback WHERE used_for_training = TRUE")
        used_entries = cursor.fetchone()[0]
        
        cursor.execute("""
            SELECT correct_emotion, COUNT(*) as count 
            FROM feedback 
            GROUP BY correct_emotion 
            ORDER BY count DESC
        """)
        emotion_distribution = dict(cursor.fetchall())
        
        conn.close()
        
        stats.update({
            "database_entries": total_entries,
            "used_for_training": used_entries,
            "unused_entries": total_entries - used_entries,
            "emotion_feedback_distribution": emotion_distribution
        })
        
    except Exception as e:
        stats["database_error"] = str(e)
    
    return stats

@app.post("/feedback/retrain")
async def trigger_retraining():
    """Manually trigger retraining with accumulated feedback."""
    
    if not feedback_trainer:
        return {"error": "Feedback system not initialized"}
    
    try:
        # Force training with current buffer
        feedback_trainer.train_step()
        
        # Save the updated model
        feedback_trainer.save_model("feedback_model.pth")
        
        return {
            "success": True,
            "message": "Retraining completed",
            "feedback_count": feedback_trainer.feedback_count
        }
        
    except Exception as e:
        return {
            "success": False,
            "message": f"Retraining failed: {str(e)}"
        }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)