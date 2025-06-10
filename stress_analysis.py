# stress_analysis.py
"""
Advanced stress analysis functions for emotion recognition system.
This module provides functions to analyze emotion data and produce stress levels
and personalized suggestions.
"""

import numpy as np
from typing import Dict, List, Tuple

# Map emotions to stress impact factors (0-10 scale)
EMOTION_STRESS_IMPACT = {
    'angry': 8.5,
    'contempt': 6.0,
    'disgust': 7.0,
    'fear': 9.0,
    'happy': 1.0,
    'neutral': 3.0,
    'sad': 7.0,
    'surprise': 5.0
}

def calculate_stress_level(emotion_probabilities: Dict[str, float]) -> Tuple[float, str]:
    """
    Calculate stress level based on emotion probabilities.
    
    Args:
        emotion_probabilities: Dictionary mapping emotion names to their probability values
        
    Returns:
        Tuple of (stress_level, primary_emotion)
    """
    if not emotion_probabilities:
        return 0.0, "neutral"
    
    # Find the primary emotion (highest probability)
    primary_emotion = max(emotion_probabilities.items(), key=lambda x: x[1])[0]
    
    # Base stress on primary emotion impact
    base_stress = EMOTION_STRESS_IMPACT.get(primary_emotion, 3.0)
    
    # Calculate weighted stress level considering all emotions
    weighted_stress = 0
    total_weight = 0
    
    for emotion, probability in emotion_probabilities.items():
        if probability > 0.05:  # Only consider emotions with >5% probability
            impact = EMOTION_STRESS_IMPACT.get(emotion, 3.0)
            weighted_stress += impact * probability
            total_weight += probability
    
    # Normalize to 0-10 scale
    if total_weight > 0:
        final_stress = min(weighted_stress / total_weight, 10.0)
    else:
        final_stress = base_stress
        
    return final_stress, primary_emotion

def get_stress_category(stress_level: float) -> str:
    """Categorize stress level into descriptive categories."""
    if stress_level < 3.0:
        return "Low"
    elif stress_level < 6.0:
        return "Moderate"
    elif stress_level < 8.0:
        return "High"
    else:
        return "Severe"

def generate_suggestions(emotion: str, stress_level: float) -> List[str]:
    """
    Generate personalized suggestions based on emotion and stress level.
    
    Args:
        emotion: Primary detected emotion
        stress_level: Calculated stress level (0-10)
        
    Returns:
        List of suggestion strings
    """
    suggestions = []
    stress_category = get_stress_category(stress_level)
    
    # Common suggestions based on stress level
    if stress_level > 7.0:
        suggestions.append("Take 5 deep breaths, inhaling for 4 counts and exhaling for 6 counts")
        suggestions.append("Consider stepping away from your current situation for a 10-minute break")
    elif stress_level > 5.0:
        suggestions.append("Practice mindful breathing for 2-3 minutes")
        suggestions.append("Drink some water and stretch your body")
    
    # Emotion-specific suggestions
    if emotion == 'angry':
        suggestions.append("Count slowly to 10 before responding to the situation")
        suggestions.append("Write down what's bothering you to gain perspective")
        if stress_level > 6.0:
            suggestions.append("Try a physical release like a brisk walk or stretching")
            
    elif emotion == 'fear':
        suggestions.append("Practice the 5-4-3-2-1 grounding technique: identify 5 things you see, 4 things you feel, 3 things you hear, 2 things you smell, and 1 thing you taste")
        suggestions.append("Remind yourself that you are safe in this moment")
        if stress_level > 6.0:
            suggestions.append("Consider talking to someone you trust about your concerns")
            
    elif emotion == 'sad':
        suggestions.append("Listen to uplifting music that you enjoy")
        suggestions.append("Recall a positive memory or achievement")
        if stress_level > 6.0:
            suggestions.append("Reach out to a friend or family member")
            
    elif emotion == 'disgust':
        suggestions.append("Shift your attention to something pleasant or neutral")
        suggestions.append("If appropriate, remove yourself from the situation")
        
    elif emotion == 'contempt':
        suggestions.append("Practice empathy by considering alternative perspectives")
        suggestions.append("Reflect on whether your reaction is proportional to the situation")
        
    elif emotion == 'surprise':
        suggestions.append("Take a moment to process the unexpected information")
        suggestions.append("Break down any complex situation into manageable parts")
        
    elif emotion == 'neutral':
        suggestions.append("Take this opportunity to practice mindfulness and stay present")
        if stress_level < 3.0:
            suggestions.append("Great job maintaining composure and emotional balance")
            
    elif emotion == 'happy':
        suggestions.append("Savor this positive feeling and note what contributed to it")
        suggestions.append("Share your positive experience with someone else")
    
    # General wellness suggestions
    if len(suggestions) < 3:
        suggestions.append("Take a short walk if possible")
        
    if len(suggestions) < 3:
        suggestions.append("Practice gratitude by noting three things you appreciate")
    
    # Return a maximum of 3 suggestions
    return suggestions[:3]