// Enhanced API Service for nIsoT Stress Analysis App with Feedback Learning

const API_URL = 'http://127.0.0.1:5000';

// Define TypeScript interfaces for API responses
export interface EmotionAnalysisResult {
  emotion: string;
  confidence: number;
  stressLevel: number;
  stressCategory: string;
  allEmotions: { [key: string]: number };
  suggestions: string[];
  success: boolean;
  message?: string;
  correction_applied: boolean;
  feedback_count: number;
}

export interface FeedbackData {
  image: string; // Base64 encoded image (same as in prediction)
  predicted_emotion: string;
  predicted_stress: number;
  predicted_confidence: number; // 0-1 scale
  correct_emotion: string;
  correct_stress: number;
  user_id?: string;
}

export interface FeedbackResponse {
  success: boolean;
  message: string;
  reward: number;
  total_feedback: number;
}

export interface FeedbackStats {
  total_feedback_entries: number;
  experience_buffer_size: number;
  model_parameters: number;
  last_update: number;
  database_entries: number;
  used_for_training: number;
  unused_entries: number;
  emotion_feedback_distribution: { [key: string]: number };
}

/**
 * Analyze facial emotion from captured image
 * @param imageData - Base64 encoded image data
 * @param userId - Optional user identifier
 */
export const analyzeEmotion = async (
  imageData: string, 
  userId?: string
): Promise<EmotionAnalysisResult> => {
  try {
    const response = await fetch(`${API_URL}/predict`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ 
        image: imageData,
        user_id: userId 
      }),
    });

    if (!response.ok) {
      throw new Error(`Analysis failed: ${response.statusText}`);
    }

    const result = await response.json();
    return result;
  } catch (error) {
    console.error('Error analyzing emotion:', error);
    throw error;
  }
};

/**
 * Submit user feedback to improve the model
 * @param feedbackData - User feedback data
 */
export const submitFeedback = async (feedbackData: FeedbackData): Promise<FeedbackResponse> => {
  try {
    const response = await fetch(`${API_URL}/feedback`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(feedbackData),
    });

    if (!response.ok) {
      throw new Error(`Feedback submission failed: ${response.statusText}`);
    }

    const result = await response.json();
    return result;
  } catch (error) {
    console.error('Error submitting feedback:', error);
    throw error;
  }
};

/**
 * Get feedback and learning statistics
 */
export const getFeedbackStats = async (): Promise<FeedbackStats> => {
  try {
    const response = await fetch(`${API_URL}/feedback/stats`, {
      method: 'GET',
      headers: {
        'Content-Type': 'application/json',
      },
    });

    if (!response.ok) {
      throw new Error(`Failed to get stats: ${response.statusText}`);
    }

    const result = await response.json();
    return result;
  } catch (error) {
    console.error('Error getting feedback stats:', error);
    throw error;
  }
};

/**
 * Trigger manual retraining of the feedback model
 */
export const triggerRetraining = async (): Promise<{ success: boolean; message: string; feedback_count?: number }> => {
  try {
    const response = await fetch(`${API_URL}/feedback/retrain`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
    });

    if (!response.ok) {
      throw new Error(`Retraining failed: ${response.statusText}`);
    }

    const result = await response.json();
    return result;
  } catch (error) {
    console.error('Error triggering retraining:', error);
    throw error;
  }
};

/**
 * Check API health and model status
 */
export const checkAPIHealth = async (): Promise<{
  status: string;
  message: string;
  model_loaded: boolean;
  feedback_enabled: boolean;
}> => {
  try {
    const response = await fetch(`${API_URL}/`, {
      method: 'GET',
    });

    if (!response.ok) {
      throw new Error(`Health check failed: ${response.statusText}`);
    }

    const result = await response.json();
    return result;
  } catch (error) {
    console.error('Error checking API health:', error);
    throw error;
  }
};

/**
 * Utility function to validate feedback data before submission
 */
export const validateFeedbackData = (feedbackData: FeedbackData): { valid: boolean; errors: string[] } => {
  const errors: string[] = [];
  
  if (!feedbackData.image || feedbackData.image.length === 0) {
    errors.push("Image data is required");
  }
  
  if (!feedbackData.predicted_emotion || feedbackData.predicted_emotion.length === 0) {
    errors.push("Predicted emotion is required");
  }
  
  if (!feedbackData.correct_emotion || feedbackData.correct_emotion.length === 0) {
    errors.push("Correct emotion is required");
  }
  
  if (feedbackData.predicted_stress < 0 || feedbackData.predicted_stress > 10) {
    errors.push("Predicted stress level must be between 0 and 10");
  }
  
  if (feedbackData.correct_stress < 0 || feedbackData.correct_stress > 10) {
    errors.push("Correct stress level must be between 0 and 10");
  }
  
  if (feedbackData.predicted_confidence < 0 || feedbackData.predicted_confidence > 1) {
    errors.push("Predicted confidence must be between 0 and 1");
  }
  
  const validEmotions = ['angry', 'contempt', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise'];
  if (!validEmotions.includes(feedbackData.predicted_emotion)) {
    errors.push("Invalid predicted emotion");
  }
  
  if (!validEmotions.includes(feedbackData.correct_emotion)) {
    errors.push("Invalid correct emotion");
  }
  
  return {
    valid: errors.length === 0,
    errors
  };
};

/**
 * Batch feedback submission (for future use)
 */
export const submitBatchFeedback = async (feedbackBatch: FeedbackData[]): Promise<{
  success: boolean;
  processed: number;
  errors: string[];
}> => {
  const results = {
    success: true,
    processed: 0,
    errors: [] as string[]
  };
  
  for (let i = 0; i < feedbackBatch.length; i++) {
    try {
      const validation = validateFeedbackData(feedbackBatch[i]);
      if (!validation.valid) {
        results.errors.push(`Item ${i}: ${validation.errors.join(', ')}`);
        continue;
      }
      
      await submitFeedback(feedbackBatch[i]);
      results.processed++;
    } catch (error) {
      results.errors.push(`Item ${i}: ${error instanceof Error ? error.message : 'Unknown error'}`);
    }
  }
  
  if (results.errors.length > 0) {
    results.success = false;
  }
  
  return results;
};

// Export all functions for easy access
export default {
  analyzeEmotion,
  submitFeedback,
  getFeedbackStats,
  triggerRetraining,
  checkAPIHealth,
  validateFeedbackData,
  submitBatchFeedback
};