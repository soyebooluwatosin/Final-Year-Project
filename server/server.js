const express = require('express');
const cors = require('cors');
const bodyParser = require('body-parser');
const { spawn } = require('child_process');
const fs = require('fs');
const path = require('path');

const app = express();
const PORT = process.env.PORT || 5000;

// Middleware
app.use(cors());
app.use(bodyParser.json({ limit: '50mb' }));

// Path to your model
const MODEL_PATH = path.join(__dirname, 'model', 'emotion_model_final.pth');

// Ensure the temp directory exists
const TEMP_DIR = path.join(__dirname, 'temp');
if (!fs.existsSync(TEMP_DIR)) {
  fs.mkdirSync(TEMP_DIR, { recursive: true });
}

// Process image and run through emotion model
const analyzeImage = (imagePath) => {
  return new Promise((resolve, reject) => {
    // Run the Python script that uses your PyTorch model
    const pythonProcess = spawn('python', [
      path.join(__dirname, 'model', 'predict_emotion.py'),
      '--model', MODEL_PATH,
      '--image', imagePath
    ]);

    let result = '';
    let error = '';

    pythonProcess.stdout.on('data', (data) => {
      result += data.toString();
    });

    pythonProcess.stderr.on('data', (data) => {
      error += data.toString();
    });

    pythonProcess.on('close', (code) => {
      if (code !== 0) {
        console.error(`Python process exited with code ${code}`);
        console.error(error);
        reject(new Error(`Analysis failed: ${error}`));
      } else {
        try {
          // Parse the output from the Python script
          const lines = result.trim().split('\n');
          
          // Extract emotion and confidence
          const emotionLine = lines.find(line => line.includes('Predicted emotion:'));
          const [emotion, confidence] = emotionLine
            ? emotionLine.replace('Predicted emotion:', '').trim().split('with')
            : ['unknown', '0%'];
          
          // Extract top 3 emotions
          const top3Start = lines.indexOf('Top 3 predictions:');
          const top3 = [];
          
          if (top3Start !== -1) {
            for (let i = top3Start + 1; i < lines.length && i < top3Start + 4; i++) {
              const line = lines[i].trim();
              if (line.startsWith('  ')) {
                const [emotionName, confidenceStr] = line.trim().split(':');
                top3.push([emotionName.trim(), parseFloat(confidenceStr.replace('%', '').trim())]);
              }
            }
          }

          // Calculate stress level based on emotion
          const stressLevels = {
            'angry': 85,
            'contempt': 65,
            'disgust': 70,
            'fear': 80,
            'sad': 60,
            'surprise': 50,
            'neutral': 20,
            'happy': 10
          };

          const detectedEmotion = emotion.trim().toLowerCase();
          const stressLevel = stressLevels[detectedEmotion] || 50;
          
          resolve({
            emotion: detectedEmotion,
            confidence: parseFloat(confidence.replace('%', '').trim()),
            top3,
            stressLevel
          });
        } catch (err) {
          console.error('Error parsing Python output:', err);
          reject(new Error('Failed to parse analysis results'));
        }
      }
    });
  });
};

// API endpoint for emotion analysis
app.post('/analyze', async (req, res) => {
  try {
    const { image } = req.body;
    
    if (!image) {
      return res.status(400).json({ error: 'No image data provided' });
    }

    // Convert base64 to image file
    const base64Data = image.replace(/^data:image\/\w+;base64,/, '');
    const imageBuffer = Buffer.from(base64Data, 'base64');
    const imagePath = path.join(TEMP_DIR, `${Date.now()}.png`);
    
    fs.writeFileSync(imagePath, imageBuffer);

    // Analyze the image
    const result = await analyzeImage(imagePath);

    // Clean up the temporary file
    fs.unlinkSync(imagePath);

    res.json(result);
  } catch (error) {
    console.error('Error during analysis:', error);
    res.status(500).json({ error: error.message || 'Analysis failed' });
  }
});

// API endpoint for getting suggestions
app.post('/suggestions', (req, res) => {
  try {
    const { emotion, stressLevel } = req.body;
    
    if (!emotion) {
      return res.status(400).json({ error: 'Emotion is required' });
    }

    // Create suggestions based on emotion and stress level
    const suggestions = [];
    
    // High stress suggestions
    if (stressLevel > 70) {
      suggestions.push('Take deep breaths and count to 10 slowly');
      suggestions.push('Try a quick 5-minute meditation session');
      suggestions.push('Step outside for fresh air if possible');
      suggestions.push('Use the 5-4-3-2-1 grounding technique: identify 5 things you see, 4 things you feel, 3 things you hear, 2 things you smell, and 1 thing you taste');
    } 
    // Medium stress suggestions
    else if (stressLevel > 40) {
      suggestions.push('Consider a short walk to clear your mind');
      suggestions.push('Listen to calming music for a few minutes');
      suggestions.push('Practice mindful breathing for 2-3 minutes');
      suggestions.push('Stretch at your desk or current location');
    } 
    // Low stress or positive emotions
    else {
      suggestions.push('Maintain your current positive state');
      suggestions.push('Share your positive feelings with someone');
      suggestions.push('Express gratitude for three things in your life');
      suggestions.push('Use this positive energy to tackle a challenging task');
    }

    // Emotion-specific suggestions
    switch(emotion.toLowerCase()) {
      case 'angry':
        suggestions.push('Write down what is making you angry, then tear up the paper');
        suggestions.push('Try physical activity to release tension');
        break;
      case 'sad':
        suggestions.push('Reach out to a friend or family member');
        suggestions.push('Do something creative that you enjoy');
        break;
      case 'fear':
      case 'disgust':
        suggestions.push('Practice the 4-7-8 breathing technique');
        suggestions.push('Focus on what you can control in the situation');
        break;
      case 'contempt':
        suggestions.push('Practice empathy by considering other perspectives');
        suggestions.push('Challenge negative thoughts with positive alternatives');
        break;
      case 'surprise':
        suggestions.push('Take a moment to process the unexpected information');
        suggestions.push('Use journaling to organize your thoughts');
        break;
      case 'happy':
        suggestions.push('Engage in activities that maintain this positive state');
        suggestions.push('Share your joy with others around you');
        break;
      case 'neutral':
        suggestions.push('Set an intention for how you want to feel');
        suggestions.push('Try a quick activity that usually boosts your mood');
        break;
    }

    // Return 3-5 suggestions randomly selected from the list
    const shuffledSuggestions = suggestions.sort(() => 0.5 - Math.random());
    const selectedSuggestions = shuffledSuggestions.slice(0, Math.min(4, shuffledSuggestions.length));
    
    res.json({ suggestions: selectedSuggestions });
  } catch (error) {
    console.error('Error generating suggestions:', error);
    res.status(500).json({ error: error.message || 'Failed to generate suggestions' });
  }
});

// Start the server
app.listen(PORT, () => {
  console.log(`Server running on port ${PORT}`);
  console.log(`Model path: ${MODEL_PATH}`);
});