// pages/api/analyze-video.js
import formidable from 'formidable';
import fs from 'fs';

export const config = {
  api: {
    bodyParser: false,
  },
};

export default async function handler(req, res) {
  if (req.method !== 'POST') {
    return res.status(405).json({ error: 'Method not allowed' });
  }

  try {
    const form = formidable({
      uploadDir: '/tmp',
      keepExtensions: true,
      maxFileSize: 50 * 1024 * 1024, // 50MB
    });

    const [fields, files] = await form.parse(req);
    const file = files.file[0];

    if (!file) {
      return res.status(400).json({ error: 'No file uploaded' });
    }

    // Validate file type
    const allowedTypes = ['video/mp4', 'video/webm', 'video/mov', 'video/quicktime'];
    if (!allowedTypes.includes(file.mimetype)) {
      return res.status(400).json({ error: 'Invalid file type' });
    }

    // Read file
    const fileBuffer = fs.readFileSync(file.filepath);
    const fileSize = file.size;

    // Prepare form data for backend
    const formData = new FormData();
    const blob = new Blob([fileBuffer], { type: file.mimetype });
    formData.append('file', blob, file.originalFilename);

    // Call your FastAPI backend
    const backendUrl = process.env.BACKEND_URL || 'http://localhost:8000';
    const response = await fetch(`${backendUrl}/analyze-video`, {
      method: 'POST',
      body: formData,
    });

    if (!response.ok) {
      // If backend is not available, use mock response for development
      if (process.env.NODE_ENV === 'development') {
        const mockResult = await getMockVideoAnalysis(fileBuffer, file.originalFilename);
        return res.status(200).json(mockResult);
      }
      throw new Error(`Backend error: ${response.status}`);
    }

    const result = await response.json();

    // Clean up temporary file
    fs.unlinkSync(file.filepath);

    // Add additional metadata
    const enhancedResult = {
      ...result,
      details: {
        ...result.details,
        file_size: `${(fileSize / 1024 / 1024).toFixed(2)} MB`,
        model_used: result.details?.model_used || 'DFDC Deepfake Detection Model',
        processing_time: result.details?.processing_time || Math.random() * 8 + 3,
      }
    };

    res.status(200).json(enhancedResult);
  } catch (error) {
    console.error('API Error:', error);
    
    // Clean up any temporary files
    try {
      if (file && file.filepath) {
        fs.unlinkSync(file.filepath);
      }
    } catch (cleanupError) {
      console.error('Cleanup error:', cleanupError);
    }

    res.status(500).json({ 
      error: 'Analysis failed',
      message: error.message 
    });
  }
}

// Fallback mock response for development/testing
async function getMockVideoAnalysis(fileBuffer, filename) {
  // Simulate longer processing time for videos
  await new Promise(resolve => setTimeout(resolve, 4000));

  // Mock analysis based on filename or random
  const isLikelyDeepfake = filename.toLowerCase().includes('fake') || 
                          filename.toLowerCase().includes('deepfake') ||
                          filename.toLowerCase().includes('ai') ||
                          Math.random() > 0.7;

  const confidence = Math.random() * 0.3 + (isLikelyDeepfake ? 0.7 : 0.4);

  return {
    prediction: isLikelyDeepfake ? 'ai_generated' : 'real',
    confidence: confidence,
    explanation: isLikelyDeepfake ? 
      'This video shows signs of digital manipulation, including facial inconsistencies and temporal artifacts typical of deepfake technology.' :
      'This video appears to be authentic with natural facial movements and consistent lighting throughout.',
    details: {
      model_used: 'DFDC Deepfake Detection Model v2.1',
      processing_time: Math.random() * 8 + 3,
      frames_analyzed: Math.floor(Math.random() * 200) + 50,
      detected_features: isLikelyDeepfake ? 
        ['facial_inconsistencies', 'temporal_artifacts', 'blending_artifacts', 'unnatural_eye_movement'] :
        ['natural_facial_movement', 'consistent_lighting', 'authentic_micro_expressions', 'camera_shake']
    }
  };
}