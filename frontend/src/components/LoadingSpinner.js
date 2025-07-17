// components/LoadingSpinner.js
import { useState, useEffect } from 'react';

const LoadingSpinner = () => {
  const [loadingStep, setLoadingStep] = useState(0);
  const [progress, setProgress] = useState(0);

  const steps = [
    { text: "Uploading file...", icon: "📤" },
    { text: "Preprocessing media...", icon: "🔧" },
    { text: "Running AI analysis...", icon: "🧠" },
    { text: "Calculating confidence...", icon: "📊" },
    { text: "Almost done...", icon: "✨" }
  ];

  useEffect(() => {
    const stepInterval = setInterval(() => {
      setLoadingStep(prev => (prev + 1) % steps.length);
    }, 2000);

    const progressInterval = setInterval(() => {
      setProgress(prev => {
        const newProgress = prev + Math.random() * 15;
        return newProgress > 95 ? 95 : newProgress;
      });
    }, 300);

    return () => {
      clearInterval(stepInterval);
      clearInterval(progressInterval);
    };
  }, []);

  return (
    <div className="mt-8 p-8 text-center">
      {/* Main spinner */}
      <div className="relative inline-block mb-6">
        <div className="animate-spin rounded-full h-16 w-16 border-4 border-indigo-200 border-t-indigo-600 mx-auto"></div>
        <div className="absolute inset-0 flex items-center justify-center">
          <span className="text-2xl animate-pulse">🔍</span>
        </div>
      </div>

      {/* Current step */}
      <div className="mb-6">
        <div className="text-3xl mb-2">
          {steps[loadingStep].icon}
        </div>
        <h3 className="text-xl font-semibold text-gray-800 mb-2">
          Analyzing Your Content
        </h3>
        <p className="text-gray-600">
          {steps[loadingStep].text}
        </p>
      </div>

      {/* Progress bar */}
      <div className="max-w-md mx-auto mb-6">
        <div className="flex justify-between items-center mb-2">
          <span className="text-sm font-medium text-gray-600">Progress</span>
          <span className="text-sm font-bold text-gray-800">{Math.round(progress)}%</span>
        </div>
        <div className="w-full bg-gray-200 rounded-full h-2">
          <div 
            className="bg-indigo-600 h-2 rounded-full transition-all duration-300 ease-out"
            style={{ width: `${progress}%` }}
          />
        </div>
      </div>

      {/* Fun facts while waiting */}
      <div className="bg-blue-50 border border-blue-200 rounded-lg p-4 max-w-md mx-auto">
        <h4 className="font-semibold text-blue-800 mb-2">💡 Did you know?</h4>
        <p className="text-blue-700 text-sm">
          {loadingStep === 0 && "AI image generators create images pixel by pixel using neural networks trained on millions of images."}
          {loadingStep === 1 && "Our detection models look for subtle patterns that are invisible to the human eye but characteristic of AI generation."}
          {loadingStep === 2 && "Deepfake detection analyzes facial movements, lighting consistency, and temporal artifacts frame by frame."}
          {loadingStep === 3 && "The confidence score is calculated based on multiple detection algorithms working together."}
          {loadingStep === 4 && "Even the best AI detectors aren't perfect - that's why we provide explanations with our results!"}
        </p>
      </div>

      {/* Estimated time */}
      <div className="mt-4 text-sm text-gray-500">
        <p>This usually takes 10-30 seconds depending on file size</p>
      </div>
    </div>
  );
};

export default LoadingSpinner;