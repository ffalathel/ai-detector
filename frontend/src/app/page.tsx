// pages/index.js
"use client"
import { useState } from 'react';
import Head from 'next/head';
import FileUpload from '../components/FileUpload';
import ResultDisplay from '../components/ResultDisplay';
import LoadingSpinner from '../components/LoadingSpinner';

export default function Home() {
  const [file, setFile] = useState(null);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const handleFileSelect = (selectedFile) => {
    setFile(selectedFile);
    setResult(null);
    setError(null);
  };

  const handleAnalyze = async () => {
    if (!file) return;

    setLoading(true);
    setError(null);

    try {
      const formData = new FormData();
      formData.append('file', file);

      const endpoint = file.type.startsWith('video/') ? '/api/analyze-video' : '/api/analyze-image';
      
      const response = await fetch(endpoint, {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        throw new Error('Analysis failed');
      }

      const data = await response.json();
      setResult(data);
    } catch (err) {
      setError('Sorry, something went wrong. Please try again.');
      console.error('Analysis error:', err);
    } finally {
      setLoading(false);
    }
  };

  const handleReset = () => {
    setFile(null);
    setResult(null);
    setError(null);
  };

  return (
    <>
      <Head>
        <title>AI Content Detector - Is it Real or AI-Generated?</title>
        <meta name="description" content="Detect if images and videos are AI-generated or real. Simple tool for everyone." />
        <meta name="viewport" content="width=device-width, initial-scale=1" />
        <link rel="icon" href="/favicon.ico" />
      </Head>

      <div className="min-h-screen bg-gradient-to-br from-blue-50 to-indigo-100">
        <div className="container mx-auto px-4 py-8">
          {/* Header */}
          <div className="text-center mb-12">
            <h1 className="text-4xl md:text-5xl font-bold text-gray-800 mb-4">
              🧠✨ AI Content Detector
            </h1>
            <p className="text-lg text-gray-600 max-w-2xl mx-auto">
              A simple, friendly tool to help you figure out if an image or video is AI-generated. 
              Perfect for everyone - no tech knowledge required!
            </p>
          </div>

          {/* Main Content */}
          <div className="max-w-4xl mx-auto">
            {!result ? (
              <div className="bg-white rounded-2xl shadow-xl p-8">
                <div className="text-center mb-8">
                  <h2 className="text-2xl font-semibold text-gray-800 mb-2">
                    Upload Your Media
                  </h2>
                  <p className="text-gray-600">
                    Choose an image or video to analyze
                  </p>
                </div>

                <FileUpload 
                  onFileSelect={handleFileSelect}
                  selectedFile={file}
                />

                {error && (
                  <div className="mt-6 p-4 bg-red-50 border border-red-200 rounded-lg">
                    <p className="text-red-700 text-center">{error}</p>
                  </div>
                )}

                {file && !loading && (
                  <div className="mt-8 text-center">
                    <button
                      onClick={handleAnalyze}
                      className="bg-indigo-600 hover:bg-indigo-700 text-white font-medium py-3 px-8 rounded-lg transition-colors duration-200 shadow-lg hover:shadow-xl"
                    >
                      🔍 Analyze Content
                    </button>
                  </div>
                )}

                {loading && <LoadingSpinner />}
              </div>
            ) : (
              <ResultDisplay 
                result={result}
                fileName={file?.name}
                onReset={handleReset}
              />
            )}
          </div>

          {/* Info Cards */}
          <div className="grid md:grid-cols-2 gap-6 mt-12 max-w-4xl mx-auto">
            <div className="bg-white rounded-xl shadow-lg p-6">
              <h3 className="text-xl font-semibold text-gray-800 mb-3">
                🖼️ Image Detection
              </h3>
              <p className="text-gray-600">
                Our custom-trained model can detect AI-generated images from tools like 
                Midjourney, DALL·E, and Stable Diffusion.
              </p>
            </div>
            
            <div className="bg-white rounded-xl shadow-lg p-6">
              <h3 className="text-xl font-semibold text-gray-800 mb-3">
                🎥 Video Detection
              </h3>
              <p className="text-gray-600">
                Advanced deepfake detection using research-grade models to identify 
                manipulated or AI-generated videos.
              </p>
            </div>
          </div>
        </div>
      </div>
    </>
  );
}