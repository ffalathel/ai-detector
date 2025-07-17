// components/FileUpload.js
import { useState, useRef } from 'react';

const FileUpload = ({ onFileSelect, selectedFile }) => {
  const [dragActive, setDragActive] = useState(false);
  const inputRef = useRef(null);

  const handleFiles = (files) => {
    if (files && files[0]) {
      const file = files[0];
      
      // Check file type
      const validTypes = ['image/jpeg', 'image/png', 'image/gif', 'image/webp', 'video/mp4', 'video/webm', 'video/mov'];
      if (!validTypes.includes(file.type)) {
        alert('Please select a valid image (JPEG, PNG, GIF, WebP) or video (MP4, WebM, MOV) file.');
        return;
      }

      // Check file size (50MB limit)
      if (file.size > 50 * 1024 * 1024) {
        alert('File size must be less than 50MB.');
        return;
      }

      onFileSelect(file);
    }
  };

  const handleDrag = (e) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.type === 'dragenter' || e.type === 'dragover') {
      setDragActive(true);
    } else if (e.type === 'dragleave') {
      setDragActive(false);
    }
  };

  const handleDrop = (e) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);
    
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      handleFiles(e.dataTransfer.files);
    }
  };

  const handleChange = (e) => {
    e.preventDefault();
    if (e.target.files && e.target.files[0]) {
      handleFiles(e.target.files);
    }
  };

  const openFileDialog = () => {
    inputRef.current?.click();
  };

  const formatFileSize = (bytes) => {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
  };

  const getFileIcon = (type) => {
    if (type.startsWith('image/')) return '🖼️';
    if (type.startsWith('video/')) return '🎥';
    return '📄';
  };

  return (
    <div className="w-full">
      <input
        ref={inputRef}
        type="file"
        accept="image/*,video/*"
        onChange={handleChange}
        className="hidden"
      />
      
      <div
        className={`relative border-2 border-dashed rounded-xl p-8 text-center transition-all duration-200 ${
          dragActive 
            ? 'border-indigo-500 bg-indigo-50' 
            : 'border-gray-300 hover:border-indigo-400 hover:bg-gray-50'
        }`}
        onDragEnter={handleDrag}
        onDragLeave={handleDrag}
        onDragOver={handleDrag}
        onDrop={handleDrop}
      >
        {!selectedFile ? (
          <div>
            <div className="text-6xl mb-4">📁</div>
            <h3 className="text-xl font-semibold text-gray-700 mb-2">
              Drop your file here
            </h3>
            <p className="text-gray-500 mb-4">
              Or click to browse for images and videos
            </p>
            <button
              onClick={openFileDialog}
              className="bg-indigo-600 hover:bg-indigo-700 text-white font-medium py-2 px-6 rounded-lg transition-colors duration-200"
            >
              Choose File
            </button>
            <p className="text-sm text-gray-400 mt-4">
              Supports: JPEG, PNG, GIF, WebP, MP4, WebM, MOV • Max 50MB
            </p>
          </div>
        ) : (
          <div>
            <div className="text-6xl mb-4">
              {getFileIcon(selectedFile.type)}
            </div>
            <h3 className="text-xl font-semibold text-gray-700 mb-2">
              File Selected
            </h3>
            <div className="bg-gray-100 rounded-lg p-4 mb-4 max-w-md mx-auto">
              <p className="font-medium text-gray-800 truncate">
                {selectedFile.name}
              </p>
              <p className="text-sm text-gray-600">
                {formatFileSize(selectedFile.size)} • {selectedFile.type}
              </p>
            </div>
            
            {selectedFile.type.startsWith('image/') && (
              <div className="mb-4">
                <img
                  src={URL.createObjectURL(selectedFile)}
                  alt="Preview"
                  className="max-w-xs max-h-48 mx-auto rounded-lg shadow-md"
                />
              </div>
            )}
            
            {selectedFile.type.startsWith('video/') && (
              <div className="mb-4">
                <video
                  src={URL.createObjectURL(selectedFile)}
                  controls
                  className="max-w-xs max-h-48 mx-auto rounded-lg shadow-md"
                />
              </div>
            )}
            
            <button
              onClick={openFileDialog}
              className="text-indigo-600 hover:text-indigo-800 font-medium underline"
            >
              Choose Different File
            </button>
          </div>
        )}
      </div>
    </div>
  );
};

export default FileUpload;