'use client';

import { useState, useRef } from 'react';
import { motion } from 'framer-motion';
import { 
  Upload, 
  Shield, 
  Download, 
  Settings, 
  Search,
  User,
  CheckCircle,
  AlertTriangle,
  Info,
  Clock,
  Activity,
  Video,
  Image,
  BarChart3,
  Trash2
} from 'lucide-react';

export default function FirearmDetectionDashboard() {
  const [activeTab, setActiveTab] = useState('image');
  const [isProcessing, setIsProcessing] = useState(false);
  const [results, setResults] = useState<any>(null);
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [annotatedImage, setAnnotatedImage] = useState<string | null>(null);
  const [confidenceThreshold, setConfidenceThreshold] = useState(0.5);
  const [returnAnnotated, setReturnAnnotated] = useState(false);
  const [detectionHistory, setDetectionHistory] = useState<any[]>([]);
  const [detectionStats, setDetectionStats] = useState<any>(null);
  const [healthStatus, setHealthStatus] = useState<any>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleFileSelect = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (file) {
      setSelectedFile(file);
      const url = URL.createObjectURL(file);
      setPreviewUrl(url);
      setResults(null);
      setAnnotatedImage(null);
    }
  };

  const handleImageDetection = async () => {
    if (!selectedFile) return;
    
    setIsProcessing(true);
    
    try {
      const formData = new FormData();
      formData.append('file', selectedFile);
      formData.append('confidence_threshold', confidenceThreshold.toString());
      formData.append('return_annotated', returnAnnotated.toString());
      
      const response = await fetch('http://localhost:8000/detect/image', {
        method: 'POST',
        body: formData,
      });
      
      if (response.ok) {
        const data = await response.json();
        setResults(data);
        if (data.annotated_image) {
          setAnnotatedImage(`data:image/jpeg;base64,${data.annotated_image}`);
        }
      } else {
        throw new Error('Detection failed');
      }
    } catch (error) {
      console.error('Error:', error);
      alert('Detection failed. Please try again.');
    } finally {
      setIsProcessing(false);
    }
  };

  const handleVideoDetection = async () => {
    if (!selectedFile) return;
    
    setIsProcessing(true);
    
    try {
      const formData = new FormData();
      formData.append('file', selectedFile);
      formData.append('confidence_threshold', confidenceThreshold.toString());
      
      const response = await fetch('http://localhost:8000/detect/video', {
        method: 'POST',
        body: formData,
      });
      
      if (response.ok) {
        // Create download link for video
        const blob = await response.blob();
        const url = window.URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = 'detected_video.mp4';
        document.body.appendChild(a);
        a.click();
        window.URL.revokeObjectURL(url);
        document.body.removeChild(a);
        
        alert('Video processing completed! Download started.');
      } else {
        throw new Error('Video detection failed');
      }
    } catch (error) {
      console.error('Error:', error);
      alert('Video detection failed. Please try again.');
    } finally {
      setIsProcessing(false);
    }
  };

  const fetchHealthStatus = async () => {
    try {
      const response = await fetch('http://localhost:8000/health');
      if (response.ok) {
        const data = await response.json();
        setHealthStatus(data);
      }
    } catch (error) {
      console.error('Error fetching health status:', error);
    }
  };

  const fetchDetectionHistory = async () => {
    try {
      const response = await fetch('http://localhost:8000/detections/history?limit=10');
      if (response.ok) {
        const data = await response.json();
        setDetectionHistory(data.recent_detections || []);
      }
    } catch (error) {
      console.error('Error fetching detection history:', error);
    }
  };

  const fetchDetectionStats = async () => {
    try {
      const response = await fetch('http://localhost:8000/detections/stats');
      if (response.ok) {
        const data = await response.json();
        setDetectionStats(data);
      }
    } catch (error) {
      console.error('Error fetching detection stats:', error);
    }
  };

  const clearDetectionHistory = async () => {
    try {
      const response = await fetch('http://localhost:8000/detections/history', {
        method: 'DELETE',
      });
      if (response.ok) {
        setDetectionHistory([]);
        setDetectionStats(null);
        alert('Detection history cleared successfully!');
      }
    } catch (error) {
      console.error('Error clearing detection history:', error);
    }
  };

  // Fetch health status on component mount
  useState(() => {
    fetchHealthStatus();
  });

  return (
    <div className="min-h-screen bg-gray-50">
      {/* Navigation */}
      <nav className="bg-white shadow-sm border-b">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex justify-between items-center h-16">
            <div className="flex items-center space-x-3">
              <div className="w-8 h-8 bg-red-600 rounded-lg flex items-center justify-center">
                <Shield className="w-5 h-5 text-white" />
              </div>
              <span className="text-xl font-bold text-gray-900">Firearm Detection</span>
            </div>
          </div>
        </div>
      </nav>

      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        <div className="grid grid-cols-1 lg:grid-cols-4 gap-8">
          {/* Sidebar */}
          <div className="lg:col-span-1">
            <div className="card">
              <h3 className="text-lg font-semibold text-gray-900 mb-4 flex items-center">
                <Activity className="w-5 h-5 mr-2 text-red-600" />
                System Status
              </h3>
              
              {healthStatus ? (
                <div className="space-y-4">
                  <div className="flex items-center justify-between p-3 bg-gray-50 rounded-lg">
                    <span className="text-sm text-gray-600">Model Loaded</span>
                    <span className={`px-2 py-1 rounded text-xs font-medium ${
                      healthStatus.model_loaded ? 'bg-green-100 text-green-800' : 'bg-red-100 text-red-800'
                    }`}>
                      {healthStatus.model_loaded ? 'Yes' : 'No'}
                    </span>
                  </div>
                  <div className="flex items-center justify-between p-3 bg-gray-50 rounded-lg">
                    <span className="text-sm text-gray-600">GPU Available</span>
                    <span className={`px-2 py-1 rounded text-xs font-medium ${
                      healthStatus.gpu_available ? 'bg-green-100 text-green-800' : 'bg-yellow-100 text-yellow-800'
                    }`}>
                      {healthStatus.gpu_available ? 'Yes' : 'No'}
                    </span>
                  </div>
                  <div className="flex items-center justify-between p-3 bg-gray-50 rounded-lg">
                    <span className="text-sm text-gray-600">Total Detections</span>
                    <span className="font-medium">{healthStatus.total_detections}</span>
                  </div>
                  <div className="flex items-center justify-between p-3 bg-gray-50 rounded-lg">
                    <span className="text-sm text-gray-600">Uptime</span>
                    <span className="font-medium">{Math.round(healthStatus.uptime / 60)}m</span>
                  </div>
                </div>
              ) : (
                <div className="text-center py-4 text-gray-500">
                  <Activity className="w-8 h-8 mx-auto mb-2 text-gray-300" />
                  <p className="text-sm">Loading status...</p>
                </div>
              )}

              <div className="mt-6 space-y-3">
                <button
                  onClick={fetchDetectionHistory}
                  className="w-full btn-secondary text-sm"
                >
                  <BarChart3 className="w-4 h-4 mr-2" />
                  Load History
                </button>
                <button
                  onClick={fetchDetectionStats}
                  className="w-full btn-secondary text-sm"
                >
                  <Info className="w-4 h-4 mr-2" />
                  Load Stats
                </button>
                <button
                  onClick={clearDetectionHistory}
                  className="w-full bg-red-100 hover:bg-red-200 text-red-800 font-medium py-2 px-4 rounded-lg transition-colors duration-200 text-sm"
                >
                  <Trash2 className="w-4 h-4 mr-2" />
                  Clear History
                </button>
              </div>
            </div>
          </div>

          {/* Main Content */}
          <div className="lg:col-span-3">
            {/* Tab Navigation */}
            <div className="flex space-x-1 bg-white rounded-lg p-1 shadow-sm mb-6">
              {[
                { id: 'image', label: 'Image Detection', icon: Image },
                { id: 'video', label: 'Video Detection', icon: Video }
              ].map((tab) => (
                <button
                  key={tab.id}
                  onClick={() => setActiveTab(tab.id)}
                  className={`flex items-center space-x-2 px-4 py-2 rounded-md text-sm font-medium transition-colors ${
                    activeTab === tab.id
                      ? 'bg-red-600 text-white'
                      : 'text-gray-600 hover:text-gray-900 hover:bg-gray-100'
                  }`}
                >
                  <tab.icon className="w-4 h-4" />
                  <span>{tab.label}</span>
                </button>
              ))}
            </div>

            <motion.div
              key={activeTab}
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.3 }}
              className="card"
            >
              <h2 className="text-2xl font-bold text-gray-900 mb-6">
                {activeTab === 'image' ? 'Image Detection' : 'Video Detection'}
              </h2>
              
              <div className="grid md:grid-cols-2 gap-8">
                <div>
                  <div className="border-2 border-dashed border-gray-300 rounded-lg p-8 text-center hover:border-red-400 transition-colors">
                    {previewUrl ? (
                      <div className="space-y-4">
                        <img src={previewUrl} alt="Preview" className="max-w-full h-64 object-cover rounded-lg mx-auto" />
                        <button
                          onClick={() => {
                            setSelectedFile(null);
                            setPreviewUrl(null);
                            setResults(null);
                            setAnnotatedImage(null);
                          }}
                          className="btn-secondary"
                        >
                          Remove File
                        </button>
                      </div>
                    ) : (
                      <div className="space-y-4">
                        <Upload className="w-12 h-12 text-gray-400 mx-auto" />
                        <div>
                          <p className="text-lg font-medium text-gray-900">
                            Upload {activeTab === 'image' ? 'image' : 'video'}
                          </p>
                          <p className="text-gray-500">
                            {activeTab === 'image' ? 'PNG, JPG, or any image format' : 'MP4, AVI, or any video format'}
                          </p>
                        </div>
                        <button
                          onClick={() => fileInputRef.current?.click()}
                          className="btn-primary"
                        >
                          Choose File
                        </button>
                      </div>
                    )}
                    <input
                      ref={fileInputRef}
                      type="file"
                      accept={activeTab === 'image' ? 'image/*' : 'video/*'}
                      onChange={handleFileSelect}
                      className="hidden"
                    />
                  </div>

                  {activeTab === 'image' && (
                    <div className="mt-4 space-y-4">
                      <div>
                        <label className="block text-sm font-medium text-gray-700 mb-2">
                          Confidence Threshold
                        </label>
                        <input
                          type="range"
                          min="0"
                          max="1"
                          step="0.1"
                          value={confidenceThreshold}
                          onChange={(e) => setConfidenceThreshold(parseFloat(e.target.value))}
                          className="w-full"
                        />
                        <div className="text-sm text-gray-600 mt-1">
                          {confidenceThreshold}
                        </div>
                      </div>
                      
                      <div className="flex items-center">
                        <input
                          type="checkbox"
                          id="returnAnnotated"
                          checked={returnAnnotated}
                          onChange={(e) => setReturnAnnotated(e.target.checked)}
                          className="mr-2"
                        />
                        <label htmlFor="returnAnnotated" className="text-sm text-gray-700">
                          Return annotated image
                        </label>
                      </div>
                    </div>
                  )}

                  {selectedFile && (
                    <div className="mt-4">
                      <button
                        onClick={activeTab === 'image' ? handleImageDetection : handleVideoDetection}
                        disabled={isProcessing}
                        className="w-full btn-primary disabled:opacity-50 disabled:cursor-not-allowed"
                      >
                        {isProcessing ? 'Processing...' : `Detect ${activeTab === 'image' ? 'Image' : 'Video'}`}
                      </button>
                    </div>
                  )}
                </div>

                <div>
                  <h3 className="text-lg font-semibold text-gray-900 mb-4">Results</h3>
                  
                  {isProcessing ? (
                    <div className="text-center py-8">
                      <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-red-600 mx-auto mb-4"></div>
                      <p className="text-gray-600">Processing {activeTab}...</p>
                    </div>
                  ) : results ? (
                    <div className="space-y-4">
                      <div className="grid grid-cols-2 gap-4">
                        <div className="bg-red-50 p-4 rounded-lg">
                          <div className="text-sm text-red-600 font-medium">Detections</div>
                          <div className="text-2xl font-bold text-red-900">{results.detections.length}</div>
                        </div>
                        <div className="bg-blue-50 p-4 rounded-lg">
                          <div className="text-sm text-blue-600 font-medium">Processing Time</div>
                          <div className="text-2xl font-bold text-blue-900">{results.processing_time.toFixed(2)}s</div>
                        </div>
                      </div>
                      
                      {results.detections.length > 0 && (
                        <div className="bg-gray-50 p-4 rounded-lg">
                          <div className="text-sm text-gray-600 font-medium mb-2">Detected Objects</div>
                          <div className="space-y-2">
                            {results.detections.map((detection: any, index: number) => (
                              <div key={index} className="flex items-center justify-between text-sm">
                                <span className="font-medium text-gray-900">{detection.class}</span>
                                <span className="text-gray-600">{(detection.confidence * 100).toFixed(1)}%</span>
                              </div>
                            ))}
                          </div>
                        </div>
                      )}

                      {annotatedImage && (
                        <div>
                          <div className="text-sm text-gray-600 font-medium mb-2">Annotated Image</div>
                          <img src={annotatedImage} alt="Annotated" className="w-full rounded-lg shadow-md" />
                        </div>
                      )}
                    </div>
                  ) : (
                    <div className="text-center py-8 text-gray-500">
                      <Shield className="w-12 h-12 mx-auto mb-4 text-gray-300" />
                      <p>Upload a file to see detection results</p>
                    </div>
                  )}
                </div>
              </div>
            </motion.div>

            {/* Detection History */}
            {detectionHistory.length > 0 && (
              <div className="card mt-6">
                <h3 className="text-lg font-semibold text-gray-900 mb-4">Recent Detection History</h3>
                <div className="space-y-2">
                  {detectionHistory.slice(0, 5).map((record, index) => (
                    <div key={index} className="flex items-center justify-between p-3 bg-gray-50 rounded-lg">
                      <div>
                        <div className="text-sm font-medium text-gray-900">{record.filename}</div>
                        <div className="text-xs text-gray-600">{record.timestamp}</div>
                      </div>
                      <div className="text-sm text-gray-600">
                        {record.detections.length} detections
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            )}

            {/* Detection Stats */}
            {detectionStats && (
              <div className="card mt-6">
                <h3 className="text-lg font-semibold text-gray-900 mb-4">Detection Statistics</h3>
                <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                  <div className="bg-blue-50 p-3 rounded-lg">
                    <div className="text-sm text-blue-600 font-medium">Total Requests</div>
                    <div className="text-lg font-bold text-blue-900">{detectionStats.total_requests}</div>
                  </div>
                  <div className="bg-red-50 p-3 rounded-lg">
                    <div className="text-sm text-red-600 font-medium">Gun Detections</div>
                    <div className="text-lg font-bold text-red-900">{detectionStats.gun_detections}</div>
                  </div>
                  <div className="bg-green-50 p-3 rounded-lg">
                    <div className="text-sm text-green-600 font-medium">Person Detections</div>
                    <div className="text-lg font-bold text-green-900">{detectionStats.person_detections}</div>
                  </div>
                  <div className="bg-purple-50 p-3 rounded-lg">
                    <div className="text-sm text-purple-600 font-medium">Avg Processing</div>
                    <div className="text-lg font-bold text-purple-900">{detectionStats.average_processing_time.toFixed(2)}s</div>
                  </div>
                </div>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}