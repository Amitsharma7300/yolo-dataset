'use client';

import axios from 'axios';
import { Camera, CheckCircle, Info, Loader2, Upload, Video, VideoOff, XCircle, Zap } from 'lucide-react';
import { useCallback, useEffect, useRef, useState } from 'react';

interface Detection {
  class_id: number;
  class_name: string;
  confidence: number;
  bbox: {
    x1: number;
    y1: number;
    x2: number;
    y2: number;
    width: number;
    height: number;
  };
}

interface DetectionResult {
  success: boolean;
  detections: Detection[];
  total_detections: number;
  result_image: string;
  timestamp?: string;
}

const CLASS_COLORS: { [key: string]: string } = {
  part1: '#22c55e', // Green
  part2: '#3b82f6', // Blue
  part3: '#ef4444', // Red
  part4: '#eab308', // Yellow
};

export default function Home() {
  // Image upload state
  const [selectedImage, setSelectedImage] = useState<string | null>(null);
  const [resultImage, setResultImage] = useState<string | null>(null);
  const [detections, setDetections] = useState<Detection[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [dragActive, setDragActive] = useState(false);
  const fileInputRef = useRef<HTMLInputElement>(null);

  // Real-time camera state
  const [mode, setMode] = useState<'upload' | 'camera'>('upload');
  const [cameraActive, setCameraActive] = useState(false);
  const [cameraStream, setCameraStream] = useState<MediaStream | null>(null);
  const [fps, setFps] = useState(0);
  const [isDetecting, setIsDetecting] = useState(false);
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const frameCountRef = useRef<number>(0);
  const isDetectingRef = useRef<boolean>(false);
  const cameraActiveRef = useRef<boolean>(false);
  const isProcessingRef = useRef<boolean>(false);
  const detectionIntervalRef = useRef<NodeJS.Timeout | null>(null);

  // Cleanup camera on unmount
  useEffect(() => {
    return () => {
      stopCamera();
    };
  }, []);

  // FPS counter - only count when detecting
  useEffect(() => {
    const fpsInterval = setInterval(() => {
      if (isDetectingRef.current) {
        setFps(frameCountRef.current);
      }
      frameCountRef.current = 0;
    }, 1000);
    return () => clearInterval(fpsInterval);
  }, []);

  const startCamera = async () => {
    try {
      setError(null);
      const stream = await navigator.mediaDevices.getUserMedia({
        video: {
          width: { ideal: 480 },
          height: { ideal: 360 },
          facingMode: 'environment',
          frameRate: { ideal: 30 }
        }
      });
      
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        await videoRef.current.play();
        // Wait for video to be ready
        await new Promise<void>((resolve) => {
          if (videoRef.current!.videoWidth > 0) {
            resolve();
          } else {
            videoRef.current!.onloadedmetadata = () => {
              resolve();
            };
          }
        });
      }
      
      setCameraStream(stream);
      setCameraActive(true);
      cameraActiveRef.current = true;
    } catch (err: any) {
      console.error('Camera error:', err);
      setError('Failed to access camera. Please allow camera permissions.');
    }
  };

  const stopCamera = () => {
    // Clear detection interval first
    if (detectionIntervalRef.current) {
      clearInterval(detectionIntervalRef.current);
      detectionIntervalRef.current = null;
    }
    
    isDetectingRef.current = false;
    cameraActiveRef.current = false;
    
    if (cameraStream) {
      cameraStream.getTracks().forEach(track => track.stop());
      setCameraStream(null);
    }
    
    if (videoRef.current) {
      videoRef.current.srcObject = null;
    }
    
    setCameraActive(false);
    setIsDetecting(false);
    setResultImage(null);
    setDetections([]);
  };

  const captureFrame = (): string | null => {
    if (!videoRef.current || !canvasRef.current) {
      return null;
    }
    
    const video = videoRef.current;
    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d');
    
    if (!ctx || video.videoWidth === 0) {
      return null;
    }
    
    // Use smaller resolution for faster transfer
    const targetWidth = 320;
    const scale = targetWidth / video.videoWidth;
    canvas.width = targetWidth;
    canvas.height = video.videoHeight * scale;
    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
    
    return canvas.toDataURL('image/jpeg', 0.5);
  };

  const processFrame = async () => {
    if (!isDetectingRef.current || !cameraActiveRef.current || isProcessingRef.current) {
      return;
    }
    
    isProcessingRef.current = true;
    const frameData = captureFrame();
    
    if (frameData) {
      try {
        const response = await axios.post<DetectionResult>(
          'http://localhost:5000/api/detect/frame',
          { image: frameData },
          { 
            headers: { 'Content-Type': 'application/json' },
            timeout: 10000
          }
        );
        
        if (response.data.success && isDetectingRef.current) {
          setResultImage(response.data.result_image);
          setDetections(response.data.detections);
          frameCountRef.current++;
        }
      } catch (err) {
        console.error('Detection error:', err);
      }
    }
    
    isProcessingRef.current = false;
  };

  const startDetection = () => {
    setFps(0);
    setIsDetecting(true);
    isDetectingRef.current = true;
    isProcessingRef.current = false;
    frameCountRef.current = 0;
    
    // Use interval instead of while loop for more reliable execution
    detectionIntervalRef.current = setInterval(() => {
      processFrame();
    }, 50); // Process every 50ms for faster detection
  };

  const stopDetection = () => {
    if (detectionIntervalRef.current) {
      clearInterval(detectionIntervalRef.current);
      detectionIntervalRef.current = null;
    }
    setIsDetecting(false);
    isDetectingRef.current = false;
    isProcessingRef.current = false;
  };

  // Image upload handlers
  const handleDrag = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.type === 'dragenter' || e.type === 'dragover') {
      setDragActive(true);
    } else if (e.type === 'dragleave') {
      setDragActive(false);
    }
  }, []);

  const handleDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      handleFile(e.dataTransfer.files[0]);
    }
  }, []);

  const handleFile = (file: File) => {
    if (!file.type.startsWith('image/')) {
      setError('Please upload an image file');
      return;
    }
    
    const reader = new FileReader();
    reader.onload = (e) => {
      setSelectedImage(e.target?.result as string);
      setResultImage(null);
      setDetections([]);
      setError(null);
    };
    reader.readAsDataURL(file);
  };

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      handleFile(e.target.files[0]);
    }
  };

  const detectObjects = async () => {
    if (!selectedImage) return;
    
    setLoading(true);
    setError(null);
    
    try {
      const response = await axios.post<DetectionResult>(
        'http://localhost:5000/api/detect/base64',
        { image: selectedImage },
        { headers: { 'Content-Type': 'application/json' } }
      );
      
      if (response.data.success) {
        setResultImage(response.data.result_image);
        setDetections(response.data.detections);
      } else {
        setError('Detection failed');
      }
    } catch (err: any) {
      setError(err.response?.data?.error || 'Failed to connect to server');
    } finally {
      setLoading(false);
    }
  };

  const resetAll = () => {
    setSelectedImage(null);
    setResultImage(null);
    setDetections([]);
    setError(null);
    if (fileInputRef.current) {
      fileInputRef.current.value = '';
    }
  };

  const switchMode = (newMode: 'upload' | 'camera') => {
    if (newMode === 'upload' && cameraActive) {
      stopCamera();
    }
    resetAll();
    setMode(newMode);
  };

  return (
    <main className="min-h-screen p-4 md:p-8">
      {/* Header */}
      <div className="max-w-6xl mx-auto mb-8">
        <div className="flex items-center justify-center gap-3 mb-2">
          <Zap className="w-10 h-10 text-yellow-400" />
          <h1 className="text-4xl font-bold text-white">
            YOLO Parts Detection
          </h1>
        </div>
        <p className="text-center text-gray-400">
          Construction Machinery Parts Detection System
        </p>
      </div>

      {/* Mode Toggle */}
      <div className="max-w-6xl mx-auto mb-6">
        <div className="flex justify-center gap-4">
          <button
            onClick={() => switchMode('upload')}
            className={`flex items-center gap-2 px-6 py-3 rounded-xl font-semibold transition-all ${
              mode === 'upload'
                ? 'bg-yellow-500 text-black'
                : 'bg-gray-700 text-gray-300 hover:bg-gray-600'
            }`}
          >
            <Upload className="w-5 h-5" />
            Upload Image
          </button>
          <button
            onClick={() => switchMode('camera')}
            className={`flex items-center gap-2 px-6 py-3 rounded-xl font-semibold transition-all ${
              mode === 'camera'
                ? 'bg-yellow-500 text-black'
                : 'bg-gray-700 text-gray-300 hover:bg-gray-600'
            }`}
          >
            <Video className="w-5 h-5" />
            Real-Time Camera
          </button>
        </div>
      </div>

      <div className="max-w-6xl mx-auto grid grid-cols-1 lg:grid-cols-2 gap-8">
        {/* Left Panel */}
        <div className="bg-gray-800/50 rounded-2xl p-6 backdrop-blur">
          {mode === 'upload' ? (
            <>
              <h2 className="text-xl font-semibold mb-4 flex items-center gap-2">
                <Upload className="w-5 h-5" />
                Upload Image
              </h2>
              
              {/* Drop Zone */}
              <div
                className={`drop-zone ${dragActive ? 'active' : ''} ${selectedImage ? 'border-green-500' : ''}`}
                onDragEnter={handleDrag}
                onDragLeave={handleDrag}
                onDragOver={handleDrag}
                onDrop={handleDrop}
                onClick={() => fileInputRef.current?.click()}
              >
                <input
                  ref={fileInputRef}
                  type="file"
                  accept="image/*"
                  onChange={handleFileChange}
                  className="hidden"
                />
                
                {selectedImage ? (
                  <div className="relative">
                    <img
                      src={selectedImage}
                      alt="Selected"
                      className="max-h-64 mx-auto rounded-lg"
                    />
                    <CheckCircle className="absolute top-2 right-2 w-6 h-6 text-green-500" />
                  </div>
                ) : (
                  <div className="py-8">
                    <Camera className="w-16 h-16 mx-auto text-gray-500 mb-4" />
                    <p className="text-gray-400">
                      Drag & drop an image here, or click to select
                    </p>
                    <p className="text-sm text-gray-500 mt-2">
                      Supports: JPG, PNG, WEBP
                    </p>
                  </div>
                )}
              </div>

              {/* Action Buttons */}
              <div className="flex gap-4 mt-6">
                <button
                  onClick={detectObjects}
                  disabled={!selectedImage || loading}
                  className="flex-1 bg-yellow-500 hover:bg-yellow-600 disabled:bg-gray-600 disabled:cursor-not-allowed text-black font-semibold py-3 px-6 rounded-xl transition-all flex items-center justify-center gap-2"
                >
                  {loading ? (
                    <>
                      <Loader2 className="w-5 h-5 animate-spin" />
                      Detecting...
                    </>
                  ) : (
                    <>
                      <Zap className="w-5 h-5" />
                      Detect Parts
                    </>
                  )}
                </button>
                
                <button
                  onClick={resetAll}
                  className="bg-gray-700 hover:bg-gray-600 text-white font-semibold py-3 px-6 rounded-xl transition-all"
                >
                  Reset
                </button>
              </div>
            </>
          ) : (
            <>
              <h2 className="text-xl font-semibold mb-4 flex items-center gap-2">
                <Video className="w-5 h-5" />
                Real-Time Camera
                {isDetecting && (
                  <span className="ml-auto text-sm bg-green-500/20 text-green-400 px-3 py-1 rounded-full">
                    {fps} FPS
                  </span>
                )}
              </h2>
              
              {/* Camera View */}
              <div className="bg-gray-900 rounded-xl p-4 min-h-[300px] flex items-center justify-center relative">
                <video
                  ref={videoRef}
                  className={`max-w-full max-h-[350px] rounded-lg absolute ${cameraActive && !isDetecting ? 'opacity-100' : 'opacity-0 pointer-events-none'}`}
                  autoPlay
                  playsInline
                  muted
                />
                <canvas ref={canvasRef} className="hidden" />
                
                {!cameraActive && (
                  <div className="text-center text-gray-500">
                    <VideoOff className="w-16 h-16 mx-auto mb-4 opacity-50" />
                    <p>Camera is off</p>
                    <p className="text-sm mt-2">Click &quot;Start Camera&quot; to begin</p>
                  </div>
                )}
                
                {cameraActive && isDetecting && resultImage && (
                  <img
                    src={resultImage}
                    alt="Detection Result"
                    className="max-w-full max-h-[350px] rounded-lg"
                  />
                )}
                
                {cameraActive && isDetecting && !resultImage && (
                  <div className="absolute inset-0 flex items-center justify-center bg-gray-900/80 rounded-xl">
                    <Loader2 className="w-10 h-10 animate-spin text-yellow-400" />
                  </div>
                )}
              </div>

              {/* Camera Controls */}
              <div className="flex gap-4 mt-6">
                {!cameraActive ? (
                  <button
                    onClick={startCamera}
                    className="flex-1 bg-green-500 hover:bg-green-600 text-white font-semibold py-3 px-6 rounded-xl transition-all flex items-center justify-center gap-2"
                  >
                    <Video className="w-5 h-5" />
                    Start Camera
                  </button>
                ) : (
                  <>
                    {!isDetecting ? (
                      <button
                        onClick={startDetection}
                        className="flex-1 bg-yellow-500 hover:bg-yellow-600 text-black font-semibold py-3 px-6 rounded-xl transition-all flex items-center justify-center gap-2"
                      >
                        <Zap className="w-5 h-5" />
                        Start Detection
                      </button>
                    ) : (
                      <button
                        onClick={stopDetection}
                        className="flex-1 bg-orange-500 hover:bg-orange-600 text-white font-semibold py-3 px-6 rounded-xl transition-all flex items-center justify-center gap-2"
                      >
                        <VideoOff className="w-5 h-5" />
                        Pause Detection
                      </button>
                    )}
                    <button
                      onClick={stopCamera}
                      className="bg-red-500 hover:bg-red-600 text-white font-semibold py-3 px-6 rounded-xl transition-all"
                    >
                      Stop
                    </button>
                  </>
                )}
              </div>
            </>
          )}

          {/* Error Message */}
          {error && (
            <div className="mt-4 p-4 bg-red-500/20 border border-red-500 rounded-xl flex items-center gap-2">
              <XCircle className="w-5 h-5 text-red-500" />
              <span className="text-red-400">{error}</span>
            </div>
          )}
        </div>

        {/* Right Panel - Results */}
        <div className="bg-gray-800/50 rounded-2xl p-6 backdrop-blur">
          <h2 className="text-xl font-semibold mb-4 flex items-center gap-2">
            <CheckCircle className="w-5 h-5" />
            Detection Results
          </h2>
          
          {/* Result Image */}
          {mode === 'upload' && (
            <div className="bg-gray-900 rounded-xl p-4 min-h-[300px] flex items-center justify-center">
              {resultImage ? (
                <img
                  src={resultImage}
                  alt="Detection Result"
                  className="max-w-full max-h-[400px] rounded-lg"
                />
              ) : (
                <div className="text-center text-gray-500">
                  <Info className="w-16 h-16 mx-auto mb-4 opacity-50" />
                  <p>Detection results will appear here</p>
                </div>
              )}
            </div>
          )}

          {/* Detection Stats */}
          <div className={mode === 'camera' ? '' : 'mt-6'}>
            <h3 className="text-lg font-semibold mb-3 flex items-center justify-between">
              <span>Detected Parts ({detections.length})</span>
              {mode === 'camera' && isDetecting && (
                <span className="text-sm text-gray-400">Live</span>
              )}
            </h3>
            
            {detections.length > 0 ? (
              <div className="space-y-2 max-h-[400px] overflow-y-auto">
                {detections.map((det, idx) => (
                  <div
                    key={idx}
                    className="flex items-center justify-between bg-gray-900 rounded-lg p-3"
                  >
                    <div className="flex items-center gap-3">
                      <div
                        className="w-4 h-4 rounded"
                        style={{ backgroundColor: CLASS_COLORS[det.class_name] }}
                      />
                      <span className="font-medium capitalize">
                        {det.class_name}
                      </span>
                    </div>
                    <div className="flex items-center gap-4">
                      <span className="text-sm text-gray-400">
                        {det.bbox.width}x{det.bbox.height}px
                      </span>
                      <span
                        className="font-bold px-3 py-1 rounded-full text-sm"
                        style={{
                          backgroundColor: `${CLASS_COLORS[det.class_name]}20`,
                          color: CLASS_COLORS[det.class_name],
                        }}
                      >
                        {det.confidence}%
                      </span>
                    </div>
                  </div>
                ))}
              </div>
            ) : (
              <div className="bg-gray-900 rounded-xl p-6 text-center text-gray-500">
                <Info className="w-10 h-10 mx-auto mb-2 opacity-50" />
                <p>{mode === 'camera' ? 'Point camera at parts to detect' : 'No detections yet'}</p>
              </div>
            )}
          </div>
        </div>
      </div>

      {/* Class Legend */}
      <div className="max-w-6xl mx-auto mt-8">
        <div className="bg-gray-800/50 rounded-2xl p-6 backdrop-blur">
          <h3 className="text-lg font-semibold mb-4">Part Classes</h3>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
            {Object.entries(CLASS_COLORS).map(([name, color]) => (
              <div
                key={name}
                className="flex items-center gap-3 bg-gray-900 rounded-lg p-3"
              >
                <div
                  className="w-6 h-6 rounded"
                  style={{ backgroundColor: color }}
                />
                <span className="capitalize font-medium">{name}</span>
              </div>
            ))}
          </div>
        </div>
      </div>

      {/* Footer */}
      <footer className="max-w-6xl mx-auto mt-8 text-center text-gray-500 text-sm">
        <p>Powered by YOLOv8 • Flask Backend • Next.js Frontend • Real-Time Detection</p>
      </footer>
    </main>
  );
}
