import { useState, useEffect } from 'react';
import axios from 'axios';
import { UploadCloud, Image as ImageIcon, AlertCircle } from 'lucide-react';

interface ModelInfo {
  filename: str;
}

export default function InferenceUI() {
  const [models, setModels] = useState<string[]>([]);
  const [selectedModel, setSelectedModel] = useState('');
  const [modelArch, setModelArch] = useState('FasterRCNN');
  const [confidence, setConfidence] = useState(0.5);
  const [file, setFile] = useState<File | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [sourcePath, setSourcePath] = useState<string | null>(null);
  
  const [loading, setLoading] = useState(false);
  const [resultImg, setResultImg] = useState<string | null>(null);
  const [boxes, setBoxes] = useState<any[]>([]);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const fetchModels = async () => {
      try {
        const response = await axios.get('http://localhost:8082/api/models');
        const modelNames = response.data.models.map((m: any) => m.filename);
        setModels(modelNames);
        setSourcePath(response.data.source_path);
        if (modelNames.length > 0) {
          setSelectedModel(modelNames[0]);
        }
      } catch (error) {
        console.error('Failed to fetch models', error);
      }
    };
    fetchModels();
  }, []);

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      const selectedFile = e.target.files[0];
      setFile(selectedFile);
      setPreviewUrl(URL.createObjectURL(selectedFile));
      setResultImg(null);
      setBoxes([]);
    }
  };

  const handlePredict = async () => {
    if (!file) return;
    if (!selectedModel) {
      setError("Please select a model weights file.");
      return;
    }

    setLoading(true);
    setError(null);
    setResultImg(null);

    const formData = new FormData();
    formData.append('file', file);
    formData.append('model_arch', modelArch);
    formData.append('model_file', selectedModel);
    formData.append('confidence', confidence.toString());

    try {
      const response = await axios.post('http://localhost:8082/api/predict', formData, {
        headers: {
          'Content-Type': 'multipart/form-data'
        }
      });
      
      setResultImg(`data:image/jpeg;base64,${response.data.image_base64}`);
      setBoxes(response.data.boxes);
    } catch (err: any) {
      console.error(err);
      setError(err.response?.data?.detail || "An error occurred during prediction.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="grid grid-cols-1 md:grid-cols-3 gap-8">
      {/* Settings Column */}
      <div className="md:col-span-1 bg-gray-700 p-6 rounded-lg border border-gray-600 flex flex-col gap-5">
        <h3 className="text-lg font-semibold border-b border-gray-600 pb-2">Configuration</h3>
        
        <div>
          <label className="block text-sm font-medium text-gray-300 mb-1">Model Architecture</label>
          <select 
            value={modelArch}
            onChange={(e) => setModelArch(e.target.value)}
            className="w-full bg-gray-800 border border-gray-600 rounded px-3 py-2 text-white outline-none focus:border-blue-500"
          >
            <option value="FasterRCNN">FasterRCNN</option>
            <option value="RetinaNet">RetinaNet</option>
            <option value="SSD">SSD</option>
          </select>
        </div>

        <div>
          <label className="block text-sm font-medium text-gray-300 mb-1">Model Weights</label>
          {sourcePath && (
            <div className="text-xs text-gray-400 mb-2 truncate" title={sourcePath}>
              Loaded from: <span className="font-mono text-gray-500">{sourcePath}</span>
            </div>
          )}
          {models.length > 0 ? (
            <select 
              value={selectedModel}
              onChange={(e) => setSelectedModel(e.target.value)}
              className="w-full bg-gray-800 border border-gray-600 rounded px-3 py-2 text-white outline-none focus:border-blue-500"
            >
              {models.map(m => (
                <option key={m} value={m}>{m}</option>
              ))}
            </select>
          ) : (
            <div className="text-sm text-yellow-500 flex items-center gap-1">
              <AlertCircle className="w-4 h-4" /> No models found in /out/saved_models
            </div>
          )}
        </div>

        <div>
          <label className="block text-sm font-medium text-gray-300 mb-1">
            Confidence Threshold: {Math.round(confidence * 100)}%
          </label>
          <input 
            type="range" 
            min="0" max="1" step="0.05"
            value={confidence}
            onChange={(e) => setConfidence(parseFloat(e.target.value))}
            className="w-full"
          />
        </div>

        <div>
          <label className="block text-sm font-medium text-gray-300 mb-2">Upload Image</label>
          <label className="flex flex-col items-center justify-center w-full h-32 border-2 border-gray-500 border-dashed rounded-lg cursor-pointer bg-gray-800 hover:bg-gray-750 transition-colors">
            <div className="flex flex-col items-center justify-center pt-5 pb-6">
              <UploadCloud className="w-8 h-8 text-gray-400 mb-2" />
              <p className="text-sm text-gray-400">Click to upload image</p>
            </div>
            <input type="file" className="hidden" accept="image/*" onChange={handleFileChange} />
          </label>
        </div>

        <button
          onClick={handlePredict}
          disabled={!file || loading || models.length === 0}
          className={`w-full py-3 rounded font-medium mt-4 transition-colors ${
            !file || loading || models.length === 0
              ? 'bg-gray-600 text-gray-400 cursor-not-allowed'
              : 'bg-blue-600 hover:bg-blue-500 text-white shadow-lg'
          }`}
        >
          {loading ? 'Running Model...' : 'Run Detection'}
        </button>

        {error && (
          <div className="mt-4 p-3 bg-red-900/50 border border-red-500 rounded text-red-200 text-sm">
            {error}
          </div>
        )}
      </div>

      {/* Image Preview / Result Column */}
      <div className="md:col-span-2 bg-gray-900 rounded-lg border border-gray-700 p-4 flex flex-col items-center justify-center min-h-[400px]">
        {!previewUrl && !resultImg ? (
          <div className="flex flex-col items-center text-gray-500">
            <ImageIcon className="w-16 h-16 mb-4 opacity-50" />
            <p>Upload an image to see results</p>
          </div>
        ) : (
          <div className="w-full flex flex-col items-center">
            {loading ? (
              <div className="animate-pulse flex flex-col items-center">
                <div className="w-12 h-12 border-4 border-blue-500 border-t-transparent rounded-full animate-spin mb-4"></div>
                <p className="text-blue-400">Processing image through PyTorch...</p>
              </div>
            ) : (
              <>
                <img 
                  src={resultImg || previewUrl || ''} 
                  alt="Prediction" 
                  className="max-w-full max-h-[500px] object-contain rounded shadow-lg border border-gray-700"
                />
                
                {resultImg && (
                  <div className="mt-6 w-full max-w-lg">
                    <h4 className="text-md font-semibold mb-2 flex justify-between items-center">
                      Detection Results 
                      <span className="bg-blue-900 text-blue-300 text-xs px-2 py-1 rounded">
                        {boxes.length} found
                      </span>
                    </h4>
                    {boxes.length > 0 ? (
                      <div className="bg-gray-800 rounded border border-gray-700 overflow-hidden">
                        <table className="w-full text-sm text-left">
                          <thead className="bg-gray-700 text-gray-300">
                            <tr>
                              <th className="px-4 py-2">Score</th>
                              <th className="px-4 py-2">Bounding Box (x1, y1, x2, y2)</th>
                            </tr>
                          </thead>
                          <tbody>
                            {boxes.map((b, i) => (
                              <tr key={i} className="border-t border-gray-700">
                                <td className="px-4 py-2 font-medium text-red-400">{(b.score * 100).toFixed(1)}%</td>
                                <td className="px-4 py-2 text-gray-400">
                                  [{b.x_min.toFixed(0)}, {b.y_min.toFixed(0)}, {b.x_max.toFixed(0)}, {b.y_max.toFixed(0)}]
                                </td>
                              </tr>
                            ))}
                          </tbody>
                        </table>
                      </div>
                    ) : (
                      <p className="text-gray-400 text-sm text-center py-4 bg-gray-800 rounded border border-gray-700">
                        No polyps detected above the {(confidence * 100).toFixed(0)}% confidence threshold.
                      </p>
                    )}
                  </div>
                )}
              </>
            )}
          </div>
        )}
      </div>
    </div>
  );
}
