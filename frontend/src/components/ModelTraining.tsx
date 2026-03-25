import { useState, useEffect } from 'react';
import axios from 'axios';
import { Play, Settings, AlertCircle, CheckCircle } from 'lucide-react';

export default function ModelTraining() {
  const [modelArch, setModelArch] = useState('FasterRCNN');
  const [batchSize, setBatchSize] = useState(2);
  const [learningRate, setLearningRate] = useState(0.005);
  const [weightDecay, setWeightDecay] = useState(0.0005);
  const [numEpochs, setNumEpochs] = useState(10);
  
  const [status, setStatus] = useState({
    is_training: false,
    current_model: null,
    message: 'Idle'
  });
  const [error, setError] = useState<string | null>(null);
  
  useEffect(() => {
    // Poll status every 3 seconds
    const fetchStatus = async () => {
      try {
        const response = await axios.get('http://localhost:8082/api/train/status');
        setStatus(response.data);
      } catch (err) {
        console.error("Failed to fetch training status:", err);
      }
    };
    
    fetchStatus();
    const interval = setInterval(fetchStatus, 3000);
    return () => clearInterval(interval);
  }, []);

  const handleStartTraining = async () => {
    setError(null);
    try {
      await axios.post('http://localhost:8082/api/train', {
        model_name: modelArch,
        batch_size: batchSize,
        lr: learningRate,
        weight_decay: weightDecay,
        num_epochs: numEpochs
      });
      // Immediately refresh status to show training UI
      const response = await axios.get('http://localhost:8082/api/train/status');
      setStatus(response.data);
    } catch (err: any) {
      console.error(err);
      setError(err.response?.data?.detail || "An error occurred while starting training.");
    }
  };

  return (
    <div className="max-w-3xl mx-auto flex flex-col gap-6">
      <div>
        <h2 className="text-xl font-semibold mb-2">Train Model</h2>
        <p className="text-gray-400 text-sm">
          Configure hyperparameters and trigger PyTorch training. 
          The training runs asynchronously in the backend. Check the backend terminal for epoch/batch loss logs.
        </p>
      </div>

      {status.is_training ? (
        <div className="bg-blue-900/30 border border-blue-800 rounded-lg p-8 flex flex-col items-center justify-center text-center">
          <div className="w-16 h-16 border-4 border-blue-500 border-t-transparent rounded-full animate-spin mb-4"></div>
          <h3 className="text-xl font-medium text-blue-400 mb-2">Training in Progress</h3>
          <p className="text-gray-300">
            Currently training <span className="font-bold text-white">{status.current_model}</span>
          </p>
          <p className="text-sm text-gray-400 mt-4">
            {status.message}
          </p>
          <p className="text-xs text-yellow-500 mt-6 bg-yellow-900/20 px-4 py-2 rounded border border-yellow-700/50">
            You can navigate away from this tab. The training will continue in the background.
          </p>
        </div>
      ) : (
        <div className="bg-gray-700 p-6 rounded-lg border border-gray-600 shadow-sm">
          <div className="flex items-center gap-2 mb-6 border-b border-gray-600 pb-3">
            <Settings className="text-gray-400 w-5 h-5" />
            <h3 className="text-lg font-medium text-gray-200">Hyperparameters</h3>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
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
              <label className="block text-sm font-medium text-gray-300 mb-1">Batch Size</label>
              <input 
                type="number" 
                min="1"
                value={batchSize}
                onChange={(e) => setBatchSize(parseInt(e.target.value))}
                className="w-full bg-gray-800 border border-gray-600 rounded px-3 py-2 text-white outline-none focus:border-blue-500"
              />
            </div>

            <div>
              <label className="block text-sm font-medium text-gray-300 mb-1">Learning Rate</label>
              <input 
                type="number" 
                step="0.001"
                min="0.0001"
                value={learningRate}
                onChange={(e) => setLearningRate(parseFloat(e.target.value))}
                className="w-full bg-gray-800 border border-gray-600 rounded px-3 py-2 text-white outline-none focus:border-blue-500"
              />
            </div>

            <div>
              <label className="block text-sm font-medium text-gray-300 mb-1">Weight Decay</label>
              <input 
                type="number" 
                step="0.0001"
                min="0"
                value={weightDecay}
                onChange={(e) => setWeightDecay(parseFloat(e.target.value))}
                className="w-full bg-gray-800 border border-gray-600 rounded px-3 py-2 text-white outline-none focus:border-blue-500"
              />
            </div>

            <div className="md:col-span-2">
              <label className="block text-sm font-medium text-gray-300 mb-1">Number of Epochs</label>
              <input 
                type="number" 
                min="1"
                value={numEpochs}
                onChange={(e) => setNumEpochs(parseInt(e.target.value))}
                className="w-full bg-gray-800 border border-gray-600 rounded px-3 py-2 text-white outline-none focus:border-blue-500"
              />
            </div>
          </div>

          <div className="mt-8 pt-6 border-t border-gray-600 flex flex-col gap-4">
            <button
              onClick={handleStartTraining}
              className="flex items-center justify-center gap-2 w-full py-3 bg-blue-600 hover:bg-blue-500 text-white rounded font-medium shadow-lg transition-colors"
            >
              <Play className="w-5 h-5" />
              Start Training
            </button>
            
            {error && (
              <div className="flex items-center gap-2 p-3 bg-red-900/50 border border-red-500 rounded text-red-200 text-sm">
                <AlertCircle className="w-5 h-5 flex-shrink-0" />
                {error}
              </div>
            )}

            {!error && status.message !== 'Idle' && (
              <div className="flex items-center gap-2 p-3 bg-green-900/30 border border-green-800 rounded text-green-300 text-sm">
                <CheckCircle className="w-5 h-5 flex-shrink-0 text-green-500" />
                {status.message}
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  );
}