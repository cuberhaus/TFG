import { useState, useEffect, useRef } from 'react';
import axios from 'axios';
import { Play, Square, Settings, AlertCircle, CheckCircle, Terminal } from 'lucide-react';

export default function ModelTraining() {
  const [modelArch, setModelArch] = useState('FasterRCNN');
  const [batchSize, setBatchSize] = useState(2);
  const [learningRate, setLearningRate] = useState(0.005);
  const [weightDecay, setWeightDecay] = useState(0.0005);
  const [numEpochs, setNumEpochs] = useState(10);
  const [maxSamples, setMaxSamples] = useState<number | ''>('');
  const [debug, setDebug] = useState(false);
  
  const [status, setStatus] = useState({
    is_training: false,
    current_model: null,
    message: 'Idle'
  });
  const [error, setError] = useState<string | null>(null);
  
  const logRef = useRef<HTMLPreElement>(null);

  useEffect(() => {
    const fetchStatus = async () => {
      try {
        const response = await axios.get('http://localhost:8082/api/train/status');
        setStatus(response.data);
      } catch (err) {
        console.error("Failed to fetch training status:", err);
      }
    };
    
    fetchStatus();
    const interval = setInterval(fetchStatus, status.is_training ? 1000 : 3000);
    return () => clearInterval(interval);
  }, [status.is_training]);

  useEffect(() => {
    if (logRef.current) {
      logRef.current.scrollTop = logRef.current.scrollHeight;
    }
  }, [status.message]);

  const handleStartTraining = async () => {
    setError(null);
    try {
      await axios.post('http://localhost:8082/api/train', {
        model_name: modelArch,
        batch_size: batchSize,
        lr: learningRate,
        weight_decay: weightDecay,
        num_epochs: numEpochs,
        ...(maxSamples ? { max_samples: maxSamples } : {}),
        debug
      });
      const response = await axios.get('http://localhost:8082/api/train/status');
      setStatus(response.data);
    } catch (err: any) {
      console.error(err);
      setError(err.response?.data?.detail || "An error occurred while starting training.");
    }
  };

  const handleCancelTraining = async () => {
    try {
      await axios.post('http://localhost:8082/api/train/cancel');
    } catch (err: any) {
      console.error(err);
      setError(err.response?.data?.detail || "Failed to cancel training.");
    }
  };

  return (
    <div className="max-w-3xl mx-auto flex flex-col gap-6">
      <div>
        <h2 className="text-xl font-semibold mb-2">Train Detection Model</h2>
        <p className="text-gray-400 text-sm">
          Configure hyperparameters and trigger PyTorch training for object detection models (FasterRCNN, RetinaNet, SSD) to identify polyps. 
          The training runs asynchronously in the backend. Check the backend terminal for epoch/batch loss logs.
        </p>
      </div>

      {status.is_training ? (
        <div className="bg-gray-800 border border-gray-700 rounded-lg overflow-hidden">
          <div className="flex items-center gap-3 px-5 py-4 border-b border-gray-700 bg-gray-800">
            <div className="w-5 h-5 border-2 border-blue-500 border-t-transparent rounded-full animate-spin flex-shrink-0"></div>
            <div>
              <h3 className="text-base font-medium text-blue-400">
                Training <span className="text-white">{status.current_model}</span>
              </h3>
              <p className="text-xs text-gray-500 mt-0.5">Live logs — polling every second</p>
            </div>
          </div>
          <div className="relative">
            <div className="absolute top-2 right-2 flex items-center gap-1.5 text-xs text-gray-500 bg-gray-900/80 px-2 py-1 rounded">
              <Terminal className="w-3 h-3" />
              stdout
            </div>
            <pre
              ref={logRef}
              className="p-4 text-xs font-mono text-gray-300 bg-gray-900 overflow-auto max-h-80 leading-relaxed whitespace-pre-wrap"
            >{status.message || 'Waiting for output...'}</pre>
          </div>
          <div className="px-4 py-3 border-t border-gray-700 bg-gray-800 flex items-center justify-between">
            <p className="text-xs text-yellow-500">
              You can navigate away — training continues in the background.
            </p>
            <button
              onClick={handleCancelTraining}
              className="flex items-center gap-1.5 px-3 py-1.5 bg-red-600 hover:bg-red-500 text-white text-sm rounded transition-colors"
            >
              <Square className="w-3.5 h-3.5" />
              Cancel
            </button>
          </div>
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

            <div>
              <label className="block text-sm font-medium text-gray-300 mb-1">Number of Epochs</label>
              <input 
                type="number" 
                min="1"
                value={numEpochs}
                onChange={(e) => setNumEpochs(parseInt(e.target.value))}
                className="w-full bg-gray-800 border border-gray-600 rounded px-3 py-2 text-white outline-none focus:border-blue-500"
              />
            </div>

            <div>
              <label className="block text-sm font-medium text-gray-300 mb-1">
                Max Samples <span className="text-gray-500 font-normal">(optional)</span>
              </label>
              <input 
                type="number" 
                min="2"
                placeholder="All"
                value={maxSamples}
                onChange={(e) => setMaxSamples(e.target.value ? parseInt(e.target.value) : '')}
                className="w-full bg-gray-800 border border-gray-600 rounded px-3 py-2 text-white outline-none focus:border-blue-500 placeholder-gray-500"
              />
              <p className="text-xs text-gray-500 mt-1">Limit dataset size for quick testing</p>
            </div>
          </div>

          <div className="mt-4 flex items-center gap-3">
            <label className="relative inline-flex items-center cursor-pointer">
              <input
                type="checkbox"
                checked={debug}
                onChange={(e) => setDebug(e.target.checked)}
                className="sr-only peer"
              />
              <div className="w-9 h-5 bg-gray-600 peer-focus:outline-none rounded-full peer peer-checked:after:translate-x-full peer-checked:after:border-white after:content-[''] after:absolute after:top-[2px] after:left-[2px] after:bg-white after:rounded-full after:h-4 after:w-4 after:transition-all peer-checked:bg-yellow-600"></div>
            </label>
            <div>
              <span className="text-sm font-medium text-gray-300">Debug mode</span>
              <p className="text-xs text-gray-500">Saves to separate debug directories (saved_models_debug, losses_debug)</p>
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
              <div className={`flex items-start gap-2 p-3 border rounded text-sm whitespace-pre-wrap ${
                status.message.toLowerCase().includes('failed') || status.message.toLowerCase().includes('error')
                  ? 'bg-red-900/30 border-red-800 text-red-300'
                  : 'bg-green-900/30 border-green-800 text-green-300'
              }`}>
                {status.message.toLowerCase().includes('failed') || status.message.toLowerCase().includes('error') ? (
                  <AlertCircle className="w-5 h-5 flex-shrink-0 text-red-500 mt-0.5" />
                ) : (
                  <CheckCircle className="w-5 h-5 flex-shrink-0 text-green-500 mt-0.5" />
                )}
                <div className="font-mono overflow-x-auto">{status.message}</div>
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  );
}