import { useState, useEffect, useRef } from 'react';
import { api } from '../api';
import { Play, Square, X, Settings, AlertCircle, CheckCircle, Terminal, Crosshair } from 'lucide-react';

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
  const [lastResult, setLastResult] = useState<{ type: 'success' | 'error' | 'cancelled'; message: string } | null>(null);
  const wasTraining = useRef(false);
  
  const logRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const fetchStatus = async () => {
      try {
        const response = await api.get('/api/train/status');
        const newStatus = response.data;

        if (wasTraining.current && !newStatus.is_training) {
          const msg = newStatus.message;
          if (msg.toLowerCase().includes('cancelled')) {
            setLastResult({ type: 'cancelled', message: msg });
          } else if (msg.toLowerCase().includes('failed') || msg.toLowerCase().includes('error')) {
            setLastResult({ type: 'error', message: msg });
          } else if (msg.toLowerCase().includes('success') || msg.toLowerCase().includes('completed')) {
            setLastResult({ type: 'success', message: msg });
          }
        }
        wasTraining.current = newStatus.is_training;
        setStatus(newStatus);
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
      await api.post('/api/train', {
        model_name: modelArch,
        batch_size: batchSize,
        lr: learningRate,
        weight_decay: weightDecay,
        num_epochs: numEpochs,
        ...(maxSamples ? { max_samples: maxSamples } : {}),
        debug
      });
      const response = await api.get('/api/train/status');
      setStatus(response.data);
    } catch (err: any) {
      console.error(err);
      setError(err.response?.data?.detail || "An error occurred while starting training.");
    }
  };

  const handleCancelTraining = async () => {
    try {
      await api.post('/api/train/cancel');
    } catch (err: any) {
      console.error(err);
      setError(err.response?.data?.detail || "Failed to cancel training.");
    }
  };

  return (
    <div className="max-w-6xl mx-auto flex flex-col gap-6 pt-2">
      <div>
        <h2 className="text-2xl font-bold mb-2 flex items-center gap-2">
          <Crosshair className="w-6 h-6 text-teal-400" />
          Detection Training
        </h2>
        <p className="text-gray-400">
          Configure hyperparameters and trigger PyTorch training for object detection models (FasterRCNN, RetinaNet, SSD) to identify polyps. 
          The training runs asynchronously in the backend.
        </p>
      </div>

      {lastResult && (
        <div className={`flex items-start gap-3 p-4 rounded-lg border animate-in ${
          lastResult.type === 'success'
            ? 'bg-green-900/30 border-green-700 text-green-200'
            : lastResult.type === 'cancelled'
            ? 'bg-yellow-900/30 border-yellow-700 text-yellow-200'
            : 'bg-red-900/30 border-red-700 text-red-200'
        }`}>
          {lastResult.type === 'success' ? (
            <CheckCircle className="w-5 h-5 flex-shrink-0 mt-0.5 text-green-400" />
          ) : lastResult.type === 'cancelled' ? (
            <Square className="w-5 h-5 flex-shrink-0 mt-0.5 text-yellow-400" />
          ) : (
            <AlertCircle className="w-5 h-5 flex-shrink-0 mt-0.5 text-red-400" />
          )}
          <pre className="text-sm font-mono whitespace-pre-wrap flex-1 overflow-x-auto max-h-40 overflow-y-auto log-scroll">{lastResult.message}</pre>
          <button onClick={() => setLastResult(null)} className="text-gray-400 hover:text-white flex-shrink-0">
            <X className="w-4 h-4" />
          </button>
        </div>
      )}

      {status.is_training ? (
        <div className="bg-teal-900/10 border border-teal-900/50 rounded-xl overflow-hidden shadow-lg flex flex-col h-[500px]">
          <div className="bg-gray-800/80 border-b border-gray-700 p-4 flex items-center justify-between">
            <div className="flex items-center gap-3">
              <div className="w-6 h-6 border-2 border-teal-500 border-t-transparent rounded-full animate-spin"></div>
              <h3 className="font-medium text-teal-400">
                Training <span className="font-bold text-white">{status.current_model}</span>
              </h3>
            </div>
            <span className="text-xs text-gray-500 flex items-center gap-1">
              <Terminal className="w-3 h-3" /> Live Output
            </span>
          </div>

          <div ref={logRef} className="p-4 bg-[#0d1117] flex-1 overflow-y-auto font-mono text-sm text-gray-300 leading-relaxed log-scroll">
            <div className="whitespace-pre-wrap">{status.message || 'Waiting for output...'}</div>
            <span className="animate-pulse inline-block w-2 h-4 bg-teal-500 ml-1 align-middle"></span>
          </div>

          <div className="p-3 bg-gray-800 border-t border-gray-700 flex items-center justify-between">
            <span className="text-xs text-yellow-500">Job running in background. You can safely navigate to other tabs.</span>
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
        <div className="bg-gray-800 p-6 rounded-xl border border-gray-700 shadow-sm">
          <div className="flex items-center gap-2 mb-6 border-b border-gray-700 pb-3">
            <Settings className="text-gray-400 w-5 h-5" />
            <h3 className="text-lg font-medium text-gray-200">Hyperparameters</h3>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            <div>
              <label className="block text-sm font-medium text-gray-300 mb-1">Model Architecture</label>
              <select 
                value={modelArch}
                onChange={(e) => setModelArch(e.target.value)}
                className="w-full bg-gray-900 border border-gray-600 rounded-lg px-3 py-2.5 text-white outline-none focus:border-teal-500 transition-colors"
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
                onChange={(e) => setBatchSize(parseInt(e.target.value) || 1)}
                className="w-full bg-gray-900 border border-gray-600 rounded-lg px-3 py-2.5 text-white outline-none focus:border-teal-500 transition-colors"
              />
            </div>

            <div>
              <label className="block text-sm font-medium text-gray-300 mb-1">Learning Rate</label>
              <input 
                type="number" 
                step="0.001"
                min="0.0001"
                value={learningRate}
                onChange={(e) => setLearningRate(parseFloat(e.target.value) || 0.0001)}
                className="w-full bg-gray-900 border border-gray-600 rounded-lg px-3 py-2.5 text-white outline-none focus:border-teal-500 transition-colors"
              />
            </div>

            <div>
              <label className="block text-sm font-medium text-gray-300 mb-1">Weight Decay</label>
              <input 
                type="number" 
                step="0.0001"
                min="0"
                value={weightDecay}
                onChange={(e) => setWeightDecay(parseFloat(e.target.value) || 0.0001)}
                className="w-full bg-gray-900 border border-gray-600 rounded-lg px-3 py-2.5 text-white outline-none focus:border-teal-500 transition-colors"
              />
            </div>

            <div>
              <label className="block text-sm font-medium text-gray-300 mb-1">Number of Epochs</label>
              <input 
                type="number" 
                min="1"
                value={numEpochs}
                onChange={(e) => setNumEpochs(parseInt(e.target.value) || 1)}
                className="w-full bg-gray-900 border border-gray-600 rounded-lg px-3 py-2.5 text-white outline-none focus:border-teal-500 transition-colors"
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
                className="w-full bg-gray-900 border border-gray-600 rounded-lg px-3 py-2.5 text-white outline-none focus:border-teal-500 placeholder-gray-500 transition-colors"
              />
              <p className="text-xs text-gray-500 mt-1">Limit dataset size for quick testing</p>
            </div>
          </div>

          <div className="mt-5 flex items-center justify-between bg-gray-900/50 rounded-lg p-3 border border-gray-700">
            <div>
              <span className="text-sm font-medium text-gray-300">Debug Mode</span>
              <p className="text-xs text-gray-500 mt-0.5">Saves to separate debug directories (saved_models_debug, losses_debug)</p>
            </div>
            <button
              type="button"
              onClick={() => setDebug(!debug)}
              className={`relative inline-flex h-6 w-11 items-center rounded-full transition-colors ${debug ? 'bg-teal-600' : 'bg-gray-600'}`}
            >
              <span className={`inline-block h-4 w-4 transform rounded-full bg-white transition-transform ${debug ? 'translate-x-6' : 'translate-x-1'}`} />
            </button>
          </div>

          <div className="mt-8 pt-6 border-t border-gray-700">
            <button
              onClick={handleStartTraining}
              className="w-full flex items-center justify-center gap-2 bg-teal-600 hover:bg-teal-500 text-white px-8 py-4 rounded-xl font-medium transition-all hover:scale-[1.02] active:scale-[0.98] shadow-lg shadow-teal-900/20 text-lg"
            >
              <Play className="w-5 h-5 fill-current" />
              Start Training
            </button>
            
            {error && (
              <div className="flex items-center gap-2 p-3 mt-4 bg-red-900/50 border border-red-500 rounded-lg text-red-200 text-sm">
                <AlertCircle className="w-5 h-5 flex-shrink-0" />
                {error}
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  );
}