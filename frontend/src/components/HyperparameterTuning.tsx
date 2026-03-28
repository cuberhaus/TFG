import { useState, useEffect, useRef } from 'react';
import axios from 'axios';
import { Sliders, Settings, Play, Square, X, AlertCircle, CheckCircle, Terminal } from 'lucide-react';

interface HPOStatus {
  is_tuning: boolean;
  current_model: string | null;
  message: string;
}

export default function HyperparameterTuning() {
  const [modelArch, setModelArch] = useState('FasterRCNN');
  const [numTrials, setNumTrials] = useState(10);
  const [maxEpochs, setMaxEpochs] = useState(5);
  const [maxSamples, setMaxSamples] = useState<number | ''>('');
  const [debug, setDebug] = useState(false);
  
  const [status, setStatus] = useState<HPOStatus>({
    is_tuning: false,
    current_model: null,
    message: 'Idle'
  });
  const [error, setError] = useState<string | null>(null);
  const [lastResult, setLastResult] = useState<{ type: 'success' | 'error' | 'cancelled'; message: string } | null>(null);
  const wasTuning = useRef(false);
  
  const fetchStatus = async () => {
    try {
      const response = await axios.get('http://localhost:8082/api/hpo/status');
      const newStatus = response.data;

      if (wasTuning.current && !newStatus.is_tuning) {
        const msg = newStatus.message;
        if (msg.toLowerCase().includes('cancelled')) {
          setLastResult({ type: 'cancelled', message: msg });
        } else if (msg.toLowerCase().includes('failed') || msg.toLowerCase().includes('error')) {
          setLastResult({ type: 'error', message: msg });
        } else {
          setLastResult({ type: 'success', message: msg });
        }
      }
      wasTuning.current = newStatus.is_tuning;
      setStatus(newStatus);
    } catch (err) {
      console.error("Failed to fetch tuning status:", err);
    }
  };

  useEffect(() => {
    fetchStatus();
    const interval = setInterval(fetchStatus, status.is_tuning ? 1000 : 3000);
    return () => clearInterval(interval);
  }, [status.is_tuning]);

  const handleStartTuning = async () => {
    setError(null);
    try {
      await axios.post('http://localhost:8082/api/hpo/start', {
        model_name: modelArch,
        num_trials: numTrials,
        max_epochs: maxEpochs,
        ...(maxSamples ? { max_samples: maxSamples } : {}),
        debug,
      });
      fetchStatus();
    } catch (err: any) {
      console.error(err);
      setError(err.response?.data?.detail || "An error occurred while starting the tuning job.");
    }
  };

  const handleCancelTuning = async () => {
    try {
      await axios.post('http://localhost:8082/api/hpo/cancel');
    } catch (err: any) {
      console.error(err);
      setError(err.response?.data?.detail || "Failed to cancel tuning.");
    }
  };

  return (
    <div className="max-w-4xl mx-auto flex flex-col gap-6 pt-2">
      <div>
        <h2 className="text-2xl font-bold mb-2 flex items-center gap-2">
          <Sliders className="w-6 h-6 text-emerald-400" />
          Hyperparameter Tuning
        </h2>
        <p className="text-gray-400">
          Run automated Optuna trials to find the optimal Learning Rate, Batch Size, and Weight Decay for your object detection models. 
          This process is computationally intensive and runs asynchronously.
        </p>
      </div>

      {lastResult && (
        <div className={`flex items-start gap-3 p-4 rounded-lg border ${
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
          <pre className="text-sm font-mono whitespace-pre-wrap flex-1 overflow-x-auto max-h-40 overflow-y-auto">{lastResult.message}</pre>
          <button onClick={() => setLastResult(null)} className="text-gray-400 hover:text-white flex-shrink-0">
            <X className="w-4 h-4" />
          </button>
        </div>
      )}

      {status.is_tuning ? (
        <div className="bg-emerald-900/10 border border-emerald-900/50 rounded-xl overflow-hidden shadow-lg flex flex-col h-[500px]">
          <div className="bg-gray-800/80 border-b border-gray-700 p-4 flex items-center justify-between">
             <div className="flex items-center gap-3">
                <div className="w-6 h-6 border-2 border-emerald-500 border-t-transparent rounded-full animate-spin"></div>
                <h3 className="font-medium text-emerald-400">
                  Tuning <span className="font-bold text-white">{status.current_model}</span>
                </h3>
             </div>
             <span className="text-xs text-gray-500 flex items-center gap-1">
               <Terminal className="w-3 h-3" /> Live Output
             </span>
          </div>
          
          <div className="p-4 bg-[#0d1117] flex-1 overflow-y-auto font-mono text-sm text-gray-300 leading-relaxed">
            <div className="whitespace-pre-wrap">
              {status.message}
            </div>
            {/* Blinking cursor effect */}
            <span className="animate-pulse inline-block w-2 h-4 bg-emerald-500 ml-1 align-middle"></span>
          </div>
          
          <div className="p-3 bg-gray-800 border-t border-gray-700 flex items-center justify-between">
             <span className="text-xs text-yellow-500">Job running in background. You can safely navigate to other tabs.</span>
             <button
               onClick={handleCancelTuning}
               className="flex items-center gap-1.5 px-3 py-1.5 bg-red-600 hover:bg-red-500 text-white text-sm rounded transition-colors"
             >
               <Square className="w-3.5 h-3.5" />
               Cancel
             </button>
          </div>
        </div>
      ) : (
        <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
          <div className="md:col-span-2 space-y-6">
            <div className="bg-gray-800 p-6 rounded-xl border border-gray-700 shadow-sm">
              <div className="flex items-center gap-2 mb-6 border-b border-gray-700 pb-3">
                <Settings className="text-gray-400 w-5 h-5" />
                <h3 className="text-lg font-medium text-gray-200">Search Configuration</h3>
              </div>

              <div className="space-y-5">
                <div>
                  <label className="block text-sm font-medium text-gray-300 mb-1">Model Architecture</label>
                  <select 
                    value={modelArch}
                    onChange={(e) => setModelArch(e.target.value)}
                    className="w-full bg-gray-900 border border-gray-600 rounded-lg px-4 py-3 text-white outline-none focus:border-emerald-500 transition-colors"
                  >
                    <option value="FasterRCNN">FasterRCNN</option>
                    <option value="RetinaNet">RetinaNet</option>
                    <option value="SSD">SSD</option>
                  </select>
                </div>

                <div>
                  <label className="block text-sm font-medium text-gray-300 mb-1 flex justify-between">
                    <span>Number of Trials</span>
                    <span className="text-gray-500">{numTrials} trials</span>
                  </label>
                  <input 
                    type="range" 
                    min="1"
                    max="50"
                    step="1"
                    value={numTrials}
                    onChange={(e) => setNumTrials(parseInt(e.target.value))}
                    className="w-full accent-emerald-500 h-2 bg-gray-700 rounded-lg appearance-none cursor-pointer"
                  />
                  <div className="flex justify-between text-xs text-gray-500 mt-2">
                    <span>Quick Test (1)</span>
                    <span>Deep Search (50)</span>
                  </div>
                </div>

                <div className="grid grid-cols-2 gap-4">
                  <div>
                    <label className="block text-sm font-medium text-gray-300 mb-1">Max Epochs per Trial</label>
                    <input
                      type="number"
                      min="1"
                      max="100"
                      value={maxEpochs}
                      onChange={(e) => setMaxEpochs(parseInt(e.target.value))}
                      className="w-full bg-gray-900 border border-gray-600 rounded-lg px-4 py-3 text-white outline-none focus:border-emerald-500 transition-colors"
                    />
                    <p className="text-xs text-gray-500 mt-1">Optuna picks 1–{maxEpochs} per trial</p>
                  </div>
                  <div>
                    <label className="block text-sm font-medium text-gray-300 mb-1">
                      Max Samples <span className="text-gray-500">(optional)</span>
                    </label>
                    <input
                      type="number"
                      min="1"
                      placeholder="All"
                      value={maxSamples}
                      onChange={(e) => setMaxSamples(e.target.value ? parseInt(e.target.value) : '')}
                      className="w-full bg-gray-900 border border-gray-600 rounded-lg px-4 py-3 text-white outline-none focus:border-emerald-500 transition-colors placeholder-gray-500"
                    />
                    <p className="text-xs text-gray-500 mt-1">Limit images per trial for quick testing</p>
                  </div>
                </div>

                <div className="flex items-center justify-between bg-gray-900/50 rounded-lg p-3 border border-gray-700">
                  <div>
                    <span className="text-sm font-medium text-gray-300">Debug Mode</span>
                    <p className="text-xs text-gray-500 mt-0.5">Use only 20 train + 10 test samples</p>
                  </div>
                  <button
                    type="button"
                    onClick={() => setDebug(!debug)}
                    className={`relative inline-flex h-6 w-11 items-center rounded-full transition-colors ${debug ? 'bg-emerald-600' : 'bg-gray-600'}`}
                  >
                    <span className={`inline-block h-4 w-4 transform rounded-full bg-white transition-transform ${debug ? 'translate-x-6' : 'translate-x-1'}`} />
                  </button>
                </div>
              </div>

              <div className="mt-8 pt-6 border-t border-gray-700">
                <button
                  onClick={handleStartTuning}
                  className="w-full flex items-center justify-center gap-2 bg-emerald-600 hover:bg-emerald-500 text-white px-8 py-4 rounded-xl font-medium transition-all hover:scale-[1.02] active:scale-[0.98] shadow-lg shadow-emerald-900/20 text-lg"
                >
                  <Play className="w-5 h-5 fill-current" />
                  Start HPO Job
                </button>
              </div>
            </div>
          </div>
          
          <div className="md:col-span-1 space-y-4">
             <div className="bg-gray-800 border border-gray-700 rounded-xl p-5 shadow-lg">
                <h4 className="font-medium text-gray-200 mb-3 pb-2 border-b border-gray-700">Last Job Status</h4>
                
                {error ? (
                  <div className="flex items-start gap-2 text-sm p-3 rounded bg-red-900/30 border border-red-800 text-red-300">
                    <AlertCircle className="w-5 h-5 flex-shrink-0 mt-0.5" />
                    <span className="font-mono overflow-x-auto whitespace-pre-wrap">{error}</span>
                  </div>
                ) : status.message !== 'Idle' ? (
                  <div className={`flex items-start gap-2 text-sm p-3 rounded border w-full whitespace-pre-wrap ${
                    status.message.toLowerCase().includes('failed') || status.message.toLowerCase().includes('error')
                      ? 'bg-red-900/30 border-red-800 text-red-300'
                      : 'bg-emerald-900/30 border-emerald-800 text-emerald-300'
                  }`}>
                    {status.message.toLowerCase().includes('failed') || status.message.toLowerCase().includes('error') ? (
                      <AlertCircle className="w-5 h-5 flex-shrink-0 mt-0.5" />
                    ) : (
                      <CheckCircle className="w-5 h-5 flex-shrink-0 mt-0.5" />
                    )}
                    <span className="font-mono overflow-x-auto max-h-[300px] block">{status.message}</span>
                  </div>
                ) : (
                  <div className="text-gray-500 text-sm text-center py-6 italic">
                    No recent activity.
                  </div>
                )}
             </div>
             
             <div className="bg-emerald-900/10 border border-emerald-900/50 rounded-xl p-5 text-sm text-emerald-200/80 shadow-lg">
                 <h4 className="font-medium text-emerald-400 mb-2 flex items-center gap-1">
                   <Sliders className="w-4 h-4" /> How it Works
                 </h4>
                 <p>
                   We use Optuna to search the hyperparameter space. It trains a small version of the model on a subset of data over multiple "trials", iteratively narrowing down the best parameters. Results are saved to <code className="bg-emerald-900/50 px-1 py-0.5 rounded text-emerald-300">best_hyperparameters.csv</code>.
                 </p>
             </div>
          </div>
        </div>
      )}
    </div>
  );
}