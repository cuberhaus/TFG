import { useState, useEffect, useRef } from 'react';
import { api } from '../api';
import { Sliders, Settings, Play, Square, X, AlertCircle, CheckCircle, Terminal, Lock, Trophy, BarChart3 } from 'lucide-react';

interface HPOStatus {
  is_tuning: boolean;
  current_model: string | null;
  message: string;
}

interface Trial {
  number: number;
  value: number | null;
  params: Record<string, number | string>;
  state: string;
}

interface HPOResults {
  found: boolean;
  model_name?: string;
  n_trials?: number;
  best_params?: Record<string, number | string>;
  best_value?: number;
  trials?: Trial[];
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
  const [hpoResults, setHpoResults] = useState<HPOResults | null>(null);
  const wasTuning = useRef(false);

  const fetchResults = async () => {
    try {
      const res = await api.get('/api/hpo/results');
      if (res.data.found) setHpoResults(res.data);
    } catch { /* ignore */ }
  };
  
  const fetchStatus = async () => {
    try {
      const response = await api.get('/api/hpo/status');
      const newStatus = response.data;

      if (wasTuning.current && !newStatus.is_tuning) {
        const msg = newStatus.message;
        if (msg.toLowerCase().includes('cancelled')) {
          setLastResult({ type: 'cancelled', message: msg });
        } else if (msg.toLowerCase().includes('failed') || msg.toLowerCase().includes('error')) {
          setLastResult({ type: 'error', message: msg });
        } else {
          setLastResult({ type: 'success', message: msg });
          fetchResults();
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
    fetchResults();
    const interval = setInterval(fetchStatus, status.is_tuning ? 1000 : 3000);
    return () => clearInterval(interval);
  }, [status.is_tuning]);

  const handleStartTuning = async () => {
    setError(null);
    try {
      await api.post('/api/hpo/start', {
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
      await api.post('/api/hpo/cancel');
    } catch (err: any) {
      console.error(err);
      setError(err.response?.data?.detail || "Failed to cancel tuning.");
    }
  };

  return (
    <div className="max-w-7xl mx-auto flex flex-col gap-6 pt-2">
      <div>
        <h2 className="text-2xl font-bold mb-2 flex items-center gap-2">
          <Sliders className="w-6 h-6 text-fuchsia-400" />
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
          <pre className="text-sm font-mono whitespace-pre-wrap flex-1 overflow-x-auto max-h-40 overflow-y-auto log-scroll">{lastResult.message}</pre>
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
          
          <div className="p-4 bg-[#0d1117] flex-1 overflow-y-auto font-mono text-sm text-gray-300 leading-relaxed log-scroll">
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
                    onChange={(e) => setNumTrials(parseInt(e.target.value) || 1)}
                    className="w-full accent-emerald-500 h-2 bg-gray-700 rounded-lg appearance-none cursor-pointer"
                  />
                  <div className="flex justify-between text-xs text-gray-500 mt-2">
                    <span>Quick Test (1)</span>
                    <span>Deep Search (50)</span>
                  </div>
                </div>

                <div className="flex items-center justify-between bg-gray-900/50 rounded-lg p-3 border border-gray-700">
                  <div>
                    <span className="text-sm font-medium text-gray-300">Debug Mode</span>
                    <p className="text-xs text-gray-500 mt-0.5">8 train + 4 test samples, 1 epoch, batch [2,4]</p>
                  </div>
                  <button
                    type="button"
                    onClick={() => setDebug(!debug)}
                    className={`relative inline-flex h-6 w-11 items-center rounded-full transition-colors ${debug ? 'bg-emerald-600' : 'bg-gray-600'}`}
                  >
                    <span className={`inline-block h-4 w-4 transform rounded-full bg-white transition-transform ${debug ? 'translate-x-6' : 'translate-x-1'}`} />
                  </button>
                </div>

                <div className="grid grid-cols-2 gap-4">
                  <div className={`relative ${debug ? 'opacity-50' : ''}`}>
                    <label className="block text-sm font-medium text-gray-300 mb-1 flex items-center gap-1.5">
                      Max Epochs per Trial
                      {debug && <Lock className="w-3 h-3 text-yellow-500" />}
                    </label>
                    <input
                      type="number"
                      min="1"
                      max="100"
                      value={debug ? 1 : maxEpochs}
                      onChange={(e) => setMaxEpochs(parseInt(e.target.value) || 1)}
                      disabled={debug}
                      className={`w-full bg-gray-900 border rounded-lg px-4 py-3 text-white outline-none transition-colors ${
                        debug ? 'border-yellow-700/50 cursor-not-allowed' : 'border-gray-600 focus:border-emerald-500'
                      }`}
                    />
                    <p className="text-xs mt-1">
                      {debug
                        ? <span className="text-yellow-500 flex items-center gap-1"><Lock className="w-3 h-3" /> Fixed to 1 in debug</span>
                        : <span className="text-gray-500">Optuna picks 1–{maxEpochs} per trial</span>
                      }
                    </p>
                  </div>
                  <div className={`relative ${debug ? 'opacity-50' : ''}`}>
                    <label className="block text-sm font-medium text-gray-300 mb-1 flex items-center gap-1.5">
                      Max Samples
                      {debug ? <Lock className="w-3 h-3 text-yellow-500" /> : <span className="text-gray-500">(optional)</span>}
                    </label>
                    <input
                      type="number"
                      min="1"
                      placeholder="All"
                      value={debug ? 8 : maxSamples}
                      onChange={(e) => setMaxSamples(e.target.value ? parseInt(e.target.value) : '')}
                      disabled={debug}
                      className={`w-full bg-gray-900 border rounded-lg px-4 py-3 text-white outline-none transition-colors placeholder-gray-500 ${
                        debug ? 'border-yellow-700/50 cursor-not-allowed' : 'border-gray-600 focus:border-emerald-500'
                      }`}
                    />
                    <p className="text-xs mt-1">
                      {debug
                        ? <span className="text-yellow-500 flex items-center gap-1"><Lock className="w-3 h-3" /> Fixed to 8 in debug</span>
                        : <span className="text-gray-500">Limit images per trial for quick testing</span>
                      }
                    </p>
                  </div>
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
                   We use Optuna to search the hyperparameter space. It trains a small version of the model on a subset of data over multiple "trials", iteratively narrowing down the best parameters.
                 </p>
             </div>
          </div>
        </div>
      )}

      {hpoResults && !status.is_tuning && (
        <div className="bg-gray-800 border border-gray-700 rounded-xl shadow-lg overflow-hidden">
          <div className="p-5 border-b border-gray-700 bg-gradient-to-r from-emerald-900/20 to-gray-800 flex items-center justify-between">
            <h3 className="text-lg font-medium text-emerald-400 flex items-center gap-2">
              <Trophy className="w-5 h-5 text-yellow-400" />
              Best Hyperparameters
              <span className="text-sm font-normal text-gray-400">— {hpoResults.model_name}, {hpoResults.n_trials} trials</span>
            </h3>
            <span className="text-sm font-mono bg-emerald-900/40 text-emerald-300 px-3 py-1 rounded-full border border-emerald-700/50">
              F1 = {hpoResults.best_value?.toFixed(4)}
            </span>
          </div>

          <div className="p-5">
            <div className="grid grid-cols-2 md:grid-cols-4 gap-3 mb-6">
              {hpoResults.best_params && Object.entries(hpoResults.best_params).map(([key, value]) => (
                <div key={key} className="bg-gray-900 border border-gray-700 rounded-lg p-3 text-center">
                  <p className="text-xs text-gray-500 uppercase tracking-wider mb-1">{key.replace('_', ' ')}</p>
                  <p className="text-lg font-mono font-semibold text-white">
                    {typeof value === 'number' ? (value < 0.01 ? value.toExponential(2) : Number(value.toFixed(6))) : value}
                  </p>
                </div>
              ))}
            </div>

            {hpoResults.trials && hpoResults.trials.length > 0 && (
              <div>
                <h4 className="text-sm font-medium text-gray-400 mb-3 flex items-center gap-1.5">
                  <BarChart3 className="w-4 h-4" /> Trial History
                </h4>
                <div className="overflow-x-auto log-scroll">
                  <table className="w-full text-sm">
                    <thead>
                      <tr className="border-b border-gray-700 text-gray-400">
                        <th className="text-left py-2 px-3 font-medium">#</th>
                        <th className="text-left py-2 px-3 font-medium">F1 Score</th>
                        {hpoResults.trials[0]?.params && Object.keys(hpoResults.trials[0].params).map(k => (
                          <th key={k} className="text-left py-2 px-3 font-medium">{k.replace('_', ' ')}</th>
                        ))}
                        <th className="text-left py-2 px-3 font-medium">Status</th>
                      </tr>
                    </thead>
                    <tbody>
                      {hpoResults.trials.map((trial) => {
                        const isBest = trial.value === hpoResults.best_value && trial.state === 'COMPLETE';
                        return (
                          <tr key={trial.number} className={`border-b border-gray-700/50 ${isBest ? 'bg-emerald-900/20' : 'hover:bg-gray-700/30'}`}>
                            <td className="py-2 px-3 font-mono text-gray-300">
                              {trial.number + 1}
                              {isBest && <Trophy className="w-3 h-3 text-yellow-400 inline ml-1.5" />}
                            </td>
                            <td className={`py-2 px-3 font-mono ${isBest ? 'text-emerald-400 font-semibold' : 'text-gray-300'}`}>
                              {trial.value !== null ? trial.value.toFixed(4) : '—'}
                            </td>
                            {Object.values(trial.params).map((v, i) => (
                              <td key={i} className="py-2 px-3 font-mono text-gray-400">
                                {typeof v === 'number' ? (v < 0.01 ? v.toExponential(2) : Number(v.toFixed(6))) : v}
                              </td>
                            ))}
                            <td className="py-2 px-3">
                              <span className={`text-xs px-2 py-0.5 rounded-full ${
                                trial.state === 'COMPLETE' ? 'bg-emerald-900/40 text-emerald-400' :
                                trial.state === 'PRUNED' ? 'bg-yellow-900/40 text-yellow-400' :
                                'bg-red-900/40 text-red-400'
                              }`}>
                                {trial.state}
                              </span>
                            </td>
                          </tr>
                        );
                      })}
                    </tbody>
                  </table>
                </div>
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  );
}