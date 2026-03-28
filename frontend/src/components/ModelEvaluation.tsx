import { useState, useEffect, useRef } from 'react';
import axios from 'axios';
import { Play, AlertTriangle, CheckCircle2, Square, X, Terminal } from 'lucide-react';

interface EvaluationStatus {
  is_evaluating: boolean;
  message: string;
}

export default function ModelEvaluation() {
  const [status, setStatus] = useState<EvaluationStatus>({
    is_evaluating: false,
    message: 'Idle',
  });
  const [error, setError] = useState('');
  const [lastResult, setLastResult] = useState<{ type: 'success' | 'error' | 'cancelled'; message: string } | null>(null);
  const wasEvaluating = useRef(false);
  const logRef = useRef<HTMLDivElement>(null);

  const fetchStatus = async () => {
    try {
      const response = await axios.get('http://localhost:8082/api/evaluate/status');
      const newStatus = response.data;

      if (wasEvaluating.current && !newStatus.is_evaluating) {
        const msg = newStatus.message;
        if (msg.toLowerCase().includes('cancelled')) {
          setLastResult({ type: 'cancelled', message: msg });
        } else if (msg.toLowerCase().includes('failed') || msg.toLowerCase().includes('error')) {
          setLastResult({ type: 'error', message: msg });
        } else {
          setLastResult({ type: 'success', message: msg });
        }
      }
      wasEvaluating.current = newStatus.is_evaluating;
      setStatus(newStatus);
    } catch (err) {
      console.error("Failed to fetch evaluation status", err);
    }
  };

  useEffect(() => {
    fetchStatus();
    const interval = setInterval(fetchStatus, status.is_evaluating ? 1000 : 3000);
    return () => clearInterval(interval);
  }, [status.is_evaluating]);

  useEffect(() => {
    if (logRef.current) {
      logRef.current.scrollTop = logRef.current.scrollHeight;
    }
  }, [status.message]);

  const handleStartEvaluation = async () => {
    setError('');
    setLastResult(null);
    try {
      await axios.post('http://localhost:8082/api/evaluate');
      fetchStatus();
    } catch (err: any) {
      setError(err.response?.data?.detail || 'Failed to start evaluation.');
    }
  };

  const handleCancel = async () => {
    try {
      await axios.post('http://localhost:8082/api/evaluate/cancel');
    } catch (err: any) {
      setError(err.response?.data?.detail || 'Failed to cancel evaluation.');
    }
  };

  return (
    <div className="max-w-4xl mx-auto flex flex-col gap-6 pt-2">
      <div>
        <h2 className="text-2xl font-bold mb-2 flex items-center gap-2">
          <Terminal className="w-6 h-6 text-blue-400" />
          Model Evaluation
        </h2>
        <p className="text-gray-400">
          Compute COCO performance metrics for all saved models against the test dataset. Results appear in the Performance Explorer tab.
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
            <CheckCircle2 className="w-5 h-5 flex-shrink-0 mt-0.5 text-green-400" />
          ) : lastResult.type === 'cancelled' ? (
            <Square className="w-5 h-5 flex-shrink-0 mt-0.5 text-yellow-400" />
          ) : (
            <AlertTriangle className="w-5 h-5 flex-shrink-0 mt-0.5 text-red-400" />
          )}
          <pre className="text-sm font-mono whitespace-pre-wrap flex-1 overflow-x-auto max-h-40 overflow-y-auto">{lastResult.message}</pre>
          <button onClick={() => setLastResult(null)} className="text-gray-400 hover:text-white flex-shrink-0">
            <X className="w-4 h-4" />
          </button>
        </div>
      )}

      {error && (
        <div className="bg-red-900/50 border border-red-500 text-red-200 px-4 py-3 rounded text-sm">
          {error}
        </div>
      )}

      {status.is_evaluating ? (
        <div className="bg-blue-900/10 border border-blue-900/50 rounded-xl overflow-hidden shadow-lg flex flex-col h-[500px]">
          <div className="bg-gray-800/80 border-b border-gray-700 p-4 flex items-center justify-between">
            <div className="flex items-center gap-3">
              <div className="w-6 h-6 border-2 border-blue-500 border-t-transparent rounded-full animate-spin"></div>
              <h3 className="font-medium text-blue-400">Evaluation in Progress</h3>
            </div>
            <span className="text-xs text-gray-500 flex items-center gap-1">
              <Terminal className="w-3 h-3" /> Live Output
            </span>
          </div>

          <div ref={logRef} className="p-4 bg-[#0d1117] flex-1 overflow-y-auto font-mono text-sm text-gray-300 leading-relaxed">
            <div className="whitespace-pre-wrap">{status.message}</div>
            <span className="animate-pulse inline-block w-2 h-4 bg-blue-500 ml-1 align-middle"></span>
          </div>

          <div className="p-3 bg-gray-800 border-t border-gray-700 flex items-center justify-between">
            <span className="text-xs text-yellow-500">Job running in background. You can safely navigate to other tabs.</span>
            <button
              onClick={handleCancel}
              className="flex items-center gap-1.5 px-3 py-1.5 bg-red-600 hover:bg-red-500 text-white text-sm rounded transition-colors"
            >
              <Square className="w-3.5 h-3.5" />
              Cancel
            </button>
          </div>
        </div>
      ) : (
        <div className="space-y-6">
          <div className="bg-yellow-900/30 border border-yellow-700/50 rounded-lg p-5 flex gap-4 text-yellow-200">
            <AlertTriangle className="w-6 h-6 flex-shrink-0 text-yellow-500" />
            <div className="space-y-1 text-sm">
              <p className="font-semibold text-yellow-400">Important Information</p>
              <p>
                This evaluates <strong>every model</strong> in <code className="bg-yellow-900/50 px-1 py-0.5 rounded text-yellow-300">out/saved_models</code> against the test dataset.
                Results update the <strong>Performance Explorer</strong> tab.
              </p>
            </div>
          </div>

          <div className="flex justify-center">
            <button
              onClick={handleStartEvaluation}
              className="flex items-center gap-2 bg-blue-600 hover:bg-blue-500 text-white px-8 py-4 rounded-xl font-medium transition-all hover:scale-[1.02] active:scale-[0.98] shadow-lg shadow-blue-900/20 text-lg"
            >
              <Play className="w-5 h-5 fill-current" />
              Start Evaluation Job
            </button>
          </div>
        </div>
      )}
    </div>
  );
}
