import { useState, useEffect } from 'react';
import axios from 'axios';
import { Play, AlertTriangle, FileText, CheckCircle2 } from 'lucide-react';

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

  const fetchStatus = async () => {
    try {
      const response = await axios.get('http://localhost:8082/api/evaluate/status');
      setStatus(response.data);
    } catch (err) {
      console.error("Failed to fetch evaluation status", err);
    }
  };

  useEffect(() => {
    fetchStatus();
    const interval = setInterval(() => {
      fetchStatus();
    }, 3000); // Poll every 3 seconds

    return () => clearInterval(interval);
  }, []);

  const handleStartEvaluation = async () => {
    setError('');
    try {
      await axios.post('http://localhost:8082/api/evaluate');
      fetchStatus();
    } catch (err: any) {
      setError(err.response?.data?.detail || 'Failed to start evaluation.');
    }
  };

  return (
    <div className="max-w-3xl mx-auto space-y-6 pt-4">
      <div className="text-center space-y-2">
        <h2 className="text-2xl font-bold">Run Model Evaluation</h2>
        <p className="text-gray-400">
          Compute performance metrics for all saved models against the test dataset.
        </p>
      </div>

      <div className="bg-yellow-900/30 border border-yellow-700/50 rounded-lg p-5 flex gap-4 text-yellow-200">
        <AlertTriangle className="w-6 h-6 flex-shrink-0 text-yellow-500" />
        <div className="space-y-1 text-sm">
          <p className="font-semibold text-yellow-400">Important Information</p>
          <p>
            Running this process will evaluate <strong>every model</strong> currently present in the <code className="bg-yellow-900/50 px-1 py-0.5 rounded text-yellow-300">out/saved_models</code> directory. 
          </p>
          <p>
            The results will overwrite or append to the <code className="bg-yellow-900/50 px-1 py-0.5 rounded text-yellow-300">model_performances.csv</code> file, which updates the data shown in the <strong>Performance Explorer</strong> tab. This process may take several minutes depending on the number of models and the size of the test dataset.
          </p>
        </div>
      </div>

      {error && (
        <div className="bg-red-900/50 border border-red-500 text-red-200 px-4 py-3 rounded text-sm">
          {error}
        </div>
      )}

      <div className="bg-gray-800 border border-gray-700 rounded-lg p-6 flex flex-col items-center justify-center min-h-[250px] space-y-6">
        {status.is_evaluating ? (
          <div className="flex flex-col items-center space-y-4">
            <div className="relative">
              <div className="w-16 h-16 border-4 border-blue-500/20 border-t-blue-500 rounded-full animate-spin"></div>
              <div className="absolute inset-0 flex items-center justify-center">
                <FileText className="w-6 h-6 text-blue-400 animate-pulse" />
              </div>
            </div>
            <div className="text-center space-y-1">
              <h3 className="font-medium text-lg text-blue-400">Evaluation in Progress</h3>
              <p className="text-sm text-gray-400 max-w-md">{status.message}</p>
            </div>
          </div>
        ) : (
          <div className="flex flex-col items-center space-y-6 w-full">
            <button
              onClick={handleStartEvaluation}
              className="flex items-center gap-2 bg-blue-600 hover:bg-blue-700 text-white px-8 py-4 rounded-lg font-medium transition-all hover:scale-105 active:scale-95 shadow-lg shadow-blue-900/20"
            >
              <Play className="w-5 h-5 fill-current" />
              Start Evaluation Job
            </button>
            
            {status.message !== 'Idle' && (
              <div className={`flex items-start gap-2 text-sm p-4 rounded-lg border w-full whitespace-pre-wrap ${
                status.message.toLowerCase().includes('failed') || status.message.toLowerCase().includes('error')
                  ? 'bg-red-900/30 border-red-800 text-red-300'
                  : 'bg-green-900/30 border-green-800 text-green-300'
              }`}>
                {status.message.toLowerCase().includes('failed') || status.message.toLowerCase().includes('error') ? (
                  <AlertTriangle className="w-5 h-5 text-red-500 flex-shrink-0 mt-0.5" />
                ) : (
                  <CheckCircle2 className="w-5 h-5 text-green-500 flex-shrink-0 mt-0.5" />
                )}
                <span className="font-mono overflow-x-auto">{status.message}</span>
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
}
