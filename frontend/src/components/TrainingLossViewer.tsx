import { useState, useEffect } from 'react';
import { api } from '../api';
import { 
  LineChart, 
  Line, 
  XAxis, 
  YAxis, 
  CartesianGrid, 
  Tooltip, 
  Legend, 
  ResponsiveContainer 
} from 'recharts';
import { TrendingUp } from 'lucide-react';

interface LossDataResponse {
  data: {
    filename: string;
    short_name: string;
    values: number[];
  }[];
}

const COLORS = [
  '#3b82f6', // blue
  '#10b981', // green
  '#f59e0b', // yellow
  '#ef4444', // red
  '#8b5cf6', // purple
  '#ec4899', // pink
  '#06b6d4', // rose
];

export default function TrainingLossViewer() {
  const [sourcePath, setSourcePath] = useState<string>('');
  const [allFiles, setAllFiles] = useState<string[]>([]);
  const [lossType, setLossType] = useState<'epoch' | 'batch'>('epoch');
  const [selectedFiles, setSelectedFiles] = useState<string[]>([]);
  const [chartData, setChartData] = useState<any[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');

  // Fetch available files on mount
  useEffect(() => {
    const fetchFiles = async () => {
      try {
        const response = await api.get('/api/losses/files');
        setAllFiles(response.data.files);
        setSourcePath(response.data.source_path);
      } catch (err) {
        setError('Failed to fetch loss files from the server.');
      }
    };
    fetchFiles();
  }, []);

  // Filter files by type
  const filteredFiles = allFiles.filter(f => 
    lossType === 'epoch' ? f.includes('_epoch_losses.txt') : f.includes('_batch_losses.txt')
  );

  // Auto-select first 3 files when loss type changes
  useEffect(() => {
    if (filteredFiles.length > 0) {
      setSelectedFiles(filteredFiles.slice(0, 3));
    } else {
      setSelectedFiles([]);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [lossType, allFiles]);

  // Fetch data when selection changes
  useEffect(() => {
    const fetchData = async () => {
      if (selectedFiles.length === 0) {
        setChartData([]);
        return;
      }

      setLoading(true);
      try {
        const response = await api.post<LossDataResponse>('/api/losses/data', {
          files: selectedFiles
        });

        // Transform data for Recharts
        // Find the maximum length of values to know how many iterations we have
        const lengths = response.data.data.map((d: any) => d.values.length);
        const maxLength = lengths.length > 0 ? Math.max(...lengths) : 0;
        
        const transformedData = [];
        for (let i = 0; i < maxLength; i++) {
          const dataPoint: any = { iteration: i + 1 };
          response.data.data.forEach(d => {
            if (i < d.values.length) {
              dataPoint[d.short_name] = d.values[i];
            }
          });
          transformedData.push(dataPoint);
        }

        setChartData(transformedData);
        setError('');
      } catch (err) {
        setError('Failed to load chart data.');
      } finally {
        setLoading(false);
      }
    };

    fetchData();
  }, [selectedFiles]);

  const toggleFileSelection = (filename: string) => {
    setSelectedFiles(prev => {
      if (prev.includes(filename)) {
        return prev.filter(f => f !== filename);
      } else {
        return [...prev, filename];
      }
    });
  };

  return (
    <div className="max-w-6xl mx-auto flex flex-col gap-6 pt-2">
      <div>
        <h2 className="text-2xl font-bold mb-2 flex items-center gap-2">
          <TrendingUp className="w-6 h-6 text-emerald-400" />
          Training Loss Viewer
        </h2>
        <p className="text-gray-400">
          Visualize epoch and batch training losses across different model runs.
          {sourcePath && <span className="ml-2 text-xs text-gray-500">Source: <code className="text-gray-400">{sourcePath}</code></span>}
        </p>
      </div>

      {error && (
        <div className="bg-red-900/50 border border-red-500 text-red-200 px-4 py-3 rounded">
          {error}
        </div>
      )}

      {allFiles.length === 0 && !error ? (
        <div className="text-center py-10 bg-gray-800 rounded border border-gray-700">
          <p className="text-gray-400">No loss files found in the directory.</p>
        </div>
      ) : (
        <div className="grid grid-cols-1 lg:grid-cols-4 gap-6">
          {/* Controls Panel */}
          <div className="lg:col-span-1 space-y-6">
            <div className="bg-gray-900 p-4 rounded-lg border border-gray-700">
              <h3 className="font-medium mb-3 text-sm text-gray-300 uppercase tracking-wider">Granularity</h3>
              <div className="flex gap-2">
                <button
                  onClick={() => setLossType('epoch')}
                  className={`flex-1 py-2 px-3 rounded text-sm font-medium transition-colors ${
                    lossType === 'epoch'
                      ? 'bg-blue-600 text-white'
                      : 'bg-gray-800 text-gray-400 hover:bg-gray-700'
                  }`}
                >
                  Epochs
                </button>
                <button
                  onClick={() => setLossType('batch')}
                  className={`flex-1 py-2 px-3 rounded text-sm font-medium transition-colors ${
                    lossType === 'batch'
                      ? 'bg-blue-600 text-white'
                      : 'bg-gray-800 text-gray-400 hover:bg-gray-700'
                  }`}
                >
                  Batches
                </button>
              </div>
            </div>

            <div className="bg-gray-900 p-4 rounded-lg border border-gray-700 max-h-[400px] overflow-y-auto">
              <h3 className="font-medium mb-3 text-sm text-gray-300 uppercase tracking-wider">
                Select Runs ({filteredFiles.length})
              </h3>
              {filteredFiles.length === 0 ? (
                <p className="text-sm text-gray-500">No {lossType} files available.</p>
              ) : (
                <div className="space-y-2">
                  {filteredFiles.map(filename => {
                    const shortName = filename.replace('_epoch_losses.txt', '').replace('_batch_losses.txt', '');
                    const isSelected = selectedFiles.includes(filename);
                    
                    return (
                      <label 
                        key={filename} 
                        className={`flex items-start gap-3 p-2 rounded cursor-pointer transition-colors ${
                          isSelected ? 'bg-gray-800' : 'hover:bg-gray-800/50'
                        }`}
                      >
                        <div className="mt-0.5">
                          <input
                            type="checkbox"
                            className="w-4 h-4 rounded border-gray-600 bg-gray-700 text-blue-500 focus:ring-blue-500 focus:ring-offset-gray-900"
                            checked={isSelected}
                            onChange={() => toggleFileSelection(filename)}
                          />
                        </div>
                        <span className={`text-sm break-all ${isSelected ? 'text-gray-200' : 'text-gray-400'}`}>
                          {shortName}
                        </span>
                      </label>
                    );
                  })}
                </div>
              )}
            </div>
          </div>

          {/* Chart Panel */}
          <div className="lg:col-span-3 bg-gray-900 p-6 rounded-lg border border-gray-700 flex flex-col min-h-[500px]">
            <h3 className="font-medium mb-6 text-center text-lg">
              {lossType === 'epoch' ? 'Epoch' : 'Batch'} Losses Over Training
            </h3>
            
            {loading ? (
              <div className="flex-1 flex items-center justify-center">
                <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-500"></div>
              </div>
            ) : chartData.length > 0 ? (
              <div className="flex-1 h-full w-full">
                <ResponsiveContainer width="100%" height={450}>
                  <LineChart
                    data={chartData}
                    margin={{ top: 5, right: 30, left: 20, bottom: 25 }}
                  >
                    <CartesianGrid strokeDasharray="3 3" stroke="#374151" vertical={false} />
                    <XAxis 
                      dataKey="iteration" 
                      stroke="#9ca3af" 
                      tick={{ fill: '#9ca3af' }}
                      label={{ 
                        value: lossType === 'epoch' ? 'Epoch' : 'Batch', 
                        position: 'insideBottom', 
                        offset: -15,
                        fill: '#9ca3af' 
                      }} 
                    />
                    <YAxis 
                      stroke="#9ca3af" 
                      tick={{ fill: '#9ca3af' }}
                      label={{ 
                        value: 'Loss', 
                        angle: -90, 
                        position: 'insideLeft',
                        fill: '#9ca3af' 
                      }} 
                    />
                    <Tooltip 
                      contentStyle={{ backgroundColor: '#1f2937', borderColor: '#374151', color: '#f3f4f6' }}
                      itemStyle={{ color: '#e5e7eb' }}
                      labelStyle={{ color: '#9ca3af', marginBottom: '8px' }}
                    />
                    <Legend 
                      wrapperStyle={{ paddingTop: '20px' }}
                    />
                    
                    {/* Render a Line for each selected file's short name */}
                    {selectedFiles.map((filename, index) => {
                      const shortName = filename.replace('_epoch_losses.txt', '').replace('_batch_losses.txt', '');
                      return (
                        <Line
                          key={shortName}
                          type="monotone"
                          dataKey={shortName}
                          stroke={COLORS[index % COLORS.length]}
                          activeDot={{ r: 6 }}
                          dot={false}
                          strokeWidth={2}
                        />
                      );
                    })}
                  </LineChart>
                </ResponsiveContainer>
              </div>
            ) : (
              <div className="flex-1 flex items-center justify-center text-gray-500">
                Select runs to visualize the losses
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  );
}
