import { useState, useEffect } from 'react';
import { api } from '../api';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';
import { BarChart3 } from 'lucide-react';

interface PerformanceRow {
  Model: string;
  BATCH_SIZE: number;
  LR: number;
  LR_fmt: string;
  NUM_EPOCHS: number;
  Config: string;
  F1: number;
  AP_50_all: number;
  AP_50_95_all: number;
  AP_75_all: number;
  AR_50_95_all_maxDets_100: number;
  [key: string]: string | number;
}

export default function PerformanceExplorer() {
  const [data, setData] = useState<PerformanceRow[]>([]);
  const [loading, setLoading] = useState(true);
  const [sourcePath, setSourcePath] = useState('');

  useEffect(() => {
    const fetchData = async () => {
      try {
        const response = await api.get('/api/performance');
        if (response.data && response.data.data) {
          setData(response.data.data);
          if (response.data.source_path) {
            setSourcePath(response.data.source_path);
          }
        }
      } catch (error) {
        console.error('Error fetching performance data:', error);
      } finally {
        setLoading(false);
      }
    };
    fetchData();
  }, []);

  if (loading) {
    return <div className="text-center py-10">Loading performance data...</div>;
  }

  if (data.length === 0) {
    return (
      <div className="text-center py-10 text-yellow-500">
        No performance data found. Make sure the backend is running and the CSV exists.
      </div>
    );
  }

  // Find the best model by F1 score
  const bestModel = [...data].sort((a, b) => (b.F1 || 0) - (a.F1 || 0))[0];

  return (
    <div className="max-w-7xl mx-auto flex flex-col gap-8 pt-2">
      <div className="flex justify-between items-start">
        <div>
          <h2 className="text-2xl font-bold mb-2 flex items-center gap-2">
            <BarChart3 className="w-6 h-6 text-indigo-400" />
            Performance Explorer
          </h2>
          <p className="text-gray-400">
            Compare COCO detection metrics across all evaluated models.
            {sourcePath && <span className="ml-2 text-xs text-gray-500">Source: <code className="text-gray-400">{sourcePath}</code></span>}
          </p>
        </div>
      </div>

      {/* KPI Cards */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
        <div className="bg-gray-800 p-4 rounded-xl border border-gray-700">
          <p className="text-sm text-gray-400">Best F1 Score</p>
          <p className="text-2xl font-bold text-blue-400">{bestModel?.F1?.toFixed(4)}</p>
        </div>
        <div className="bg-gray-800 p-4 rounded-xl border border-gray-700">
          <p className="text-sm text-gray-400">Best AP@50</p>
          <p className="text-2xl font-bold text-green-400">{bestModel?.AP_50_all?.toFixed(4)}</p>
        </div>
        <div className="bg-gray-800 p-4 rounded-xl border border-gray-700">
          <p className="text-sm text-gray-400">Best AR@100</p>
          <p className="text-2xl font-bold text-purple-400">{bestModel?.AR_50_95_all_maxDets_100?.toFixed(4)}</p>
        </div>
        <div className="bg-gray-800 p-4 rounded-xl border border-gray-700">
          <p className="text-sm text-gray-400">Total Models</p>
          <p className="text-2xl font-bold text-gray-200">{data.length}</p>
        </div>
      </div>

      {/* Chart */}
      <div className="h-96 w-full mt-2 bg-gray-800 p-4 rounded-xl border border-gray-700">
        <h3 className="text-lg font-medium mb-4 text-center">Detection Metrics by Configuration</h3>
        <ResponsiveContainer width="100%" height="100%">
          <BarChart
            data={[...data].sort((a, b) => (b.F1 || 0) - (a.F1 || 0)).slice(0, 10)}
            margin={{ top: 20, right: 30, left: 20, bottom: 5 }}
          >
            <CartesianGrid strokeDasharray="3 3" stroke="#4b5563" />
            <XAxis dataKey="Config" stroke="#9ca3af" tick={{fontSize: 12}} hide />
            <YAxis stroke="#9ca3af" />
            <Tooltip contentStyle={{ backgroundColor: '#1f2937', borderColor: '#374151' }} />
            <Legend />
            <Bar dataKey="F1" fill="#3b82f6" name="F1 Score" />
            <Bar dataKey="AP_50_all" fill="#10b981" name="AP@50" />
            <Bar dataKey="AP_75_all" fill="#8b5cf6" name="AP@75" />
          </BarChart>
        </ResponsiveContainer>
      </div>
      <p className="text-xs text-gray-400 text-center -mt-4">Hover over bars for details. Showing top models.</p>

      {/* Table */}
      <div className="bg-gray-800 border border-gray-700 rounded-xl overflow-hidden">
        <div className="p-5 border-b border-gray-700">
          <h3 className="text-lg font-medium text-gray-200">Detailed Results</h3>
        </div>
        <div className="overflow-x-auto">
          <table className="w-full text-left border-collapse text-sm">
            <thead>
              <tr className="border-b border-gray-700 text-gray-400">
                <th className="py-3 px-4 font-medium">Model</th>
                <th className="py-3 px-4 font-medium">Batch Size</th>
                <th className="py-3 px-4 font-medium">LR</th>
                <th className="py-3 px-4 font-medium">Epochs</th>
                <th className="py-3 px-4 font-medium text-blue-400">F1</th>
                <th className="py-3 px-4 font-medium text-green-400">AP@50</th>
                <th className="py-3 px-4 font-medium">AP@75</th>
              </tr>
            </thead>
            <tbody>
              {[...data].sort((a, b) => (b.F1 || 0) - (a.F1 || 0)).map((row, idx) => (
                <tr key={idx} className="border-b border-gray-700/50 hover:bg-gray-700/30 transition-colors">
                  <td className="py-2.5 px-4">{row.Model}</td>
                  <td className="py-2.5 px-4">{row.BATCH_SIZE}</td>
                  <td className="py-2.5 px-4 font-mono text-gray-400">{row.LR_fmt}</td>
                  <td className="py-2.5 px-4">{row.NUM_EPOCHS}</td>
                  <td className="py-2.5 px-4 font-medium text-blue-400">{row.F1?.toFixed(4)}</td>
                  <td className="py-2.5 px-4 font-medium text-green-400">{row.AP_50_all?.toFixed(4)}</td>
                  <td className="py-2.5 px-4 font-mono text-gray-400">{row.AP_75_all?.toFixed(4)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}
