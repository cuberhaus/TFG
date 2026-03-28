import { useState, useEffect } from 'react';
import axios from 'axios';
import { Database, ChevronLeft, ChevronRight, RefreshCw, FolderSearch } from 'lucide-react';

interface DatasetImage {
  id: string;
  subdir: string;
  filename: string;
  image: string;
  width: number;
  height: number;
  boxes: number[][]; // [xmin, ymin, xmax, ymax]
  original_boxes: number[][];
  original_width: number;
  original_height: number;
}

interface DatasetResponse {
  images: DatasetImage[];
  total: number;
  page: number;
  limit: number;
  total_pages: number;
}

export default function DatasetExplorer() {
  const [split, setSplit] = useState<'train' | 'test'>('train');
  const [page, setPage] = useState(1);
  const [data, setData] = useState<DatasetResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [showBoxes, setShowBoxes] = useState(true);

  const limit = 12;

  const fetchDataset = async (currentSplit: string, currentPage: number) => {
    setLoading(true);
    setError(null);
    try {
      const response = await axios.get(`http://localhost:8082/api/dataset/${currentSplit}?page=${currentPage}&limit=${limit}`);
      setData(response.data);
    } catch (err: any) {
      console.error(err);
      setError(err.response?.data?.detail || "Failed to load dataset.");
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchDataset(split, page);
  }, [split, page]);

  const handleSplitChange = (newSplit: 'train' | 'test') => {
    setSplit(newSplit);
    setPage(1);
  };

  return (
    <div className="max-w-6xl mx-auto flex flex-col gap-6 pt-2">
      <div className="flex items-start justify-between flex-wrap gap-4">
        <div>
          <h2 className="text-2xl font-bold mb-2 flex items-center gap-2">
            <FolderSearch className="w-6 h-6 text-blue-400" />
            Dataset Explorer
          </h2>
          <p className="text-gray-400 max-w-2xl">
            Browse through the original dataset images and inspect the ground truth polyp bounding box annotations.
          </p>
        </div>
        
        <div className="flex items-center gap-4 bg-gray-800 p-2 rounded-xl border border-gray-700">
          <div className="flex bg-gray-900 rounded-lg p-1 border border-gray-700">
            <button
              onClick={() => handleSplitChange('train')}
              className={`px-4 py-1.5 rounded-md text-sm font-medium transition-colors ${
                split === 'train' 
                  ? 'bg-blue-600 text-white shadow-sm' 
                  : 'text-gray-400 hover:text-white hover:bg-gray-800'
              }`}
            >
              Train / Valid
            </button>
            <button
              onClick={() => handleSplitChange('test')}
              className={`px-4 py-1.5 rounded-md text-sm font-medium transition-colors ${
                split === 'test' 
                  ? 'bg-blue-600 text-white shadow-sm' 
                  : 'text-gray-400 hover:text-white hover:bg-gray-800'
              }`}
            >
              Test
            </button>
          </div>
          
          <div className="w-px h-6 bg-gray-700"></div>
          
          <label className="flex items-center gap-2 text-sm text-gray-300 cursor-pointer select-none px-2">
            <input 
              type="checkbox" 
              checked={showBoxes} 
              onChange={(e) => setShowBoxes(e.target.checked)}
              className="rounded border-gray-600 bg-gray-900 text-blue-500 focus:ring-blue-600 focus:ring-offset-gray-800"
            />
            Show Bounding Boxes
          </label>
        </div>
      </div>

      {error ? (
        <div className="bg-red-900/30 border border-red-800 text-red-300 p-6 rounded-xl text-center">
          <p>{error}</p>
          <button 
            onClick={() => fetchDataset(split, page)}
            className="mt-4 px-4 py-2 bg-red-800 hover:bg-red-700 text-white rounded-lg text-sm transition-colors"
          >
            Try Again
          </button>
        </div>
      ) : loading && !data ? (
        <div className="flex flex-col items-center justify-center py-24 text-gray-400">
          <RefreshCw className="w-10 h-10 animate-spin mb-4 text-blue-500" />
          <p className="text-lg">Loading dataset {split}...</p>
        </div>
      ) : data?.images.length === 0 ? (
        <div className="flex flex-col items-center justify-center py-24 text-gray-400 bg-gray-800/50 border border-gray-700 rounded-xl border-dashed">
          <Database className="w-16 h-16 mb-4 text-gray-600" />
          <p className="text-xl font-medium text-gray-300">No images found in {split} split.</p>
          <p className="text-sm mt-2">Ensure that you have uploaded the dataset to the correct folders:</p>
          <code className="text-xs mt-4 p-2 bg-gray-900 rounded text-gray-500">
            data/{split === 'train' ? 'TrainValid' : 'Test'}/Images<br/>
            data/{split === 'train' ? 'TrainValid' : 'Test'}/Annotations
          </code>
        </div>
      ) : (
        <div className="space-y-6">
          <div className="flex justify-between items-center bg-gray-800/50 px-4 py-3 rounded-lg border border-gray-700">
            <span className="text-sm text-gray-400">
              Showing <span className="font-medium text-white">{(page - 1) * limit + 1}</span> to <span className="font-medium text-white">{Math.min(page * limit, data?.total || 0)}</span> of <span className="font-medium text-white">{data?.total}</span> images
            </span>
            
            <div className="flex items-center gap-2">
              <button 
                disabled={page === 1 || loading}
                onClick={() => setPage(p => p - 1)}
                className="p-1.5 rounded-md bg-gray-700 text-gray-300 hover:bg-gray-600 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
              >
                <ChevronLeft className="w-5 h-5" />
              </button>
              <span className="text-sm font-medium text-gray-300 min-w-[3rem] text-center">
                {page} / {data?.total_pages || 1}
              </span>
              <button 
                disabled={page === (data?.total_pages || 1) || loading}
                onClick={() => setPage(p => p + 1)}
                className="p-1.5 rounded-md bg-gray-700 text-gray-300 hover:bg-gray-600 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
              >
                <ChevronRight className="w-5 h-5" />
              </button>
            </div>
          </div>

          <div className={`grid grid-cols-1 sm:grid-cols-2 md:grid-cols-3 lg:grid-cols-4 gap-6 transition-opacity duration-300 ${loading ? 'opacity-50' : 'opacity-100'}`}>
            {data?.images.map((item) => (
              <div key={item.id} className="bg-gray-800 border border-gray-700 rounded-xl overflow-hidden shadow-lg group flex flex-col">
                <div className="relative bg-gray-900 flex-1 flex items-center justify-center p-2 min-h-[200px]">
                  <div className="relative inline-block max-w-full">
                    <img 
                      src={item.image} 
                      alt={item.filename} 
                      className="max-w-full max-h-[250px] object-contain rounded"
                    />
                    
                    {showBoxes && item.boxes.map((box, idx) => {
                      const [xmin, ymin, xmax, ymax] = box;
                      return (
                        <div 
                          key={idx}
                          className="absolute border-2 border-green-500 bg-green-500/20"
                          style={{
                            left: `${(xmin / item.width) * 100}%`,
                            top: `${(ymin / item.height) * 100}%`,
                            width: `${((xmax - xmin) / item.width) * 100}%`,
                            height: `${((ymax - ymin) / item.height) * 100}%`,
                          }}
                        >
                          <span className="absolute -top-5 left-0 bg-green-500 text-black text-[10px] font-bold px-1 rounded-sm whitespace-nowrap">
                            Polyp
                          </span>
                        </div>
                      );
                    })}
                  </div>
                </div>
                
                <div className="p-3 border-t border-gray-700 bg-gray-800">
                  <div className="flex justify-between items-start mb-1">
                    <span className="text-sm font-medium text-gray-200 truncate" title={item.filename}>
                      {item.filename}
                    </span>
                    <span className="text-xs font-mono text-gray-500 bg-gray-900 px-1.5 py-0.5 rounded">
                      {item.original_width}x{item.original_height}
                    </span>
                  </div>
                  <div className="flex justify-between items-center text-xs text-gray-400">
                    <span className="flex items-center gap-1">
                      <FolderSearch className="w-3 h-3" />
                      {item.subdir}
                    </span>
                    <span className="bg-blue-900/30 text-blue-400 px-2 py-0.5 rounded-full border border-blue-800/50">
                      {item.boxes.length} {item.boxes.length === 1 ? 'polyp' : 'polyps'}
                    </span>
                  </div>
                </div>
              </div>
            ))}
          </div>
          
          {/* Bottom Pagination */}
          {data && data.total_pages > 1 && (
            <div className="flex justify-center mt-8 mb-4">
              <div className="flex items-center gap-2 bg-gray-800 p-2 rounded-xl border border-gray-700">
                <button 
                  disabled={page === 1 || loading}
                  onClick={() => setPage(1)}
                  className="px-3 py-1.5 rounded-md text-sm text-gray-400 hover:text-white hover:bg-gray-700 disabled:opacity-50 disabled:cursor-not-allowed"
                >
                  First
                </button>
                <button 
                  disabled={page === 1 || loading}
                  onClick={() => setPage(p => p - 1)}
                  className="p-1.5 rounded-md text-gray-400 hover:text-white hover:bg-gray-700 disabled:opacity-50 disabled:cursor-not-allowed"
                >
                  <ChevronLeft className="w-5 h-5" />
                </button>
                
                <div className="px-4 text-sm font-medium text-gray-300">
                  Page {page} of {data.total_pages}
                </div>
                
                <button 
                  disabled={page === data.total_pages || loading}
                  onClick={() => setPage(p => p + 1)}
                  className="p-1.5 rounded-md text-gray-400 hover:text-white hover:bg-gray-700 disabled:opacity-50 disabled:cursor-not-allowed"
                >
                  <ChevronRight className="w-5 h-5" />
                </button>
                <button 
                  disabled={page === data.total_pages || loading}
                  onClick={() => setPage(data.total_pages)}
                  className="px-3 py-1.5 rounded-md text-sm text-gray-400 hover:text-white hover:bg-gray-700 disabled:opacity-50 disabled:cursor-not-allowed"
                >
                  Last
                </button>
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  );
}