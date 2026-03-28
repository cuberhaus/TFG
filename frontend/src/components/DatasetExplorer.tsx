import { useState, useEffect, useRef, useCallback } from 'react';
import axios from 'axios';
import { ChevronLeft, ChevronRight, RefreshCw, FolderSearch, Upload, CheckCircle, XCircle } from 'lucide-react';

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

interface UploadProgress {
  phase: 'scanning' | 'uploading' | 'done' | 'error';
  scanned: number;
  total: number;
  completed: number;
  message: string;
}

const ALLOWED_ROOTS = ['TrainValid', 'Test', 'PolypDataset', 'PolypDatasetSPADE'];
const BATCH_SIZE = 100;

async function readAllEntries(reader: FileSystemDirectoryReader): Promise<FileSystemEntry[]> {
  const all: FileSystemEntry[] = [];
  let batch: FileSystemEntry[];
  do {
    batch = await new Promise<FileSystemEntry[]>((resolve, reject) =>
      reader.readEntries(resolve, reject)
    );
    all.push(...batch);
  } while (batch.length > 0);
  return all;
}

async function traverseEntry(
  entry: FileSystemEntry,
  basePath: string,
  onFile: () => void
): Promise<{ file: File; path: string }[]> {
  if (entry.isFile) {
    const fileEntry = entry as FileSystemFileEntry;
    const file = await new Promise<File>((resolve, reject) =>
      fileEntry.file(resolve, reject)
    );
    onFile();
    return [{ file, path: basePath + entry.name }];
  }

  if (entry.isDirectory) {
    const dirEntry = entry as FileSystemDirectoryEntry;
    const children = await readAllEntries(dirEntry.createReader());
    const results: { file: File; path: string }[] = [];
    for (const child of children) {
      const childFiles = await traverseEntry(child, basePath + entry.name + '/', onFile);
      results.push(...childFiles);
    }
    return results;
  }

  return [];
}

export default function DatasetExplorer() {
  const [split, setSplit] = useState<'train' | 'test'>('train');
  const [page, setPage] = useState(1);
  const [data, setData] = useState<DatasetResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [showBoxes, setShowBoxes] = useState(true);

  const [isDragging, setIsDragging] = useState(false);
  const [uploadProgress, setUploadProgress] = useState<UploadProgress | null>(null);
  const dragCounter = useRef(0);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const limit = 12;

  const fetchDataset = useCallback(async (currentSplit: string, currentPage: number) => {
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
  }, []);

  useEffect(() => {
    fetchDataset(split, page);
  }, [split, page, fetchDataset]);

  const handleSplitChange = (newSplit: 'train' | 'test') => {
    setSplit(newSplit);
    setPage(1);
  };

  // --- Upload logic ---

  async function uploadBatched(files: { file: File; path: string }[]) {
    setUploadProgress({ phase: 'uploading', scanned: files.length, total: files.length, completed: 0, message: `Uploading ${files.length.toLocaleString()} files...` });

    let completed = 0;
    for (let i = 0; i < files.length; i += BATCH_SIZE) {
      const batch = files.slice(i, i + BATCH_SIZE);
      const formData = new FormData();
      const paths: string[] = [];

      for (const { file, path } of batch) {
        formData.append('files', file);
        paths.push(path);
      }
      formData.append('relative_paths', JSON.stringify(paths));

      try {
        await axios.post('http://localhost:8082/api/dataset/upload', formData);
      } catch (err: any) {
        setUploadProgress(prev => prev ? {
          ...prev,
          phase: 'error',
          message: `Upload failed at file ${completed}: ${err.response?.data?.detail || err.message}`
        } : null);
        return;
      }

      completed += batch.length;
      setUploadProgress(prev => prev ? {
        ...prev,
        completed,
        message: `Uploading... ${completed.toLocaleString()} / ${files.length.toLocaleString()} files`
      } : null);
    }

    setUploadProgress({
      phase: 'done',
      scanned: files.length,
      total: files.length,
      completed: files.length,
      message: `Successfully uploaded ${files.length.toLocaleString()} files`
    });

    setPage(1);
    fetchDataset(split, 1);
  }

  async function processDroppedItems(items: DataTransferItemList) {
    setUploadProgress({ phase: 'scanning', scanned: 0, total: 0, completed: 0, message: 'Scanning folders...' });

    const allFiles: { file: File; path: string }[] = [];
    let scanned = 0;

    const topEntries: FileSystemEntry[] = [];
    for (let i = 0; i < items.length; i++) {
      const entry = items[i].webkitGetAsEntry?.();
      if (entry) topEntries.push(entry);
    }

    if (topEntries.length === 0) {
      setUploadProgress({ phase: 'error', scanned: 0, total: 0, completed: 0, message: 'Could not read dropped items. Try using the "Browse Folders" button instead.' });
      return;
    }

    // Collect entries to scan: either direct allowed-root folders,
    // or look one level deeper inside parent folders for them.
    const entriesToScan: FileSystemEntry[] = [];

    for (const entry of topEntries) {
      if (!entry.isDirectory) continue;

      if (ALLOWED_ROOTS.includes(entry.name)) {
        entriesToScan.push(entry);
      } else {
        // Parent folder — look for allowed children inside it
        const children = await readAllEntries(
          (entry as FileSystemDirectoryEntry).createReader()
        );
        for (const child of children) {
          if (child.isDirectory && ALLOWED_ROOTS.includes(child.name)) {
            entriesToScan.push(child);
          }
        }
      }
    }

    if (entriesToScan.length === 0) {
      const names = topEntries.map(e => e.name).join(', ');
      setUploadProgress({
        phase: 'error', scanned: 0, total: 0, completed: 0,
        message: `Dropped "${names}" — no TrainValid, Test, PolypDataset, or PolypDatasetSPADE folder found inside.`
      });
      return;
    }

    const foundNames = entriesToScan.map(e => e.name).join(', ');
    setUploadProgress(prev => prev ? { ...prev, message: `Found ${foundNames}. Scanning files...` } : null);

    for (const entry of entriesToScan) {
      const files = await traverseEntry(entry, '', () => {
        scanned++;
        if (scanned % 200 === 0) {
          setUploadProgress(prev => prev ? { ...prev, scanned, message: `Scanning... ${scanned.toLocaleString()} files found` } : null);
        }
      });
      allFiles.push(...files);
    }

    if (allFiles.length === 0) {
      setUploadProgress({ phase: 'error', scanned: 0, total: 0, completed: 0, message: 'No files found inside the dropped folders.' });
      return;
    }

    setUploadProgress(prev => prev ? { ...prev, scanned: allFiles.length, message: `Found ${allFiles.length.toLocaleString()} files. Starting upload...` } : null);

    await uploadBatched(allFiles);
  }

  async function processFileInput(fileList: FileList) {
    setUploadProgress({ phase: 'scanning', scanned: 0, total: 0, completed: 0, message: 'Reading selected files...' });

    const allFiles: { file: File; path: string }[] = [];

    for (let i = 0; i < fileList.length; i++) {
      const file = fileList[i];
      const relativePath = (file as any).webkitRelativePath as string;
      if (!relativePath) continue;

      const parts = relativePath.split('/');

      // Direct match: TrainValid/Images/100/0001.jpg
      if (ALLOWED_ROOTS.includes(parts[0])) {
        allFiles.push({ file, path: relativePath });
        continue;
      }

      // Parent folder selected: MyData/TrainValid/Images/100/0001.jpg
      // Strip the parent and keep from the allowed root onward
      if (parts.length >= 2 && ALLOWED_ROOTS.includes(parts[1])) {
        allFiles.push({ file, path: parts.slice(1).join('/') });
      }
    }

    if (allFiles.length === 0) {
      setUploadProgress({
        phase: 'error', scanned: 0, total: 0, completed: 0,
        message: `No valid files found. The selected folder should be named ${ALLOWED_ROOTS.join(', ')}.`
      });
      return;
    }

    await uploadBatched(allFiles);
  }

  // --- Drag event handlers ---

  const handleDragEnter = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    dragCounter.current++;
    if (dragCounter.current === 1) setIsDragging(true);
  };

  const handleDragLeave = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    dragCounter.current--;
    if (dragCounter.current === 0) setIsDragging(false);
  };

  const handleDragOver = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    dragCounter.current = 0;
    setIsDragging(false);

    if (uploadProgress?.phase === 'uploading') return;
    processDroppedItems(e.dataTransfer.items);
  };

  const handleFileInputChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files.length > 0) {
      processFileInput(e.target.files);
    }
    e.target.value = '';
  };

  const isUploading = uploadProgress?.phase === 'uploading' || uploadProgress?.phase === 'scanning';

  return (
    <div
      className="max-w-6xl mx-auto flex flex-col gap-6 pt-2 relative"
      onDragEnter={handleDragEnter}
      onDragOver={handleDragOver}
      onDragLeave={handleDragLeave}
      onDrop={handleDrop}
    >
      {/* Drag overlay */}
      {isDragging && (
        <div className="absolute inset-0 bg-blue-900/50 border-2 border-dashed border-blue-400 rounded-xl flex items-center justify-center z-50 backdrop-blur-sm pointer-events-none">
          <div className="text-center">
            <Upload className="w-16 h-16 text-blue-400 mx-auto mb-4" />
            <p className="text-xl font-medium text-blue-200">Drop dataset folders here</p>
            <p className="text-sm text-blue-300/70 mt-2">TrainValid, Test, PolypDataset, or PolypDatasetSPADE</p>
          </div>
        </div>
      )}

      {/* Hidden file input for folder browsing */}
      <input
        ref={fileInputRef}
        type="file"
        className="hidden"
        onChange={handleFileInputChange}
        multiple
        {...{ webkitdirectory: '', directory: '' } as any}
      />

      {/* Header */}
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

          <div className="w-px h-6 bg-gray-700"></div>

          <button
            onClick={() => fileInputRef.current?.click()}
            disabled={isUploading}
            className="flex items-center gap-2 px-3 py-1.5 rounded-md text-sm font-medium text-gray-300 hover:text-white hover:bg-gray-700 transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
          >
            <Upload className="w-4 h-4" />
            Upload
          </button>
        </div>
      </div>

      {/* Upload progress bar */}
      {uploadProgress && (
        <div className={`rounded-xl p-4 border ${
          uploadProgress.phase === 'error'
            ? 'bg-red-900/20 border-red-800'
            : uploadProgress.phase === 'done'
            ? 'bg-green-900/20 border-green-800'
            : 'bg-blue-900/20 border-blue-800'
        }`}>
          <div className="flex items-center justify-between mb-2">
            <div className="flex items-center gap-2">
              {uploadProgress.phase === 'error' ? (
                <XCircle className="w-5 h-5 text-red-400" />
              ) : uploadProgress.phase === 'done' ? (
                <CheckCircle className="w-5 h-5 text-green-400" />
              ) : (
                <RefreshCw className="w-5 h-5 text-blue-400 animate-spin" />
              )}
              <span className={`text-sm font-medium ${
                uploadProgress.phase === 'error' ? 'text-red-300'
                  : uploadProgress.phase === 'done' ? 'text-green-300'
                  : 'text-blue-300'
              }`}>
                {uploadProgress.message}
              </span>
            </div>
            {(uploadProgress.phase === 'done' || uploadProgress.phase === 'error') && (
              <button
                onClick={() => setUploadProgress(null)}
                className="text-xs text-gray-400 hover:text-white transition-colors"
              >
                Dismiss
              </button>
            )}
          </div>
          {uploadProgress.phase === 'uploading' && uploadProgress.total > 0 && (
            <div className="w-full bg-gray-700 rounded-full h-2">
              <div
                className="bg-blue-500 h-2 rounded-full transition-all duration-300"
                style={{ width: `${(uploadProgress.completed / uploadProgress.total) * 100}%` }}
              />
            </div>
          )}
        </div>
      )}

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
        <div
          className="flex flex-col items-center justify-center py-16 text-gray-400 bg-gray-800/50 border-2 border-gray-700 rounded-xl border-dashed cursor-pointer hover:border-blue-600 hover:bg-blue-900/10 transition-colors"
          onClick={() => fileInputRef.current?.click()}
        >
          <Upload className="w-16 h-16 mb-4 text-gray-500" />
          <p className="text-xl font-medium text-gray-300">No images found in {split} split</p>
          <p className="text-sm mt-2">Drag & drop a <strong className="text-gray-200">{split === 'train' ? 'TrainValid' : 'Test'}</strong> folder here, or click to browse</p>
          <p className="text-xs mt-4 text-gray-500">
            Expected structure: {split === 'train' ? 'TrainValid' : 'Test'}/Images/&lt;sequence&gt;/*.jpg &amp; Annotations/&lt;sequence&gt;/*.txt
          </p>
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
