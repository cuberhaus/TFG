import { useState, useEffect, useRef, useCallback } from 'react';
import { api } from '../api';
import {
  ChevronLeft,
  ChevronRight,
  RefreshCw,
  FolderSearch,
  Upload,
  CheckCircle,
  XCircle,
  X,
  Image as ImageIcon,
  Maximize2,
  Ruler,
  Layers,
} from 'lucide-react';

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

  const [selectedImage, setSelectedImage] = useState<DatasetImage | null>(null);
  // When ←/→ navigation crosses a page boundary, we change `page` and need
  // to wait for the new page's data to arrive before picking which image
  // to focus on the modal. `pendingSelection` carries that intent across
  // the async fetch: 'first' means "select images[0] when data arrives",
  // 'last' means "select images[length-1]".
  const [pendingSelection, setPendingSelection] = useState<'first' | 'last' | null>(null);

  // Page-size options surfaced in the bottom selector. Kept conservative
  // upper bound — the backend has no hard cap, but 96 is already enough
  // thumbnails to slow rendering on low-end devices.
  const PAGE_SIZES = [12, 24, 48, 96] as const;
  const [limit, setLimit] = useState<number>(PAGE_SIZES[0]);

  // `currentLimit` is threaded through as a parameter (rather than read
  // from the closure) so the useCallback identity stays stable across
  // page-size changes. Otherwise every `setLimit` would recreate
  // fetchDataset and re-fire the [split, page, fetchDataset] effect twice.
  const fetchDataset = useCallback(async (currentSplit: string, currentPage: number, currentLimit: number) => {
    setLoading(true);
    setError(null);
    try {
      const response = await api.get(`/api/dataset/${currentSplit}?page=${currentPage}&limit=${currentLimit}`);
      setData(response.data);
    } catch (err: any) {
      console.error(err);
      setError(err.response?.data?.detail || "Failed to load dataset.");
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    fetchDataset(split, page, limit);
  }, [split, page, limit, fetchDataset]);

  // Switching page size: keep the user roughly in the same spot rather
  // than jumping back to page 1. We pin the index of the *first item
  // currently visible* and recompute which page contains it under the new
  // page size. The modal is closed because its `selectedImage` belongs to
  // the soon-to-be-stale `data.images` array — easier to close than to
  // chase the same image across the new page boundary.
  const handleLimitChange = (newLimit: number) => {
    if (newLimit === limit) return;
    const firstVisibleIndex = (page - 1) * limit; // 0-based
    const newPage = Math.floor(firstVisibleIndex / newLimit) + 1;
    setSelectedImage(null);
    setPendingSelection(null);
    setLimit(newLimit);
    setPage(newPage);
  };

  // Modal navigation across the *whole* dataset (not just the current page).
  // Returns true if a move was performed (so the caller can update e.g. the
  // status bar). Wraps are intentionally NOT supported — landing on the very
  // first image and pressing ← should be a no-op, same for last + →.
  const goToAdjacent = useCallback(
    (direction: 'prev' | 'next') => {
      if (!selectedImage || !data || loading) return false;
      const idx = data.images.findIndex((img) => img.id === selectedImage.id);
      if (idx === -1) return false;

      if (direction === 'prev') {
        if (idx > 0) {
          setSelectedImage(data.images[idx - 1]);
          return true;
        }
        if (page > 1) {
          setPendingSelection('last');
          setPage((p) => p - 1);
          return true;
        }
        return false;
      }

      if (idx < data.images.length - 1) {
        setSelectedImage(data.images[idx + 1]);
        return true;
      }
      if (page < (data.total_pages || 1)) {
        setPendingSelection('first');
        setPage((p) => p + 1);
        return true;
      }
      return false;
    },
    [selectedImage, data, loading, page]
  );

  // After a cross-page navigation, attach the modal to the right end of the
  // freshly-loaded page so ←/→ continues to feel uninterrupted.
  useEffect(() => {
    if (!data || !pendingSelection) return;
    const target =
      pendingSelection === 'first'
        ? data.images[0]
        : data.images[data.images.length - 1];
    if (target) setSelectedImage(target);
    setPendingSelection(null);
  }, [data, pendingSelection]);

  // Modal-scoped keyboard handlers: Esc to close, ←/→ to navigate. Mounted
  // here (not inside the modal) because the modal unmounts/remounts on every
  // selection change, which would briefly drop the listener at exactly the
  // moments the user is most likely to spam arrow keys.
  useEffect(() => {
    if (!selectedImage) return;
    const onKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape') {
        setSelectedImage(null);
        return;
      }
      if (e.key === 'ArrowLeft') {
        e.preventDefault();
        goToAdjacent('prev');
        return;
      }
      if (e.key === 'ArrowRight') {
        e.preventDefault();
        goToAdjacent('next');
      }
    };
    window.addEventListener('keydown', onKey);
    const previousOverflow = document.body.style.overflow;
    document.body.style.overflow = 'hidden';
    return () => {
      window.removeEventListener('keydown', onKey);
      document.body.style.overflow = previousOverflow;
    };
  }, [selectedImage, goToAdjacent]);

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
        await api.post('/api/dataset/upload', formData);
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
    fetchDataset(split, 1, limit);
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
            onClick={() => fetchDataset(split, page, limit)}
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
              <button
                key={item.id}
                type="button"
                onClick={() => setSelectedImage(item)}
                className="bg-gray-800 border border-gray-700 rounded-xl overflow-hidden shadow-lg group flex flex-col text-left hover:border-blue-600 hover:shadow-blue-900/30 hover:shadow-xl transition-all cursor-pointer focus:outline-none focus:ring-2 focus:ring-blue-500"
                title="Click to view larger image and details"
              >
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

                  {/* Hover affordance: small zoom badge in the corner. */}
                  <div className="absolute top-2 right-2 bg-gray-900/80 backdrop-blur-sm border border-gray-700 rounded-md p-1.5 opacity-0 group-hover:opacity-100 transition-opacity">
                    <Maximize2 className="w-3.5 h-3.5 text-blue-300" />
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
              </button>
            ))}
          </div>
          
          {/* Bottom Pagination + page-size selector. Always rendered when
              data is loaded so the user can change page size even on a
              single-page dataset. Page-navigation buttons collapse when
              there's only one page (no point showing them). */}
          {data && (
            <div className="flex justify-center mt-8 mb-4">
              <div className="flex items-center gap-3 bg-gray-800 p-2 rounded-xl border border-gray-700 flex-wrap justify-center">
                {/* Per-page segmented selector — mirrors the Train/Test
                    toggle styling in the header for visual consistency. */}
                <div className="flex items-center gap-2">
                  <span className="text-xs text-gray-500 pl-2 select-none">Per page</span>
                  <div className="flex bg-gray-900 rounded-lg p-1 border border-gray-700">
                    {PAGE_SIZES.map((size) => (
                      <button
                        key={size}
                        type="button"
                        onClick={() => handleLimitChange(size)}
                        disabled={loading && size !== limit}
                        aria-pressed={limit === size}
                        className={`px-3 py-1 rounded-md text-xs font-medium transition-colors disabled:opacity-50 disabled:cursor-not-allowed ${
                          limit === size
                            ? 'bg-blue-600 text-white shadow-sm'
                            : 'text-gray-400 hover:text-white hover:bg-gray-800'
                        }`}
                      >
                        {size}
                      </button>
                    ))}
                  </div>
                </div>

                {data.total_pages > 1 && (
                  <>
                    <div className="w-px h-6 bg-gray-700"></div>
                    <button
                      disabled={page === 1 || loading}
                      onClick={() => setPage(1)}
                      className="px-3 py-1.5 rounded-md text-sm text-gray-400 hover:text-white hover:bg-gray-700 disabled:opacity-50 disabled:cursor-not-allowed"
                    >
                      First
                    </button>
                    <button
                      disabled={page === 1 || loading}
                      onClick={() => setPage((p) => p - 1)}
                      className="p-1.5 rounded-md text-gray-400 hover:text-white hover:bg-gray-700 disabled:opacity-50 disabled:cursor-not-allowed"
                    >
                      <ChevronLeft className="w-5 h-5" />
                    </button>

                    <div className="px-4 text-sm font-medium text-gray-300">
                      Page {page} of {data.total_pages}
                    </div>

                    <button
                      disabled={page === data.total_pages || loading}
                      onClick={() => setPage((p) => p + 1)}
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
                  </>
                )}
              </div>
            </div>
          )}
        </div>
      )}

      {selectedImage && data && (() => {
        const idxInPage = data.images.findIndex((img) => img.id === selectedImage.id);
        const globalIndex =
          idxInPage === -1 ? null : (page - 1) * limit + idxInPage + 1;
        const canPrev = (idxInPage > 0) || (page > 1);
        const canNext =
          (idxInPage !== -1 && idxInPage < data.images.length - 1) ||
          page < (data.total_pages || 1);
        return (
          <DatasetImageModal
            image={selectedImage}
            showBoxes={showBoxes}
            split={split}
            onClose={() => setSelectedImage(null)}
            onPrev={() => goToAdjacent('prev')}
            onNext={() => goToAdjacent('next')}
            canPrev={canPrev}
            canNext={canNext}
            globalIndex={globalIndex}
            globalTotal={data.total}
            isLoading={loading}
          />
        );
      })()}
    </div>
  );
}

interface DatasetImageModalProps {
  image: DatasetImage;
  showBoxes: boolean;
  split: 'train' | 'test';
  onClose: () => void;
  onPrev: () => void;
  onNext: () => void;
  canPrev: boolean;
  canNext: boolean;
  globalIndex: number | null;
  globalTotal: number;
  isLoading: boolean;
}

function DatasetImageModal({
  image,
  showBoxes,
  split,
  onClose,
  onPrev,
  onNext,
  canPrev,
  canNext,
  globalIndex,
  globalTotal,
  isLoading,
}: DatasetImageModalProps) {
  const [localShowBoxes, setLocalShowBoxes] = useState(showBoxes);

  const aspectRatio = image.original_width / image.original_height;
  const totalPixels = image.original_width * image.original_height;

  const polypCount = image.original_boxes.length;
  const totalBoxArea = image.original_boxes.reduce((sum, [xmin, ymin, xmax, ymax]) => {
    return sum + (xmax - xmin) * (ymax - ymin);
  }, 0);
  const coverage = totalPixels > 0 ? (totalBoxArea / totalPixels) * 100 : 0;

  return (
    <div
      className="fixed inset-0 z-40 flex items-center justify-center bg-black/80 backdrop-blur-sm p-4 sm:p-8"
      onClick={onClose}
      role="dialog"
      aria-modal="true"
      aria-labelledby="dataset-modal-title"
    >
      <div
        className="bg-gray-900 border border-gray-700 rounded-2xl shadow-2xl w-full max-w-6xl max-h-full flex flex-col overflow-hidden"
        onClick={(e) => e.stopPropagation()}
      >
        {/* Header */}
        <div className="flex items-center justify-between px-5 py-3 border-b border-gray-700 bg-gray-800 flex-shrink-0">
          <div className="flex items-center gap-3 min-w-0">
            <ImageIcon className="w-5 h-5 text-blue-400 flex-shrink-0" />
            <div className="min-w-0">
              <h2
                id="dataset-modal-title"
                className="text-base font-semibold text-gray-100 truncate"
                title={image.filename}
              >
                {image.filename}
              </h2>
              <p className="text-xs text-gray-500 truncate" title={image.subdir}>
                {split === 'train' ? 'Train / Valid' : 'Test'} · {image.subdir}
              </p>
            </div>
          </div>
          <button
            type="button"
            onClick={onClose}
            className="p-2 rounded-md text-gray-400 hover:text-white hover:bg-gray-700 transition-colors flex-shrink-0"
            aria-label="Close (Esc)"
          >
            <X className="w-5 h-5" />
          </button>
        </div>

        {/* Body — image left, metadata right (stacked on small screens) */}
        <div className="flex flex-col md:flex-row min-h-0 flex-1 overflow-hidden">
          <div className="relative flex-1 bg-black flex items-center justify-center p-4 overflow-auto min-h-[300px]">
            {/* Prev / next overlay buttons. Positioned absolutely on the
                image pane so they don't shift the image as it loads. The
                arrow-key handlers in DatasetExplorer do the real work; these
                are the visual affordance + a fallback for touch users. */}
            <button
              type="button"
              onClick={onPrev}
              disabled={!canPrev || isLoading}
              aria-label="Previous image (Left arrow)"
              title="Previous image (←)"
              className="absolute left-3 top-1/2 -translate-y-1/2 z-10 p-2 rounded-full bg-gray-900/70 hover:bg-gray-900 border border-gray-700 text-gray-200 hover:text-white transition-colors disabled:opacity-30 disabled:cursor-not-allowed disabled:hover:bg-gray-900/70 backdrop-blur-sm"
            >
              <ChevronLeft className="w-6 h-6" />
            </button>
            <button
              type="button"
              onClick={onNext}
              disabled={!canNext || isLoading}
              aria-label="Next image (Right arrow)"
              title="Next image (→)"
              className="absolute right-3 top-1/2 -translate-y-1/2 z-10 p-2 rounded-full bg-gray-900/70 hover:bg-gray-900 border border-gray-700 text-gray-200 hover:text-white transition-colors disabled:opacity-30 disabled:cursor-not-allowed disabled:hover:bg-gray-900/70 backdrop-blur-sm"
            >
              <ChevronRight className="w-6 h-6" />
            </button>

            <div className="relative inline-block max-w-full">
              <img
                src={image.image}
                alt={image.filename}
                className="max-w-full max-h-[75vh] object-contain rounded"
              />

              {localShowBoxes && image.boxes.map((box, idx) => {
                const [xmin, ymin, xmax, ymax] = box;
                return (
                  <div
                    key={idx}
                    className="absolute border-2 border-green-500 bg-green-500/15 pointer-events-none"
                    style={{
                      left: `${(xmin / image.width) * 100}%`,
                      top: `${(ymin / image.height) * 100}%`,
                      width: `${((xmax - xmin) / image.width) * 100}%`,
                      height: `${((ymax - ymin) / image.height) * 100}%`,
                    }}
                  >
                    <span className="absolute -top-5 left-0 bg-green-500 text-black text-[10px] font-bold px-1 rounded-sm whitespace-nowrap">
                      Polyp #{idx + 1}
                    </span>
                  </div>
                );
              })}
            </div>
          </div>

          {/* Metadata panel */}
          <div className="md:w-80 md:flex-shrink-0 border-t md:border-t-0 md:border-l border-gray-700 bg-gray-800/60 overflow-y-auto">
            <div className="p-5 space-y-5">
              <ModalSection title="Overview" icon={ImageIcon}>
                <ModalStat label="Polyps" value={String(polypCount)} accent />
                <ModalStat
                  label="Coverage"
                  value={polypCount > 0 ? `${coverage.toFixed(2)}%` : '—'}
                  hint={polypCount > 0 ? 'Total bbox area / image area' : undefined}
                />
                <ModalStat label="Subdir" value={image.subdir} mono />
                <ModalStat label="Source" value={split === 'train' ? 'Train / Valid' : 'Test'} />
              </ModalSection>

              <ModalSection title="Dimensions" icon={Ruler}>
                <ModalStat
                  label="Original"
                  value={`${image.original_width} × ${image.original_height}`}
                  mono
                />
                <ModalStat
                  label="Aspect ratio"
                  value={aspectRatio.toFixed(3)}
                  mono
                  hint={aspectRatio > 1 ? 'Landscape' : aspectRatio < 1 ? 'Portrait' : 'Square'}
                />
                <ModalStat
                  label="Total pixels"
                  value={totalPixels.toLocaleString()}
                  mono
                />
              </ModalSection>

              <ModalSection title={`Bounding boxes (${polypCount})`} icon={Layers}>
                {polypCount === 0 ? (
                  <p className="text-xs text-gray-500 italic">
                    No polyp annotations on this image.
                  </p>
                ) : (
                  <ul className="space-y-2">
                    {image.original_boxes.map((box, idx) => {
                      const [xmin, ymin, xmax, ymax] = box;
                      const width = xmax - xmin;
                      const height = ymax - ymin;
                      const area = width * height;
                      const pct = totalPixels > 0 ? (area / totalPixels) * 100 : 0;
                      return (
                        <li
                          key={idx}
                          className="bg-gray-900/60 border border-gray-700 rounded-lg p-2.5 text-[11px]"
                        >
                          <div className="flex items-center justify-between mb-1.5">
                            <span className="text-gray-300 font-medium">
                              Polyp #{idx + 1}
                            </span>
                            <span className="text-green-400 font-mono">
                              {pct.toFixed(2)}%
                            </span>
                          </div>
                          <div className="grid grid-cols-2 gap-x-3 gap-y-0.5 text-gray-400 font-mono">
                            <span>x: {Math.round(xmin)}–{Math.round(xmax)}</span>
                            <span>y: {Math.round(ymin)}–{Math.round(ymax)}</span>
                            <span>w: {Math.round(width)}px</span>
                            <span>h: {Math.round(height)}px</span>
                          </div>
                        </li>
                      );
                    })}
                  </ul>
                )}
              </ModalSection>

              {/* Per-modal toggle so the user can flip boxes off here
                  without losing their grid-level setting. Initialized
                  from the parent's setting. */}
              <label className="flex items-center gap-2 text-xs text-gray-300 cursor-pointer select-none">
                <input
                  type="checkbox"
                  checked={localShowBoxes}
                  onChange={(e) => setLocalShowBoxes(e.target.checked)}
                  className="rounded border-gray-600 bg-gray-900 text-blue-500 focus:ring-blue-600 focus:ring-offset-gray-800"
                />
                Show bounding boxes in this view
              </label>
            </div>
          </div>
        </div>

        {/* Footer */}
        <div className="px-5 py-2.5 border-t border-gray-700 bg-gray-800/80 text-[11px] text-gray-500 flex items-center justify-between gap-4 flex-shrink-0 flex-wrap">
          <span className="flex items-center gap-1.5 flex-wrap">
            <kbd className="px-1 py-0.5 bg-gray-900 border border-gray-700 rounded font-mono">←</kbd>
            <kbd className="px-1 py-0.5 bg-gray-900 border border-gray-700 rounded font-mono">→</kbd>
            <span>navigate</span>
            <span className="text-gray-700">·</span>
            <kbd className="px-1 py-0.5 bg-gray-900 border border-gray-700 rounded font-mono">Esc</kbd>
            <span>close</span>
          </span>
          {globalIndex !== null && (
            <span className="font-mono text-gray-400">
              {globalIndex.toLocaleString()} / {globalTotal.toLocaleString()}
            </span>
          )}
          <span className="font-mono truncate min-w-0">id: {image.id}</span>
        </div>
      </div>
    </div>
  );
}

function ModalSection({
  title,
  icon: Icon,
  children,
}: {
  title: string;
  icon: typeof ImageIcon;
  children: React.ReactNode;
}) {
  return (
    <div>
      <h3 className="flex items-center gap-1.5 text-[10px] uppercase tracking-wider text-gray-500 mb-2">
        <Icon className="w-3 h-3" />
        {title}
      </h3>
      <div className="space-y-1.5">{children}</div>
    </div>
  );
}

function ModalStat({
  label,
  value,
  hint,
  mono,
  accent,
}: {
  label: string;
  value: string;
  hint?: string;
  mono?: boolean;
  accent?: boolean;
}) {
  return (
    <div className="flex items-baseline justify-between gap-3">
      <span className="text-xs text-gray-400 flex-shrink-0">{label}</span>
      <span className="text-right min-w-0">
        <span
          className={`text-sm ${mono ? 'font-mono' : ''} ${
            accent ? 'text-blue-300 font-semibold' : 'text-gray-100'
          } truncate block`}
          title={value}
        >
          {value}
        </span>
        {hint && <span className="text-[10px] text-gray-500 block">{hint}</span>}
      </span>
    </div>
  );
}
