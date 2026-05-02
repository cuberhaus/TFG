import { useState, useEffect, useRef } from 'react';
import { api } from '../api';
import { Play, Square, X, Sparkles, CheckCircle, AlertCircle, RefreshCw, Image as ImageIcon, Wand2, ArrowRight, FolderTree, Info, ChevronDown, ChevronUp } from 'lucide-react';

interface GenStatus {
  is_running: boolean;
  current_task: string | null;
  message: string;
}

interface PrepStatus {
  is_running: boolean;
  message: string;
}

interface GenImage {
  base_name: string;
  real: string | null;
  fake: string | null;
}

export type GenerativeView = 'tasks' | 'cyclegan-results' | 'spade-results';

interface GenerativeAugmentationProps {
  // Which sub-view to render. Driven by the sidebar so users can deep-link
  // to the gallery without going through the in-page CTA. When omitted,
  // the component runs in self-managed mode (back-compat / standalone).
  view?: GenerativeView;
  onNavigate?: (view: GenerativeView) => void;
}

export default function GenerativeAugmentation({
  view: viewProp,
  onNavigate,
}: GenerativeAugmentationProps = {}) {
  const [taskType, setTaskType] = useState('train_cyclegan');
  const [experimentName, setExperimentName] = useState('');
  const [availableExperiments, setAvailableExperiments] = useState<Record<string, string[]>>({});
  const [epoch, setEpoch] = useState('latest');

  const [spadeExperimentName, setSpadeExperimentName] = useState('');
  const [availableSpadeExperiments, setAvailableSpadeExperiments] = useState<Record<string, string[]>>({});
  const [spadeEpoch, setSpadeEpoch] = useState('latest');

  const [cganBatchSize, setCganBatchSize] = useState(4);
  const [cganEpochs, setCganEpochs] = useState(5);
  const [cganLr, setCganLr] = useState(0.0002);
  const [cganNetG, setCganNetG] = useState('resnet_9blocks');
  const [cganLoadSize, setCganLoadSize] = useState(286);
  const [cganCropSize, setCganCropSize] = useState(256);
  const [cganMaxDataset, setCganMaxDataset] = useState<number | ''>('');

  const [spadeBatchSize, setSpadeBatchSize] = useState(1);
  const [spadeNiter, setSpadeNiter] = useState(50);
  const [spadeNiterDecay, setSpadeNiterDecay] = useState(0);
  const [spadeLr, setSpadeLr] = useState(0.0002);
  const [spadeNetG, setSpadeNetG] = useState('spade');
  const [spadeLoadSize, setSpadeLoadSize] = useState(1024);
  const [spadeCropSize, setSpadeCropSize] = useState(512);
  const [spadeMaxDataset, setSpadeMaxDataset] = useState<number | ''>('');
  
  const [status, setStatus] = useState<GenStatus>({
    is_running: false,
    current_task: null,
    message: 'Idle'
  });
  const [error, setError] = useState<string | null>(null);
  const [lastResult, setLastResult] = useState<{ type: 'success' | 'error' | 'cancelled'; message: string } | null>(null);
  const wasRunning = useRef(false);

  const [prepStatus, setPrepStatus] = useState<PrepStatus>({ is_running: false, message: 'Idle' });
  const [prepError, setPrepError] = useState<string | null>(null);
  const [datasetCheck, setDatasetCheck] = useState<{
    datasets: Record<string, Record<string, number>>;
    source_available: boolean;
    ready: boolean;
  } | null>(null);
  
  // Gallery state. The "current view" (tasks vs. cyclegan-results vs.
  // spade-results) is normally driven by the sidebar via `viewProp`. When
  // the component is used standalone (no prop), we fall back to local
  // state so the in-page transitions still work — useful for tests.
  const [internalView, setInternalView] = useState<GenerativeView>('tasks');
  const view: GenerativeView = viewProp ?? internalView;
  const setView = (v: GenerativeView) => {
    if (onNavigate) onNavigate(v);
    else setInternalView(v);
  };
  const showGallery = view !== 'tasks';
  const galleryType: 'cyclegan' | 'spade' =
    view === 'spade-results' ? 'spade' : 'cyclegan';

  const [galleryImages, setGalleryImages] = useState<GenImage[]>([]);
  const [galleryLoading, setGalleryLoading] = useState(false);
  const [galleryExp, setGalleryExp] = useState('');
  const [galleryEpoch, setGalleryEpoch] = useState('');

  const fetchStatus = async () => {
    try {
      const response = await api.get('/api/generate/status');
      const newStatus = response.data;

      if (wasRunning.current && !newStatus.is_running) {
        const msg = newStatus.message;
        if (msg.toLowerCase().includes('cancelled')) {
          setLastResult({ type: 'cancelled', message: msg });
        } else if (msg.toLowerCase().includes('failed') || msg.toLowerCase().includes('error')) {
          setLastResult({ type: 'error', message: msg });
        } else {
          setLastResult({ type: 'success', message: msg });
        }
        fetchExperiments();
        fetchSpadeExperiments();
      }
      wasRunning.current = newStatus.is_running;
      setStatus(newStatus);
    } catch (err) {
      console.error("Failed to fetch generation status:", err);
    }
  };

  const fetchPrepStatus = async () => {
    try {
      const response = await api.get('/api/prepare/status');
      setPrepStatus(response.data);
    } catch (err) {
      console.error("Failed to fetch preparation status:", err);
    }
  };

  const fetchDatasetCheck = async () => {
    try {
      const response = await api.get('/api/prepare/check');
      setDatasetCheck(response.data);
    } catch (err) {
      console.error("Failed to check dataset status:", err);
    }
  };

  const handleStartPreparation = async () => {
    setPrepError(null);
    try {
      await api.post('/api/prepare');
      fetchPrepStatus();
    } catch (err: any) {
      setPrepError(err.response?.data?.detail || "Failed to start dataset preparation.");
    }
  };

  const fetchExperiments = async () => {
    try {
      const response = await api.get('/api/generate/cyclegan-experiments');
      const exps = response.data.experiments;
      setAvailableExperiments(exps);
      const expKeys = Object.keys(exps);
      if (expKeys.length > 0 && (!experimentName || !exps[experimentName])) {
        setExperimentName(expKeys[0]);
        if (exps[expKeys[0]] && exps[expKeys[0]].length > 0) {
            setEpoch(exps[expKeys[0]][0]);
        }
      }
    } catch (err) {
      console.error("Failed to fetch experiments:", err);
    }
  };

  const fetchSpadeExperiments = async () => {
    try {
      const response = await api.get('/api/generate/spade-experiments');
      const exps = response.data.experiments;
      setAvailableSpadeExperiments(exps);
      const expKeys = Object.keys(exps);
      if (expKeys.length > 0 && (!spadeExperimentName || !exps[spadeExperimentName])) {
        setSpadeExperimentName(expKeys[0]);
        if (exps[expKeys[0]] && exps[expKeys[0]].length > 0) {
            setSpadeEpoch(exps[expKeys[0]][0]);
        }
      }
    } catch (err) {
      console.error("Failed to fetch SPADE experiments:", err);
    }
  };

  const fetchGallery = async (exp: string, ep: string, type: 'cyclegan' | 'spade' = galleryType) => {
    setGalleryLoading(true);
    setError(null);
    const endpoint = type === 'spade' ? '/api/generate/spade-results' : '/api/generate/results';
    try {
      const response = await api.get(`${endpoint}?experiment=${exp}&epoch=${ep}`);
      setGalleryImages(response.data.images);
      setGalleryExp(response.data.experiment || exp);
      setGalleryEpoch(response.data.test_dir || ep);
    } catch (err) {
      console.error("Failed to fetch gallery:", err);
      setError("Failed to load results gallery.");
    } finally {
      setGalleryLoading(false);
    }
  };

  useEffect(() => {
    fetchStatus();
    fetchPrepStatus();
    fetchDatasetCheck();
    fetchExperiments();
    fetchSpadeExperiments();
    const rate = status.is_running || prepStatus.is_running ? 1000 : 3000;
    const interval = setInterval(() => {
      fetchStatus();
      fetchPrepStatus();
      if (!prepStatus.is_running) fetchDatasetCheck();
    }, rate);
    return () => clearInterval(interval);
  }, [status.is_running, prepStatus.is_running]);

  useEffect(() => {
     if (experimentName && availableExperiments[experimentName] && availableExperiments[experimentName].length > 0) {
         setEpoch(availableExperiments[experimentName][0]);
     }
  }, [experimentName, availableExperiments]);

  useEffect(() => {
     if (spadeExperimentName && availableSpadeExperiments[spadeExperimentName] && availableSpadeExperiments[spadeExperimentName].length > 0) {
         setSpadeEpoch(availableSpadeExperiments[spadeExperimentName][0]);
     }
  }, [spadeExperimentName, availableSpadeExperiments]);

  // Auto-fetch the gallery when the sidebar deep-links into a results
  // view, OR when the relevant experiment/epoch becomes available after
  // the initial fetchExperiments() round-trip. Also refires when the user
  // changes the dropdowns inside the gallery itself (those handlers just
  // update the experiment/epoch state — this effect does the actual GET).
  useEffect(() => {
    if (view === 'tasks') return;
    const isCyclegan = view === 'cyclegan-results';
    const exp = isCyclegan ? experimentName : spadeExperimentName;
    const ep = isCyclegan ? epoch : spadeEpoch;
    const type = isCyclegan ? 'cyclegan' : 'spade';
    if (exp) {
      fetchGallery(exp, ep, type);
    }
    // fetchGallery isn't memoised and its deps would just be the same
    // state we're already watching here — safe to omit from the array.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [view, experimentName, epoch, spadeExperimentName, spadeEpoch]);

  const handleCancelGeneration = async () => {
    try {
      await api.post('/api/generate/cancel');
    } catch (err: any) {
      console.error(err);
      setError(err.response?.data?.detail || "Failed to cancel generation.");
    }
  };

  const handleStartGeneration = async () => {
    setError(null);
    try {
      await api.post('/api/generate', {
        task_type: taskType,
        experiment_name: taskType === 'test_cyclegan' ? experimentName : taskType === 'test_spade' ? spadeExperimentName : undefined,
        epoch: taskType === 'test_cyclegan' ? epoch : taskType === 'test_spade' ? spadeEpoch : undefined,
        ...(taskType === 'train_cyclegan' ? {
          batch_size: cganBatchSize,
          n_epochs: cganEpochs,
          lr: cganLr,
          netG: cganNetG,
          load_size: cganLoadSize,
          crop_size: cganCropSize,
          ...(cganMaxDataset ? { max_dataset_size: cganMaxDataset } : {}),
        } : {}),
        ...(taskType === 'train_spade' ? {
          spade_batch_size: spadeBatchSize,
          spade_niter: spadeNiter,
          spade_niter_decay: spadeNiterDecay,
          spade_lr: spadeLr,
          spade_netG: spadeNetG,
          spade_load_size: spadeLoadSize,
          spade_crop_size: spadeCropSize,
          ...(spadeMaxDataset ? { spade_max_dataset_size: spadeMaxDataset } : {}),
        } : {}),
      });
      fetchStatus();
    } catch (err: any) {
      console.error(err);
      setError(err.response?.data?.detail || "An error occurred while starting the generative task.");
    }
  };

  return (
    <div className="max-w-7xl mx-auto flex flex-col gap-6 pt-2">
      {/* Header — title only. The "View results" entry points used to
          live here as buttons in the top-right, but that placement
          confused users (they look like header chrome, not navigation).
          They're now first-class entries in the sidebar under
          "Data & Exploration" → CycleGAN Results / SPADE Results. */}
      <div>
        <h2 className="text-2xl font-bold mb-2 flex items-center gap-2">
          <Sparkles className="w-6 h-6 text-purple-400" />
          {showGallery
            ? `${galleryType === 'spade' ? 'SPADE' : 'CycleGAN'} — Generated results`
            : 'Generative Data Augmentation'}
        </h2>
        <p className="text-gray-400">
          {showGallery
            ? `Browse the synthetic images produced by the most recent ${galleryType === 'spade' ? 'SPADE' : 'CycleGAN'} test run.`
            : 'Create synthetic training data using generative AI models to improve polyp detection performance.'}
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

      {/* Dataset Preparation Section */}
      {(prepStatus.is_running || prepStatus.message !== 'Idle') && (
        <div className={`rounded-xl border p-5 ${
          prepStatus.is_running
            ? 'bg-amber-900/20 border-amber-800'
            : prepStatus.message.startsWith('Error')
            ? 'bg-red-900/20 border-red-800'
            : 'bg-green-900/20 border-green-800'
        }`}>
          <div className="flex items-center gap-3">
            {prepStatus.is_running ? (
              <RefreshCw className="w-5 h-5 text-amber-400 animate-spin flex-shrink-0" />
            ) : prepStatus.message.startsWith('Error') ? (
              <AlertCircle className="w-5 h-5 text-red-400 flex-shrink-0" />
            ) : (
              <CheckCircle className="w-5 h-5 text-green-400 flex-shrink-0" />
            )}
            <span className={`text-sm font-medium ${
              prepStatus.is_running ? 'text-amber-300'
                : prepStatus.message.startsWith('Error') ? 'text-red-300'
                : 'text-green-300'
            }`}>
              {prepStatus.message}
            </span>
          </div>
        </div>
      )}

      {!showGallery && !status.is_running && (
        <PrepareDatasetCard
          prepStatus={prepStatus}
          prepError={prepError}
          datasetCheck={datasetCheck}
          onStart={handleStartPreparation}
        />
      )}

      {showGallery ? (() => {
        const exps = galleryType === 'spade' ? availableSpadeExperiments : availableExperiments;
        const curExp = galleryType === 'spade' ? spadeExperimentName : experimentName;
        const setCurExp = galleryType === 'spade' ? setSpadeExperimentName : setExperimentName;
        const curEpoch = galleryType === 'spade' ? spadeEpoch : epoch;
        const setCurEpoch = galleryType === 'spade' ? setSpadeEpoch : setEpoch;
        const expKeys = Object.keys(exps);
        const resolvedExp = exps[curExp] ? curExp : expKeys[0] || '';
        const epochList = exps[resolvedExp] || [];

        return (
        <div className="flex flex-col gap-6 animate-in fade-in slide-in-from-bottom-4 duration-500">
           <div className="flex items-center justify-between bg-gray-800 border border-gray-700 p-4 rounded-xl gap-4">
             <div className="flex items-center gap-4 flex-1 min-w-0">
               <div className="min-w-0">
                 <label className="text-xs text-gray-400 block uppercase tracking-wider mb-1">Model</label>
                 <div className="flex gap-1">
                   {/* In-gallery model toggle. Routes through setView so
                       the sidebar's active highlight stays in sync. The
                       auto-fetch useEffect above re-fires when the view
                       changes — no explicit fetchGallery() needed here. */}
                   <button onClick={() => setView('cyclegan-results')}
                     className={`px-3 py-1.5 rounded text-xs font-medium transition-colors ${galleryType === 'cyclegan' ? 'bg-purple-600 text-white' : 'bg-gray-700 text-gray-400 hover:text-white'}`}>
                     CycleGAN
                   </button>
                   <button onClick={() => setView('spade-results')}
                     className={`px-3 py-1.5 rounded text-xs font-medium transition-colors ${galleryType === 'spade' ? 'bg-purple-600 text-white' : 'bg-gray-700 text-gray-400 hover:text-white'}`}>
                     SPADE
                   </button>
                 </div>
               </div>
               <div className="w-px h-10 bg-gray-700 flex-shrink-0"></div>
               <div className="min-w-0 flex-1">
                 <label className="text-xs text-gray-400 block uppercase tracking-wider mb-1">Experiment</label>
                 {expKeys.length > 0 ? (
                   <select
                     value={resolvedExp}
                     onChange={(e) => {
                       setCurExp(e.target.value);
                       const epochs = exps[e.target.value];
                       const ep = epochs?.[0] || 'latest';
                       setCurEpoch(ep);
                       // The auto-fetch useEffect picks up the new
                       // experiment/epoch state and refreshes the gallery.
                     }}
                     className="w-full bg-gray-900 border border-gray-600 rounded px-3 py-1.5 text-sm text-purple-300 font-mono font-semibold outline-none focus:border-purple-500"
                   >
                     {expKeys.map(exp => (
                       <option key={exp} value={exp}>{exp}</option>
                     ))}
                   </select>
                 ) : (
                   <span className="font-mono text-purple-300 font-semibold text-sm">{galleryExp || 'None'}</span>
                 )}
               </div>
               <div className="w-px h-10 bg-gray-700 flex-shrink-0"></div>
               <div className="min-w-0">
                 <label className="text-xs text-gray-400 block uppercase tracking-wider mb-1">Epoch / Test</label>
                 {epochList.length > 0 ? (
                   <select
                     value={epochList.includes(curEpoch) ? curEpoch : epochList[0]}
                     onChange={(e) => {
                       setCurEpoch(e.target.value);
                     }}
                     className="w-full bg-gray-900 border border-gray-600 rounded px-3 py-1.5 text-sm text-purple-300 font-mono outline-none focus:border-purple-500"
                   >
                     {epochList.map(ep => (
                       <option key={ep} value={ep}>{ep}</option>
                     ))}
                   </select>
                 ) : (
                   <span className="font-mono text-purple-300 text-sm">{galleryEpoch || 'None'}</span>
                 )}
               </div>
             </div>
             
             <div className="flex gap-2 flex-shrink-0">
               <button 
                 onClick={() => fetchGallery(resolvedExp, epochList.includes(curEpoch) ? curEpoch : epochList[0] || 'latest', galleryType)}
                 className="p-2 bg-gray-700 hover:bg-gray-600 rounded border border-gray-600 text-gray-300"
                 title="Refresh"
               >
                 <RefreshCw className={`w-4 h-4 ${galleryLoading ? 'animate-spin' : ''}`} />
               </button>
               <button 
                 onClick={() => setView('tasks')}
                 className="px-4 py-2 bg-purple-600 hover:bg-purple-500 text-white rounded font-medium text-sm transition-colors"
               >
                 Back to Tasks
               </button>
             </div>
           </div>

           {galleryLoading ? (
             <div className="flex flex-col items-center justify-center py-20 text-gray-400">
               <RefreshCw className="w-8 h-8 animate-spin mb-4 text-purple-500" />
               <p>Loading generated images...</p>
             </div>
           ) : galleryImages.length === 0 ? (
             <div className="flex flex-col items-center justify-center py-20 text-gray-400 bg-gray-800/50 border border-gray-700 rounded-xl border-dashed">
               <ImageIcon className="w-12 h-12 mb-4 text-gray-600" />
               <p className="text-lg">No results found.</p>
               <p className="text-sm mt-2">Run a "Test" job to generate some synthetic images first!</p>
             </div>
           ) : (
             <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
               {galleryImages.map((img, idx) => (
                 <div key={idx} className="bg-gray-800 border border-gray-700 rounded-xl overflow-hidden shadow-lg group">
                   <div className="p-3 border-b border-gray-700 bg-gray-800/80 flex justify-between items-center">
                     <span className="text-xs font-mono text-gray-400 truncate max-w-[200px]" title={img.base_name}>{img.base_name}</span>
                   </div>
                   <div className="grid grid-cols-2 gap-px bg-gray-700">
                     <div className="bg-gray-900 aspect-square relative group/img">
                       {img.real ? (
                         <img src={img.real} alt="Real Mask" className="w-full h-full object-cover" />
                       ) : (
                         <div className="w-full h-full flex items-center justify-center text-gray-600 text-xs">No Real</div>
                       )}
                       <div className="absolute bottom-0 left-0 right-0 bg-black/60 text-[10px] text-center py-1 opacity-0 group-hover/img:opacity-100 transition-opacity">Input Mask</div>
                     </div>
                     <div className="bg-gray-900 aspect-square relative group/img">
                       {img.fake ? (
                         <img src={img.fake} alt="Generated Polyp" className="w-full h-full object-cover" />
                       ) : (
                         <div className="w-full h-full flex items-center justify-center text-gray-600 text-xs">No Fake</div>
                       )}
                       <div className="absolute bottom-0 left-0 right-0 bg-purple-900/80 text-[10px] text-purple-100 text-center py-1 opacity-0 group-hover/img:opacity-100 transition-opacity">Generated Polyp</div>
                     </div>
                   </div>
                 </div>
               ))}
             </div>
           )}
        </div>
        );
      })() : status.is_running ? (
        <div className="bg-purple-900/20 border border-purple-800 rounded-xl p-10 flex flex-col items-center justify-center text-center mt-4">
          <div className="relative mb-6">
            <div className="w-20 h-20 border-4 border-purple-500/30 border-t-purple-500 rounded-full animate-spin"></div>
            <div className="absolute inset-0 flex items-center justify-center">
              <RefreshCw className="w-8 h-8 text-purple-400 animate-pulse" />
            </div>
          </div>
          
          <h3 className="text-2xl font-medium text-purple-300 mb-3">
            Running Task: <span className="font-bold text-white uppercase tracking-wider">{status.current_task?.replace('_', ' ')}</span>
          </h3>
          
          <div className="bg-gray-900/80 border border-gray-700 p-4 rounded-lg w-full max-w-2xl mt-2 text-left max-h-80 overflow-y-auto log-scroll">
            <p className="text-sm font-mono text-gray-300 overflow-x-auto whitespace-pre-wrap">
              {status.message}
            </p>
          </div>
          
          <div className="flex items-center gap-4 mt-6">
            <p className="text-sm text-yellow-400 bg-yellow-900/20 px-4 py-2 rounded-full border border-yellow-700/30">
              Runs in background. You can navigate away.
            </p>
            <button
              onClick={handleCancelGeneration}
              className="flex items-center gap-1.5 px-4 py-2 bg-red-600 hover:bg-red-500 text-white text-sm rounded-full transition-colors"
            >
              <Square className="w-3.5 h-3.5" />
              Cancel
            </button>
          </div>
        </div>
      ) : (
        <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mt-4">
          
          <div className="md:col-span-2 space-y-6">
            <div className="bg-gray-800 border border-gray-700 rounded-xl overflow-hidden shadow-lg">
              <div className="p-6 border-b border-gray-700 bg-gray-800/50">
                <h3 className="text-lg font-medium text-gray-200">Select Task</h3>
              </div>
              
              <div className="p-6 space-y-4">
                <label className={`relative flex cursor-pointer rounded-lg border p-4 shadow-sm focus:outline-none ${taskType === 'train_cyclegan' ? 'border-purple-500 bg-purple-900/10' : 'border-gray-700 bg-gray-800 hover:bg-gray-700'}`}>
                  <input type="radio" name="task" value="train_cyclegan" className="sr-only" checked={taskType === 'train_cyclegan'} onChange={(e) => setTaskType(e.target.value)} />
                  <span className="flex flex-1">
                    <span className="flex flex-col w-full">
                      <span className="block text-sm font-medium text-gray-100">Train CycleGAN</span>
                      <span className="mt-1 flex items-center text-sm text-gray-400">Train an image-to-image translation model to generate realistic polyp images from masks.</span>

                      {taskType === 'train_cyclegan' && (
                        <div className="mt-4 border-t border-gray-700 pt-4 grid grid-cols-2 gap-3">
                          <div>
                            <label className="block text-xs font-medium text-gray-400 mb-1">Batch Size</label>
                            <input type="number" min="1" value={cganBatchSize} onChange={(e) => setCganBatchSize(parseInt(e.target.value) || 1)}
                              className="w-full bg-gray-900 border border-gray-600 rounded px-3 py-1.5 text-sm text-white outline-none focus:border-purple-500" />
                          </div>
                          <div>
                            <label className="block text-xs font-medium text-gray-400 mb-1">Epochs</label>
                            <input type="number" min="1" value={cganEpochs} onChange={(e) => setCganEpochs(parseInt(e.target.value) || 1)}
                              className="w-full bg-gray-900 border border-gray-600 rounded px-3 py-1.5 text-sm text-white outline-none focus:border-purple-500" />
                          </div>
                          <div>
                            <label className="block text-xs font-medium text-gray-400 mb-1">Learning Rate</label>
                            <input type="number" step="0.0001" min="0.00001" value={cganLr} onChange={(e) => setCganLr(parseFloat(e.target.value) || 0.0001)}
                              className="w-full bg-gray-900 border border-gray-600 rounded px-3 py-1.5 text-sm text-white outline-none focus:border-purple-500" />
                          </div>
                          <div>
                            <label className="block text-xs font-medium text-gray-400 mb-1">Generator</label>
                            <select value={cganNetG} onChange={(e) => setCganNetG(e.target.value)}
                              className="w-full bg-gray-900 border border-gray-600 rounded px-3 py-1.5 text-sm text-white outline-none focus:border-purple-500">
                              <option value="resnet_9blocks">ResNet 9 blocks</option>
                              <option value="resnet_6blocks">ResNet 6 blocks</option>
                              <option value="unet_256">U-Net 256</option>
                              <option value="unet_128">U-Net 128</option>
                            </select>
                          </div>
                          <div>
                            <label className="block text-xs font-medium text-gray-400 mb-1">Load Size</label>
                            <input type="number" min="64" value={cganLoadSize} onChange={(e) => setCganLoadSize(parseInt(e.target.value) || 64)}
                              className="w-full bg-gray-900 border border-gray-600 rounded px-3 py-1.5 text-sm text-white outline-none focus:border-purple-500" />
                          </div>
                          <div>
                            <label className="block text-xs font-medium text-gray-400 mb-1">Crop Size</label>
                            <input type="number" min="64" value={cganCropSize} onChange={(e) => setCganCropSize(parseInt(e.target.value) || 64)}
                              className="w-full bg-gray-900 border border-gray-600 rounded px-3 py-1.5 text-sm text-white outline-none focus:border-purple-500" />
                          </div>
                          <div className="col-span-2">
                            <label className="block text-xs font-medium text-gray-400 mb-1">
                              Max Dataset Size <span className="text-gray-500 font-normal">(optional — for quick testing)</span>
                            </label>
                            <input type="number" min="1" placeholder="All" value={cganMaxDataset}
                              onChange={(e) => setCganMaxDataset(e.target.value ? parseInt(e.target.value) : '')}
                              className="w-full bg-gray-900 border border-gray-600 rounded px-3 py-1.5 text-sm text-white outline-none focus:border-purple-500 placeholder-gray-500" />
                          </div>
                        </div>
                      )}
                    </span>
                  </span>
                  {taskType === 'train_cyclegan' && <CheckCircle className="h-5 w-5 text-purple-500 absolute right-4 top-4" />}
                </label>

                <label className={`relative flex cursor-pointer rounded-lg border p-4 shadow-sm focus:outline-none ${taskType === 'test_cyclegan' ? 'border-purple-500 bg-purple-900/10' : 'border-gray-700 bg-gray-800 hover:bg-gray-700'}`}>
                  <input type="radio" name="task" value="test_cyclegan" className="sr-only" checked={taskType === 'test_cyclegan'} onChange={(e) => setTaskType(e.target.value)} />
                  <span className="flex flex-1">
                    <span className="flex flex-col w-full">
                      <span className="block text-sm font-medium text-gray-100">Test CycleGAN (Generate Images)</span>
                      <span className="mt-1 flex items-center text-sm text-gray-400">Run inference using a trained CycleGAN model to generate the augmented dataset.</span>
                      
                      {taskType === 'test_cyclegan' && (() => {
                        const expKeys = Object.keys(availableExperiments);
                        const resolvedExp = availableExperiments[experimentName] ? experimentName : expKeys[0] || '';
                        const epochList = availableExperiments[resolvedExp] || [];
                        return (
                        <div className="mt-4 grid grid-cols-2 gap-4 border-t border-gray-700 pt-4">
                          <div>
                            <label className="block text-xs font-medium text-gray-400 mb-1">Experiment Name</label>
                            {expKeys.length > 0 ? (
                              <select 
                                value={resolvedExp}
                                onChange={(e) => setExperimentName(e.target.value)}
                                className="w-full bg-gray-900 border border-gray-600 rounded px-3 py-1.5 text-sm text-white outline-none focus:border-purple-500"
                              >
                                {expKeys.map(exp => (
                                  <option key={exp} value={exp}>{exp}</option>
                                ))}
                              </select>
                            ) : (
                              <input 
                                type="text" 
                                value={experimentName}
                                onChange={(e) => setExperimentName(e.target.value)}
                                className="w-full bg-gray-900 border border-gray-600 rounded px-3 py-1.5 text-sm text-white outline-none focus:border-purple-500"
                                placeholder="No experiments found..."
                              />
                            )}
                          </div>
                          <div>
                            <label className="block text-xs font-medium text-gray-400 mb-1">Epoch Checkpoint</label>
                            {epochList.length > 0 ? (
                              <select 
                                value={epochList.includes(epoch) ? epoch : epochList[0]}
                                onChange={(e) => setEpoch(e.target.value)}
                                className="w-full bg-gray-900 border border-gray-600 rounded px-3 py-1.5 text-sm text-white outline-none focus:border-purple-500"
                              >
                                {epochList.map(ep => (
                                  <option key={ep} value={ep}>{ep}</option>
                                ))}
                              </select>
                            ) : (
                              <input 
                                type="text" 
                                value={epoch}
                                onChange={(e) => setEpoch(e.target.value)}
                                className="w-full bg-gray-900 border border-gray-600 rounded px-3 py-1.5 text-sm text-white outline-none focus:border-purple-500"
                                placeholder="latest"
                              />
                            )}
                          </div>
                        </div>
                        );
                      })()}
                    </span>
                  </span>
                  {taskType === 'test_cyclegan' && <CheckCircle className="h-5 w-5 text-purple-500 absolute right-4 top-4" />}
                </label>

                <label className={`relative flex cursor-pointer rounded-lg border p-4 shadow-sm focus:outline-none ${taskType === 'train_spade' ? 'border-purple-500 bg-purple-900/10' : 'border-gray-700 bg-gray-800 hover:bg-gray-700'}`}>
                  <input type="radio" name="task" value="train_spade" className="sr-only" checked={taskType === 'train_spade'} onChange={(e) => setTaskType(e.target.value)} />
                  <span className="flex flex-1">
                    <span className="flex flex-col w-full">
                      <span className="block text-sm font-medium text-gray-100">Train SPADE</span>
                      <span className="mt-1 flex items-center text-sm text-gray-400">Train a Spatially-Adaptive Normalization (SPADE) model for high-fidelity image synthesis from semantic layouts.</span>

                      {taskType === 'train_spade' && (
                        <div className="mt-4 border-t border-gray-700 pt-4 grid grid-cols-2 gap-3">
                          <div>
                            <label className="block text-xs font-medium text-gray-400 mb-1">Batch Size</label>
                            <input type="number" min="1" value={spadeBatchSize} onChange={(e) => setSpadeBatchSize(parseInt(e.target.value) || 1)}
                              className="w-full bg-gray-900 border border-gray-600 rounded px-3 py-1.5 text-sm text-white outline-none focus:border-purple-500" />
                          </div>
                          <div>
                            <label className="block text-xs font-medium text-gray-400 mb-1">Learning Rate</label>
                            <input type="number" step="0.0001" min="0.00001" value={spadeLr} onChange={(e) => setSpadeLr(parseFloat(e.target.value) || 0.0001)}
                              className="w-full bg-gray-900 border border-gray-600 rounded px-3 py-1.5 text-sm text-white outline-none focus:border-purple-500" />
                          </div>
                          <div>
                            <label className="block text-xs font-medium text-gray-400 mb-1">Epochs (niter)</label>
                            <input type="number" min="1" value={spadeNiter} onChange={(e) => setSpadeNiter(parseInt(e.target.value) || 1)}
                              className="w-full bg-gray-900 border border-gray-600 rounded px-3 py-1.5 text-sm text-white outline-none focus:border-purple-500" />
                            <p className="text-[10px] text-gray-500 mt-0.5">Epochs at initial LR</p>
                          </div>
                          <div>
                            <label className="block text-xs font-medium text-gray-400 mb-1">Decay Epochs</label>
                            <input type="number" min="0" value={spadeNiterDecay} onChange={(e) => setSpadeNiterDecay(parseInt(e.target.value) || 0)}
                              className="w-full bg-gray-900 border border-gray-600 rounded px-3 py-1.5 text-sm text-white outline-none focus:border-purple-500" />
                            <p className="text-[10px] text-gray-500 mt-0.5">Epochs to decay LR to 0</p>
                          </div>
                          <div>
                            <label className="block text-xs font-medium text-gray-400 mb-1">Generator</label>
                            <select value={spadeNetG} onChange={(e) => setSpadeNetG(e.target.value)}
                              className="w-full bg-gray-900 border border-gray-600 rounded px-3 py-1.5 text-sm text-white outline-none focus:border-purple-500">
                              <option value="spade">SPADE</option>
                              <option value="pix2pixhd">Pix2PixHD</option>
                            </select>
                          </div>
                          <div>
                            <label className="block text-xs font-medium text-gray-400 mb-1">Load Size</label>
                            <input type="number" min="64" value={spadeLoadSize} onChange={(e) => setSpadeLoadSize(parseInt(e.target.value) || 64)}
                              className="w-full bg-gray-900 border border-gray-600 rounded px-3 py-1.5 text-sm text-white outline-none focus:border-purple-500" />
                          </div>
                          <div className="col-span-2 grid grid-cols-2 gap-3">
                            <div>
                              <label className="block text-xs font-medium text-gray-400 mb-1">Crop Size</label>
                              <input type="number" min="64" value={spadeCropSize} onChange={(e) => setSpadeCropSize(parseInt(e.target.value) || 64)}
                                className="w-full bg-gray-900 border border-gray-600 rounded px-3 py-1.5 text-sm text-white outline-none focus:border-purple-500" />
                            </div>
                            <div>
                              <label className="block text-xs font-medium text-gray-400 mb-1">
                                Max Dataset Size <span className="text-gray-500">(optional)</span>
                              </label>
                              <input type="number" min="1" placeholder="All" value={spadeMaxDataset}
                                onChange={(e) => setSpadeMaxDataset(e.target.value ? parseInt(e.target.value) : '')}
                                className="w-full bg-gray-900 border border-gray-600 rounded px-3 py-1.5 text-sm text-white outline-none focus:border-purple-500 placeholder-gray-500" />
                            </div>
                          </div>
                        </div>
                      )}
                    </span>
                  </span>
                  {taskType === 'train_spade' && <CheckCircle className="h-5 w-5 text-purple-500 absolute right-4 top-4" />}
                </label>

                <label className={`relative flex cursor-pointer rounded-lg border p-4 shadow-sm focus:outline-none ${taskType === 'test_spade' ? 'border-purple-500 bg-purple-900/10' : 'border-gray-700 bg-gray-800 hover:bg-gray-700'}`}>
                  <input type="radio" name="task" value="test_spade" className="sr-only" checked={taskType === 'test_spade'} onChange={(e) => setTaskType(e.target.value)} />
                  <span className="flex flex-1">
                    <span className="flex flex-col w-full">
                      <span className="block text-sm font-medium text-gray-100">Test SPADE (Generate Images)</span>
                      <span className="mt-1 flex items-center text-sm text-gray-400">Run inference using a trained SPADE model to generate high-fidelity images from semantic layouts.</span>

                      {taskType === 'test_spade' && (() => {
                        const spadeExpKeys = Object.keys(availableSpadeExperiments);
                        const resolvedSpadeExp = availableSpadeExperiments[spadeExperimentName] ? spadeExperimentName : spadeExpKeys[0] || '';
                        const spadeEpochList = availableSpadeExperiments[resolvedSpadeExp] || [];
                        return (
                        <div className="mt-4 grid grid-cols-2 gap-4 border-t border-gray-700 pt-4">
                          <div>
                            <label className="block text-xs font-medium text-gray-400 mb-1">Experiment Name</label>
                            {spadeExpKeys.length > 0 ? (
                              <select
                                value={resolvedSpadeExp}
                                onChange={(e) => setSpadeExperimentName(e.target.value)}
                                className="w-full bg-gray-900 border border-gray-600 rounded px-3 py-1.5 text-sm text-white outline-none focus:border-purple-500"
                              >
                                {spadeExpKeys.map(exp => (
                                  <option key={exp} value={exp}>{exp}</option>
                                ))}
                              </select>
                            ) : (
                              <input
                                type="text"
                                value={spadeExperimentName}
                                onChange={(e) => setSpadeExperimentName(e.target.value)}
                                className="w-full bg-gray-900 border border-gray-600 rounded px-3 py-1.5 text-sm text-white outline-none focus:border-purple-500"
                                placeholder="No experiments found..."
                              />
                            )}
                          </div>
                          <div>
                            <label className="block text-xs font-medium text-gray-400 mb-1">Epoch Checkpoint</label>
                            {spadeEpochList.length > 0 ? (
                              <select
                                value={spadeEpochList.includes(spadeEpoch) ? spadeEpoch : spadeEpochList[0]}
                                onChange={(e) => setSpadeEpoch(e.target.value)}
                                className="w-full bg-gray-900 border border-gray-600 rounded px-3 py-1.5 text-sm text-white outline-none focus:border-purple-500"
                              >
                                {spadeEpochList.map(ep => (
                                  <option key={ep} value={ep}>{ep}</option>
                                ))}
                              </select>
                            ) : (
                              <input
                                type="text"
                                value={spadeEpoch}
                                onChange={(e) => setSpadeEpoch(e.target.value)}
                                className="w-full bg-gray-900 border border-gray-600 rounded px-3 py-1.5 text-sm text-white outline-none focus:border-purple-500"
                                placeholder="latest"
                              />
                            )}
                          </div>
                        </div>
                        );
                      })()}
                    </span>
                  </span>
                  {taskType === 'test_spade' && <CheckCircle className="h-5 w-5 text-purple-500 absolute right-4 top-4" />}
                </label>
              </div>
            </div>
            
            <button
              onClick={handleStartGeneration}
              className="w-full flex items-center justify-center gap-2 bg-gradient-to-r from-purple-600 to-blue-600 hover:from-purple-500 hover:to-blue-500 text-white px-8 py-4 rounded-xl font-medium transition-all hover:scale-[1.02] active:scale-[0.98] shadow-lg shadow-purple-900/30 text-lg"
            >
              <Play className="w-5 h-5 fill-current" />
              Start Selected Job
            </button>
          </div>

          <div className="md:col-span-1 space-y-4">
            <div className="bg-gray-800 border border-gray-700 rounded-xl p-5 shadow-lg">
              <h4 className="font-medium text-gray-200 mb-3 pb-2 border-b border-gray-700">Status Output</h4>
              
              {error ? (
                <div className="flex items-start gap-2 text-sm p-3 rounded bg-red-900/30 border border-red-800 text-red-300">
                  <AlertCircle className="w-5 h-5 flex-shrink-0 mt-0.5" />
                  <span className="font-mono overflow-x-auto whitespace-pre-wrap">{error}</span>
                </div>
              ) : status.message !== 'Idle' ? (
                <div className={`flex items-start gap-2 text-sm p-3 rounded border w-full whitespace-pre-wrap ${
                  status.message.toLowerCase().includes('failed') || status.message.toLowerCase().includes('error')
                    ? 'bg-red-900/30 border-red-800 text-red-300'
                    : 'bg-green-900/30 border-green-800 text-green-300'
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
            
            <div className="bg-blue-900/20 border border-blue-800/50 rounded-xl p-5 text-sm text-blue-200 shadow-lg">
               <h4 className="font-medium text-blue-300 mb-2 flex items-center gap-1">
                 <Sparkles className="w-4 h-4" /> Why Generative Data?
               </h4>
               <p>
                 Medical datasets for polyps are often small. By training models like CycleGAN or SPADE, we can synthetically generate thousands of new, realistic polyp images from simple shape masks, dramatically improving the robustness of our detection models.
               </p>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

interface PrepareDatasetCardProps {
  prepStatus: PrepStatus;
  prepError: string | null;
  datasetCheck: {
    datasets: Record<string, Record<string, number>>;
    source_available: boolean;
    ready: boolean;
  } | null;
  onStart: () => void;
}

// The "Prepare Dataset" step is the most opaque thing on this page if you
// haven't read the CycleGAN repo's README — the button just says "Prepare"
// and the trainA/trainB/testA/testB folder names are pure jargon. This
// card breaks that down: it explains in plain English that we (1) turn
// bounding-box .txt files into binary mask PNGs and (2) restage everything
// into the layout the trainer scripts expect, with a small visual diagram.
//
// Two display modes:
//
// - **Not ready / source-missing / running**: full explanatory card. Users
//   land here once and need the diagram + readiness panel to know what's
//   about to happen (and why the train buttons below are disabled).
// - **Ready**: collapses to a one-line status pill. Once data is staged the
//   step becomes background context — the user wants the screen real
//   estate for picking experiments and looking at galleries below. The
//   pill keeps the Re-run button reachable and an expand chevron for
//   users who want to revisit the diagram (e.g. after a regen failure).
function PrepareDatasetCard({
  prepStatus,
  prepError,
  datasetCheck,
  onStart,
}: PrepareDatasetCardProps) {
  const isRunning = prepStatus.is_running;
  const isReady = !!datasetCheck?.ready;
  const sourceMissing = datasetCheck && !datasetCheck.source_available;

  // CycleGAN convention: domain A = "input" (here, masks), domain B =
  // "target" (here, real polyp images). The `prepare_dataset` step copies
  // one mask per image, so trainA == trainB and testA == testB whenever
  // prep finished successfully — collapsing them into "pairs" is more
  // honest than printing both numbers.
  const polyp = datasetCheck?.datasets['PolypDataset'] || {};
  const spade = datasetCheck?.datasets['PolypDatasetSPADE'] || {};
  const trainPairs = Math.min(polyp.trainA || 0, polyp.trainB || 0);
  const testPairs = Math.min(polyp.testA || 0, polyp.testB || 0);
  const polypMismatch =
    polyp.trainA !== polyp.trainB || polyp.testA !== polyp.testB;
  const spadeMismatch =
    spade.trainA !== spade.trainB || spade.testA !== spade.testB;

  // Manual "show me the diagram again" toggle. Only meaningful in the
  // collapsible state (ready + idle). Defaults closed: once a user has
  // staged data they shouldn't have to keep looking at the explanation
  // every time they revisit this tab.
  const [expanded, setExpanded] = useState(false);

  // Collapsible iff the step is genuinely "done" and quiet. While running
  // we always show the full card so the spinner button + status sit in a
  // recognisable place. While source is missing or unstaged, the full
  // card is what helps the user understand what to do next.
  const collapsible = isReady && !isRunning;
  const showFull = !collapsible || expanded;

  // ---- Collapsed pill (ready + not expanded) ----------------------------
  if (!showFull) {
    return (
      <div className="rounded-xl border border-emerald-800/50 bg-gradient-to-r from-emerald-900/15 to-gray-800/40 px-4 py-2.5 flex items-center justify-between gap-3 shadow-sm">
        <button
          onClick={() => setExpanded(true)}
          className="flex items-center gap-2.5 min-w-0 flex-1 text-left hover:opacity-80 transition-opacity"
          title="Show the prep step details and folder layout diagram"
        >
          <CheckCircle className="w-4 h-4 text-emerald-400 flex-shrink-0" />
          <span className="text-sm font-medium text-emerald-200 flex-shrink-0">
            Data staged for the generators
          </span>
          <span className="text-xs text-gray-400 truncate">
            {trainPairs.toLocaleString()} train · {testPairs.toLocaleString()} test pairs
            {(polypMismatch || spadeMismatch) && (
              <span className="text-amber-400 ml-2">⚠ mismatch — re-run prep</span>
            )}
          </span>
        </button>
        <div className="flex items-center gap-1 flex-shrink-0">
          <button
            onClick={onStart}
            disabled={isRunning}
            title="Re-run prep to refresh masks (idempotent)"
            className="flex items-center gap-1.5 text-xs text-gray-300 hover:text-white hover:bg-gray-700/60 px-2.5 py-1 rounded-md transition-colors"
          >
            <RefreshCw className="w-3.5 h-3.5" /> Re-run prep
          </button>
          <button
            onClick={() => setExpanded(true)}
            title="Show details"
            aria-label="Show prep step details"
            className="p-1 text-gray-500 hover:text-gray-200 transition-colors"
          >
            <ChevronDown className="w-4 h-4" />
          </button>
        </div>
      </div>
    );
  }

  const accent = isReady
    ? 'border-emerald-800/40 from-emerald-900/10'
    : sourceMissing
    ? 'border-red-800/40 from-red-900/10'
    : 'border-amber-800/40 from-amber-900/10';

  const stepBadgeAccent = isReady
    ? 'bg-emerald-500/15 border-emerald-500/40 text-emerald-300'
    : sourceMissing
    ? 'bg-red-500/15 border-red-500/40 text-red-300'
    : 'bg-amber-500/15 border-amber-500/40 text-amber-300';

  return (
    <div
      className={`rounded-xl border bg-gradient-to-br ${accent} to-gray-800 p-6 space-y-5 shadow-lg`}
    >
      {/* Header — step number, title, button */}
      <div className="flex items-start justify-between gap-4 flex-wrap">
        <div className="flex items-start gap-3 flex-1 min-w-0">
          <div
            className={`flex-shrink-0 w-9 h-9 rounded-lg border flex items-center justify-center font-bold text-sm ${stepBadgeAccent}`}
            aria-label="Step 1"
          >
            1
          </div>
          <div className="min-w-0">
            <h3 className="text-base font-semibold text-gray-100 flex items-center gap-2 flex-wrap">
              <Wand2 className="w-4 h-4 text-amber-400" />
              Stage your data for the generators
              {isReady && (
                <span className="text-[11px] font-normal text-emerald-300 bg-emerald-900/40 border border-emerald-700/60 px-2 py-0.5 rounded-full">
                  done
                </span>
              )}
            </h3>
            <p className="text-sm text-gray-400 mt-1.5 leading-relaxed max-w-2xl">
              CycleGAN and SPADE train on{' '}
              <strong className="text-gray-200">(mask, image) pairs</strong> in
              a fixed folder layout, but the raw dataset only ships with
              bounding-box <code className="text-gray-300 bg-gray-900/60 px-1 rounded">.txt</code>{' '}
              annotations. This step rasterises the boxes into binary mask PNGs
              and restages everything into the layout the trainer scripts below
              expect. Re-running is safe — it just refreshes any new
              annotations.
            </p>
          </div>
        </div>
        <div className="flex items-center gap-2 flex-shrink-0">
          <button
            onClick={onStart}
            disabled={isRunning || sourceMissing === true}
            title={
              sourceMissing
                ? 'Upload a TrainValid / Test dataset in the Dataset Explorer tab first'
                : isReady
                ? 'Re-run to refresh masks (idempotent)'
                : 'Generate masks and stage files for training'
            }
            className="flex items-center gap-2 bg-amber-600 hover:bg-amber-500 disabled:bg-gray-700 disabled:cursor-not-allowed disabled:text-gray-500 text-white px-5 py-2.5 rounded-lg text-sm font-medium transition-colors whitespace-nowrap shadow-lg"
          >
            {isRunning ? (
              <>
                <RefreshCw className="w-4 h-4 animate-spin" /> Preparing…
              </>
            ) : isReady ? (
              <>
                <RefreshCw className="w-4 h-4" /> Re-run prep
              </>
            ) : (
              <>
                <Wand2 className="w-4 h-4" /> Prepare dataset
              </>
            )}
          </button>
          {collapsible && (
            <button
              onClick={() => setExpanded(false)}
              title="Collapse this section"
              aria-label="Collapse prep step details"
              className="p-2 text-gray-500 hover:text-gray-200 hover:bg-gray-700/40 rounded-md transition-colors"
            >
              <ChevronUp className="w-4 h-4" />
            </button>
          )}
        </div>
      </div>

      {prepError && (
        <div className="flex items-start gap-2 text-xs bg-red-900/30 border border-red-800 rounded-lg px-3 py-2 text-red-300">
          <AlertCircle className="w-4 h-4 flex-shrink-0 mt-0.5" />
          <span>{prepError}</span>
        </div>
      )}

      {/* Visual transformation diagram — source layout → trainer layout. */}
      <div className="bg-gray-900/60 border border-gray-700 rounded-lg p-4">
        <div className="flex items-center gap-2 text-[10px] uppercase tracking-wider text-gray-500 mb-3">
          <FolderTree className="w-3 h-3" />
          What this step transforms
        </div>
        <div className="grid grid-cols-1 md:grid-cols-[1fr_auto_1fr] gap-4 items-center">
          {/* Input */}
          <div className="bg-gray-950/60 border border-gray-700 rounded-md p-3">
            <div className="text-[10px] text-gray-500 mb-1.5">
              Source · what you uploaded
            </div>
            <pre className="text-[11px] font-mono text-gray-300 leading-snug">
              {`TrainValid/, Test/
├─ `}
              <span className="text-blue-300">Images/</span>
              {`           `}
              <span className="text-gray-500">.jpg frames</span>
              {`
└─ `}
              <span className="text-blue-300">Annotations/</span>
              {`      `}
              <span className="text-gray-500">.txt boxes</span>
            </pre>
          </div>

          {/* Arrow + caption */}
          <div className="flex md:flex-col items-center justify-center gap-1 text-amber-400">
            <ArrowRight className="w-5 h-5 hidden md:block" />
            <ArrowRight className="w-5 h-5 md:hidden rotate-90" />
            <span className="text-[10px] text-gray-500 uppercase tracking-wider">
              prepare
            </span>
          </div>

          {/* Output */}
          <div className="bg-gray-950/60 border border-gray-700 rounded-md p-3">
            <div className="text-[10px] text-gray-500 mb-1.5">
              Trainer-ready · used by Train CycleGAN / Train SPADE below
            </div>
            <pre className="text-[11px] font-mono text-gray-300 leading-snug">
              <span className="text-emerald-300">PolypDataset/</span>
              {`            `}
              <span className="text-gray-500">(CycleGAN)</span>
              {`
├─ trainA/  `}
              <span className="text-gray-500">PNG masks</span>
              {`
├─ trainB/  `}
              <span className="text-gray-500">JPG images</span>
              {`
└─ testA/, testB/  `}
              <span className="text-gray-500">(same)</span>
              {`
`}
              <span className="text-emerald-300">PolypDatasetSPADE/</span>
              {`        `}
              <span className="text-gray-500">(SPADE)</span>
              {`
└─ identical mirror`}
            </pre>
          </div>
        </div>
        <p className="text-[11px] text-gray-500 mt-3 leading-relaxed flex items-start gap-1.5">
          <Info className="w-3 h-3 mt-0.5 flex-shrink-0" />
          <span>
            <strong className="text-gray-400">trainA / trainB</strong> are
            CycleGAN's "input domain" and "target domain". Here we put masks on
            the A side and real images on the B side, so the model learns{' '}
            <em className="text-gray-300">mask → realistic polyp image</em>.
            The two output directories are duplicates because CycleGAN and
            SPADE training scripts each expect their own copy.
          </span>
        </p>
      </div>

      {/* Readiness panel */}
      {datasetCheck && (
        <div
          className={`rounded-lg border px-4 py-3 ${
            isReady
              ? 'bg-emerald-900/20 border-emerald-800/60'
              : sourceMissing
              ? 'bg-red-900/20 border-red-800/60'
              : 'bg-amber-900/20 border-amber-800/60'
          }`}
        >
          {isReady ? (
            <div className="flex items-center gap-2 mb-2">
              <CheckCircle className="w-4 h-4 text-emerald-400 flex-shrink-0" />
              <span className="text-sm font-medium text-emerald-200">
                Ready for training — {trainPairs.toLocaleString()} train pairs
                · {testPairs.toLocaleString()} test pairs
              </span>
            </div>
          ) : sourceMissing ? (
            <div className="flex items-center gap-2 mb-2">
              <AlertCircle className="w-4 h-4 text-red-400 flex-shrink-0" />
              <span className="text-sm font-medium text-red-200">
                Source dataset missing — upload a{' '}
                <code className="bg-gray-900/70 text-red-200 px-1 rounded">
                  TrainValid/
                </code>{' '}
                folder in the{' '}
                <strong className="text-red-100">Dataset Explorer</strong> tab
                first.
              </span>
            </div>
          ) : (
            <div className="flex items-center gap-2 mb-2">
              <AlertCircle className="w-4 h-4 text-amber-400 flex-shrink-0" />
              <span className="text-sm font-medium text-amber-200">
                Source data found, but not yet staged. Click{' '}
                <strong>Prepare dataset</strong> to enable the training tasks
                below.
              </span>
            </div>
          )}

          <div className="grid grid-cols-1 sm:grid-cols-2 gap-3 mt-2">
            <DatasetReadinessBlock
              name="PolypDataset"
              subtitle="CycleGAN"
              splits={polyp}
              mismatch={polypMismatch}
            />
            <DatasetReadinessBlock
              name="PolypDatasetSPADE"
              subtitle="SPADE"
              splits={spade}
              mismatch={spadeMismatch}
            />
          </div>
        </div>
      )}
    </div>
  );
}

function DatasetReadinessBlock({
  name,
  subtitle,
  splits,
  mismatch,
}: {
  name: string;
  subtitle: string;
  splits: Record<string, number>;
  mismatch: boolean;
}) {
  const trainA = splits.trainA || 0;
  const trainB = splits.trainB || 0;
  const testA = splits.testA || 0;
  const testB = splits.testB || 0;
  const trainPairs = Math.min(trainA, trainB);
  const testPairs = Math.min(testA, testB);

  return (
    <div className="bg-gray-900/40 border border-gray-700 rounded-md p-3 text-xs">
      <div className="flex items-center justify-between mb-2">
        <span className="font-mono text-gray-200 font-medium">{name}</span>
        <span className="text-[10px] uppercase tracking-wider text-gray-500">
          {subtitle}
        </span>
      </div>
      <div className="space-y-1 text-gray-400">
        <ReadinessRow
          label="Train"
          pairs={trainPairs}
          a={trainA}
          b={trainB}
          mismatch={trainA !== trainB}
        />
        <ReadinessRow
          label="Test"
          pairs={testPairs}
          a={testA}
          b={testB}
          mismatch={testA !== testB}
        />
      </div>
      {mismatch && (
        <p className="text-[10px] text-amber-400 mt-2">
          ⚠ Mask / image counts don't match — re-run prep to resync.
        </p>
      )}
    </div>
  );
}

function ReadinessRow({
  label,
  pairs,
  a,
  b,
  mismatch,
}: {
  label: string;
  pairs: number;
  a: number;
  b: number;
  mismatch: boolean;
}) {
  return (
    <div className="flex items-baseline justify-between gap-2">
      <span className="text-gray-500">{label}</span>
      {mismatch ? (
        <span className="font-mono text-amber-300">
          {a.toLocaleString()} masks · {b.toLocaleString()} images
        </span>
      ) : (
        <span className="font-mono">
          <span className={pairs > 0 ? 'text-emerald-300' : 'text-gray-600'}>
            {pairs.toLocaleString()}
          </span>
          <span className="text-gray-500 ml-1">pairs</span>
        </span>
      )}
    </div>
  );
}