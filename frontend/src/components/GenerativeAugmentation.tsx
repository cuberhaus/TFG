import { useState, useEffect, useRef } from 'react';
import { api } from '../api';
import { Play, Square, X, Sparkles, CheckCircle, AlertCircle, RefreshCw, Image as ImageIcon, Wand2 } from 'lucide-react';

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

export default function GenerativeAugmentation() {
  const [taskType, setTaskType] = useState('train_cyclegan');
  const [experimentName, setExperimentName] = useState('');
  const [availableExperiments, setAvailableExperiments] = useState<Record<string, string[]>>({});
  const [epoch, setEpoch] = useState('latest');

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
  
  // Gallery state
  const [showGallery, setShowGallery] = useState(false);
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

  const fetchGallery = async (exp: string, ep: string) => {
    setGalleryLoading(true);
    setError(null);
    try {
      const response = await api.get(`/api/generate/results?experiment=${exp}&epoch=${ep}`);
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
    fetchExperiments();
    const rate = status.is_running || prepStatus.is_running ? 1000 : 3000;
    const interval = setInterval(() => {
      fetchStatus();
      fetchPrepStatus();
    }, rate);
    return () => clearInterval(interval);
  }, [status.is_running, prepStatus.is_running]);

  // When experiment name changes, reset the epoch to the first available one for that new experiment
  useEffect(() => {
     if (experimentName && availableExperiments[experimentName] && availableExperiments[experimentName].length > 0) {
         setEpoch(availableExperiments[experimentName][0]);
     }
  }, [experimentName, availableExperiments]);

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
        experiment_name: taskType === 'test_cyclegan' ? experimentName : undefined,
        epoch: taskType === 'test_cyclegan' ? epoch : undefined,
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
    <div className="max-w-4xl mx-auto flex flex-col gap-6 pt-2">
      <div className="flex items-start justify-between">
        <div>
          <h2 className="text-2xl font-bold mb-2 flex items-center gap-2">
            <Sparkles className="w-6 h-6 text-purple-400" />
            Generative Data Augmentation
          </h2>
          <p className="text-gray-400">
            Create synthetic training data using generative AI models to improve polyp detection performance.
          </p>
        </div>
        {!showGallery && (
           <button 
             onClick={() => {
               setShowGallery(true);
               fetchGallery(experimentName, epoch);
             }}
             className="flex items-center gap-2 bg-gray-800 hover:bg-gray-700 text-gray-200 border border-gray-600 px-4 py-2 rounded-lg text-sm transition-colors"
           >
             <ImageIcon className="w-4 h-4" />
             View Results Gallery
           </button>
        )}
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
        <div className="bg-gray-800 border border-gray-700 rounded-xl p-5 flex items-center justify-between gap-4">
          <div className="flex items-start gap-3">
            <Wand2 className="w-5 h-5 text-amber-400 mt-0.5 flex-shrink-0" />
            <div>
              <h3 className="text-sm font-medium text-gray-200">Prepare CycleGAN Dataset</h3>
              <p className="text-xs text-gray-400 mt-1">
                Generate binary masks from bounding box annotations and copy images + masks into the CycleGAN folder structure (PolypDataset &amp; PolypDatasetSPADE).
              </p>
            </div>
          </div>
          <button
            onClick={handleStartPreparation}
            disabled={prepStatus.is_running}
            className="flex items-center gap-2 bg-amber-600 hover:bg-amber-500 disabled:bg-gray-600 disabled:cursor-not-allowed text-white px-5 py-2.5 rounded-lg text-sm font-medium transition-colors whitespace-nowrap shadow-lg"
          >
            {prepStatus.is_running ? (
              <><RefreshCw className="w-4 h-4 animate-spin" /> Preparing...</>
            ) : (
              <><Wand2 className="w-4 h-4" /> Prepare Dataset</>
            )}
          </button>
          {prepError && (
            <span className="text-xs text-red-400">{prepError}</span>
          )}
        </div>
      )}

      {showGallery ? (
        <div className="flex flex-col gap-6 animate-in fade-in slide-in-from-bottom-4 duration-500">
           <div className="flex items-center justify-between bg-gray-800 border border-gray-700 p-4 rounded-xl gap-4">
             <div className="flex items-center gap-4 flex-1 min-w-0">
               <div className="min-w-0 flex-1">
                 <label className="text-xs text-gray-400 block uppercase tracking-wider mb-1">Experiment</label>
                 {Object.keys(availableExperiments).length > 0 ? (
                   <select
                     value={experimentName}
                     onChange={(e) => {
                       setExperimentName(e.target.value);
                       const epochs = availableExperiments[e.target.value];
                       const ep = epochs?.[0] || 'latest';
                       setEpoch(ep);
                       fetchGallery(e.target.value, ep);
                     }}
                     className="w-full bg-gray-900 border border-gray-600 rounded px-3 py-1.5 text-sm text-purple-300 font-mono font-semibold outline-none focus:border-purple-500"
                   >
                     {Object.keys(availableExperiments).map(exp => (
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
                 {experimentName && availableExperiments[experimentName]?.length > 0 ? (
                   <select
                     value={epoch}
                     onChange={(e) => {
                       setEpoch(e.target.value);
                       fetchGallery(experimentName, e.target.value);
                     }}
                     className="w-full bg-gray-900 border border-gray-600 rounded px-3 py-1.5 text-sm text-purple-300 font-mono outline-none focus:border-purple-500"
                   >
                     {availableExperiments[experimentName].map(ep => (
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
                 onClick={() => fetchGallery(experimentName, epoch)}
                 className="p-2 bg-gray-700 hover:bg-gray-600 rounded border border-gray-600 text-gray-300"
                 title="Refresh"
               >
                 <RefreshCw className={`w-4 h-4 ${galleryLoading ? 'animate-spin' : ''}`} />
               </button>
               <button 
                 onClick={() => setShowGallery(false)}
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
               <p className="text-sm mt-2">Run the "Test CycleGAN" job to generate some synthetic images first!</p>
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
      ) : status.is_running ? (
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