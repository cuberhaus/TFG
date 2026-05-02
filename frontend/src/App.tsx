import { useState } from 'react';
import { BarChart, Search, Activity, Play, TrendingUp, ClipboardList, Sparkles, Sliders, FolderSearch, Menu, Images, Layers } from 'lucide-react';
import PerformanceExplorer from './components/PerformanceExplorer';
import InferenceUI from './components/InferenceUI';
import ModelTraining from './components/ModelTraining';
import TrainingLossViewer from './components/TrainingLossViewer';
import ModelEvaluation from './components/ModelEvaluation';
import GenerativeAugmentation, { GenerativeView } from './components/GenerativeAugmentation';
import HyperparameterTuning from './components/HyperparameterTuning';
import DatasetExplorer from './components/DatasetExplorer';
import MlopsStatusCard from './components/MlopsStatusCard';

// Sidebar tab IDs that all render <GenerativeAugmentation> with a different
// `view` prop. Kept as a Set so the conditional in the JSX stays readable.
const GENERATIVE_TABS: Record<string, GenerativeView> = {
  generative: 'tasks',
  'cyclegan-results': 'cyclegan-results',
  'spade-results': 'spade-results',
};

function App() {
  const [activeTab, setActiveTab] = useState('dataset');
  const [isSidebarOpen, setIsSidebarOpen] = useState(true);

  const navGroups = [
    {
      title: "Data & Exploration",
      items: [
        { id: 'dataset', label: 'Dataset Explorer', icon: FolderSearch, color: 'text-blue-400' },
        { id: 'generative', label: 'Data Augmentation', icon: Sparkles, color: 'text-purple-400' },
        { id: 'cyclegan-results', label: 'CycleGAN Results', icon: Images, color: 'text-purple-300' },
        { id: 'spade-results', label: 'SPADE Results', icon: Layers, color: 'text-fuchsia-300' },
      ]
    },
    {
      title: "Training & Optimization",
      items: [
        { id: 'hpo', label: 'Hyperparameter Tuning', icon: Sliders, color: 'text-fuchsia-400' },
        { id: 'training', label: 'Detection Training', icon: Play, color: 'text-teal-400' },
        { id: 'losses', label: 'Training Losses', icon: TrendingUp, color: 'text-emerald-400' },
      ]
    },
    {
      title: "Evaluation & Analytics",
      items: [
        { id: 'evaluation', label: 'Model Evaluation', icon: ClipboardList, color: 'text-green-400' },
        { id: 'performance', label: 'Performance Explorer', icon: BarChart, color: 'text-indigo-400' },
      ]
    },
    {
      title: "Inference & Testing",
      items: [
        { id: 'inference', label: 'Inference', icon: Search, color: 'text-cyan-400' },
      ]
    }
  ];

  return (
    <div className="h-screen bg-gray-900 text-white flex flex-col w-full overflow-hidden">
      <header className="w-full bg-gray-800 border-b border-gray-700 py-4 px-6 shadow-sm z-20 flex items-center justify-between">
        <div className="flex items-center gap-4">
          <button 
            onClick={() => setIsSidebarOpen(!isSidebarOpen)}
            className="p-2 bg-gray-700 hover:bg-gray-600 rounded-md text-gray-300 transition-colors"
          >
            <Menu className="w-5 h-5" />
          </button>
          <div className="flex items-center gap-3">
            <Activity className="w-7 h-7 text-blue-400" />
            <div>
              <h1 className="text-xl font-bold leading-tight">Polyp Detection Dashboard</h1>
              <p className="text-xs text-gray-400">Deep Learning Polyp Detection with Generative Data Augmentation</p>
            </div>
          </div>
        </div>
      </header>

      <div className="flex flex-1 overflow-hidden">
        {/* Sidebar */}
        <aside 
          className="bg-gray-800 border-r border-gray-700 flex flex-col shadow-xl transition-all duration-300 ease-in-out overflow-hidden z-10"
          style={{ 
            width: isSidebarOpen ? '16rem' : '0px',
            minWidth: isSidebarOpen ? '16rem' : '0px'
          }}
        >
          <div className="w-64 flex flex-col h-full">
            <nav className="flex-1 py-4 px-3 space-y-4 overflow-y-auto">
              {navGroups.map((group) => (
                <div key={group.title} className="space-y-1.5">
                  <h3 className="px-3 text-xs font-semibold text-gray-500 uppercase tracking-wider mb-2">
                    {group.title}
                  </h3>
                  {group.items.map((item) => {
                    const Icon = item.icon;
                    const isActive = activeTab === item.id;
                    return (
                      <button
                        key={item.id}
                        onClick={() => setActiveTab(item.id)}
                        className={`w-full flex items-center gap-3 px-3 py-2.5 rounded-lg font-medium transition-all ${
                          isActive 
                            ? 'bg-gray-700 text-white shadow-sm border border-gray-600' 
                            : 'text-gray-400 hover:text-gray-200 hover:bg-gray-700 border border-transparent'
                        }`}
                      >
                        <Icon className={`w-5 h-5 flex-shrink-0 ${isActive ? item.color : 'text-gray-500'}`} />
                        <span className="text-sm text-left truncate">{item.label}</span>
                      </button>
                    );
                  })}
                </div>
              ))}
            </nav>
            <div className="p-4 border-t border-gray-700 text-xs text-gray-500 text-center">
              TFG Project <br/>© {new Date().getFullYear()}
            </div>
          </div>
        </aside>

        {/* Main Content */}
        <main className="flex-1 overflow-y-auto p-4 md:p-8 bg-gray-900 w-full relative">
          {/* Outer width cap — was `max-w-6xl` (1152px), bumped to
              `max-w-screen-2xl` (1536px) so the visible card fills more
              screen on modern 1920+px displays without becoming
              ridiculously wide on 4K. Components inside still set their
              own caps appropriate to their content (forms tighter,
              grids wider). */}
          <div className="max-w-screen-2xl mx-auto w-full">
            <div className="bg-gray-800 rounded-xl shadow-lg border border-gray-700 p-6 min-h-[600px]">
              {activeTab === 'performance' && <PerformanceExplorer />}
              {activeTab === 'inference' && <InferenceUI />}
              {activeTab === 'training' && <ModelTraining />}
              {activeTab === 'losses' && <TrainingLossViewer />}
              {activeTab === 'evaluation' && <ModelEvaluation />}
              {/* All three generative-related sidebar tabs render the
                  same component with a different `view`. Sharing one mount
                  preserves the dataset-prep status and form state when the
                  user flips between Tasks / CycleGAN / SPADE galleries. */}
              {activeTab in GENERATIVE_TABS && (
                <GenerativeAugmentation
                  view={GENERATIVE_TABS[activeTab]}
                  onNavigate={(v) => {
                    const id = Object.entries(GENERATIVE_TABS).find(([, val]) => val === v)?.[0];
                    if (id) setActiveTab(id);
                  }}
                />
              )}
              {activeTab === 'hpo' && <HyperparameterTuning />}
              {activeTab === 'dataset' && <DatasetExplorer />}
            </div>
          </div>
        </main>
        
        {/* Mobile overlay */}
        {isSidebarOpen && (
          <div 
            className="md:hidden fixed inset-0 bg-black/50 z-0" 
            onClick={() => setIsSidebarOpen(false)}
          />
        )}
      </div>

      {/* MLOps observability status — visible on every tab. Auto-renders
          a graceful "stack offline" pill when MLOPS_PREDICTION_LOG_DSN
          isn't set on the backend. */}
      <MlopsStatusCard />
    </div>
  );
}

export default App;
