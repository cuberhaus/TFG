import { useState, useEffect } from 'react';
import axios from 'axios';
import { BarChart, Search, Activity, Cpu, Play, TrendingUp, ClipboardList, Sparkles, Sliders } from 'lucide-react';
import PerformanceExplorer from './components/PerformanceExplorer';
import InferenceUI from './components/InferenceUI';
import ModelTraining from './components/ModelTraining';
import TrainingLossViewer from './components/TrainingLossViewer';
import ModelEvaluation from './components/ModelEvaluation';
import GenerativeAugmentation from './components/GenerativeAugmentation';
import HyperparameterTuning from './components/HyperparameterTuning';

function App() {
  const [activeTab, setActiveTab] = useState('performance');

  return (
    <div className="min-h-screen bg-gray-900 text-white flex flex-col items-center w-full">
      <header className="w-full bg-gray-800 border-b border-gray-700 py-6 px-8 shadow-sm">
        <div className="max-w-6xl mx-auto flex items-center gap-3">
          <Activity className="w-8 h-8 text-blue-400" />
          <div>
            <h1 className="text-2xl font-bold">Polyp Detection Dashboard</h1>
            <p className="text-sm text-gray-400">Deep Learning Polyp Detection with Generative Data Augmentation</p>
          </div>
        </div>
      </header>

      <main className="flex-1 w-full max-w-6xl mx-auto p-6">
        <div className="flex border-b border-gray-700 mb-6">
          <button
            onClick={() => setActiveTab('performance')}
            className={`flex items-center gap-2 py-3 px-6 font-medium transition-colors ${
              activeTab === 'performance'
                ? 'border-b-2 border-blue-500 text-blue-400'
                : 'text-gray-400 hover:text-gray-200'
            }`}
          >
            <BarChart className="w-4 h-4" />
            Performance Explorer
          </button>
          <button
            onClick={() => setActiveTab('inference')}
            className={`flex items-center gap-2 py-3 px-6 font-medium transition-colors ${
              activeTab === 'inference'
                ? 'border-b-2 border-blue-500 text-blue-400'
                : 'text-gray-400 hover:text-gray-200'
            }`}
          >
            <Search className="w-4 h-4" />
            Inference
          </button>
          <button
            onClick={() => setActiveTab('training')}
            className={`flex items-center gap-2 py-3 px-6 font-medium transition-colors ${
              activeTab === 'training'
                ? 'border-b-2 border-blue-500 text-blue-400'
                : 'text-gray-400 hover:text-gray-200'
            }`}
          >
            <Play className="w-4 h-4" />
            Detection Training
          </button>
          <button
            onClick={() => setActiveTab('losses')}
            className={`flex items-center gap-2 py-3 px-6 font-medium transition-colors ${
              activeTab === 'losses'
                ? 'border-b-2 border-blue-500 text-blue-400'
                : 'text-gray-400 hover:text-gray-200'
            }`}
          >
            <TrendingUp className="w-4 h-4" />
            Training Losses
          </button>
          <button
            onClick={() => setActiveTab('evaluation')}
            className={`flex items-center gap-2 py-3 px-6 font-medium transition-colors ${
              activeTab === 'evaluation'
                ? 'border-b-2 border-blue-500 text-blue-400'
                : 'text-gray-400 hover:text-gray-200'
            }`}
          >
            <ClipboardList className="w-4 h-4" />
            Model Evaluation
          </button>
          <button
            onClick={() => setActiveTab('generative')}
            className={`flex items-center gap-2 py-3 px-6 font-medium transition-colors ${
              activeTab === 'generative'
                ? 'border-b-2 border-purple-500 text-purple-400'
                : 'text-gray-400 hover:text-gray-200'
            }`}
          >
            <Sparkles className="w-4 h-4" />
            Data Augmentation
          </button>
          <button
            onClick={() => setActiveTab('hpo')}
            className={`flex items-center gap-2 py-3 px-6 font-medium transition-colors ${
              activeTab === 'hpo'
                ? 'border-b-2 border-emerald-500 text-emerald-400'
                : 'text-gray-400 hover:text-gray-200'
            }`}
          >
            <Sliders className="w-4 h-4" />
            Hyperparameter Tuning
          </button>
        </div>

        <div className="bg-gray-800 rounded-lg shadow-md border border-gray-700 p-6 min-h-[500px]">
          {activeTab === 'performance' && <PerformanceExplorer />}
          {activeTab === 'inference' && <InferenceUI />}
          {activeTab === 'training' && <ModelTraining />}
          {activeTab === 'losses' && <TrainingLossViewer />}
          {activeTab === 'evaluation' && <ModelEvaluation />}
          {activeTab === 'generative' && <GenerativeAugmentation />}
          {activeTab === 'hpo' && <HyperparameterTuning />}
        </div>
      </main>
    </div>
  );
}

export default App;
