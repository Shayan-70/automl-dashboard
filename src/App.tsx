/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 */

import React, { useState } from 'react';
import {
  DatasetSummary,
  PreprocessingConfig,
  ModelResult,
  TrainingProgress,
} from './types';
import { analyzeDataset, buildPreprocessorConfig, transformData } from './ml/preprocessor';
import { trainTestSplit, evaluateClassification, evaluateRegression } from './ml/evaluator';
import {
  LinearRegressionModel,
  RandomForestRegressorModel,
  GradientBoostingRegressorModel,
  SupportVectorRegressorModel,
  KNearestNeighborsRegressorModel,
  LogisticRegressionModel,
  RandomForestClassifierModel,
  GradientBoostingClassifierModel,
  SupportVectorClassifierModel,
  KNearestNeighborsClassifierModel,
} from './ml/models';

import DatasetUploader from './components/DatasetUploader';
import DatasetAnalyzer from './components/DatasetAnalyzer';
import ColumnPruner from './components/ColumnPruner';
import ModelTrainer from './components/ModelTrainer';
import ModelComparer from './components/ModelComparer';
import PredictorDashboard from './components/PredictorDashboard';

import {
  BarChart2,
  Cpu,
  Layers,
  Sliders,
  UploadCloud,
  CheckCircle,
  AlertTriangle,
  RotateCcw,
  Menu,
} from 'lucide-react';

type Step = 'upload' | 'analyze' | 'prune' | 'train' | 'compare' | 'predict';

const sleep = (ms: number) => new Promise(resolve => setTimeout(resolve, ms));

export default function App() {
  const [step, setStep] = useState<Step>('upload');
  const [rawData, setRawData] = useState<any[]>([]);
  const [fileName, setFileName] = useState<string>('');
  const [error, setError] = useState<string | null>(null);
  const [excludedColumns, setExcludedColumns] = useState<string[]>([]);

  // States for ML Pipeline
  const [summary, setSummary] = useState<DatasetSummary | null>(null);
  const [preprocessorConfig, setPreprocessorConfig] = useState<PreprocessingConfig | null>(null);
  const [trainingProgress, setTrainingProgress] = useState<TrainingProgress>({
    status: 'idle',
    logs: [],
  });
  const [modelResults, setModelResults] = useState<ModelResult[]>([]);
  const [selectedModelId, setSelectedModelId] = useState<string | null>(null);

  const handleError = (msg: string) => {
    setError(msg);
    setStep('upload');
  };

  const resetPipeline = () => {
    setStep('upload');
    setRawData([]);
    setFileName('');
    setError(null);
    setExcludedColumns([]);
    setSummary(null);
    setPreprocessorConfig(null);
    setTrainingProgress({ status: 'idle', logs: [] });
    setModelResults([]);
    setSelectedModelId(null);
  };

  const handleDatasetLoaded = (data: any[], uploadedFileName: string) => {
    try {
      setError(null);
      setRawData(data);
      setFileName(uploadedFileName);

      // Perform initial profile
      const initialSummary = analyzeDataset(data);
      if (initialSummary.columns.length === 0) {
        throw new Error('No valid columns found in the dataset.');
      }
      setSummary(initialSummary);
      setStep('analyze');
    } catch (err: any) {
      handleError(err.message || 'Error processing dataset.');
    }
  };

  const handleTargetChanged = (newTarget: string) => {
    if (!rawData || rawData.length === 0) return;
    try {
      const updatedSummary = analyzeDataset(rawData, newTarget);
      setSummary(updatedSummary);
    } catch (err: any) {
      handleError(err.message || 'Error updating target.');
    }
  };

  const handleProblemTypeChanged = (newType: 'classification' | 'regression') => {
    if (!summary) return;
    setSummary(prev => (prev ? { ...prev, problemType: newType } : null));
  };

  const handleProceedToPruning = () => {
    if (!summary || !summary.targetColumn || !summary.problemType) {
      setError('Please select a valid target column and problem type first.');
      return;
    }
    setStep('prune');
  };

  const handlePruningCompleted = (excluded: string[]) => {
    setExcludedColumns(excluded);
    
    // Prune summary column definitions
    if (summary) {
      const updatedColumns = summary.columns.filter(col => !excluded.includes(col.name));
      setSummary({
        ...summary,
        columnCount: updatedColumns.length,
        columns: updatedColumns,
      });
    }

    setStep('train');
    runAutonomousTraining(excluded);
  };

  const runAutonomousTraining = async (activeExclusions: string[] = excludedColumns) => {
    if (!rawData || rawData.length === 0 || !summary || !summary.targetColumn || !summary.problemType) {
      setTrainingProgress({ status: 'failed', logs: ['❌ Missing configuration files. Returning to profile step.'] });
      return;
    }

    const target = summary.targetColumn;
    const problem = summary.problemType;

    // Prune the working dataframe by removing excluded columns
    const workingData = rawData.map(row => {
      const cleanRow = { ...row };
      activeExclusions.forEach(col => {
        delete cleanRow[col];
      });
      return cleanRow;
    });

    const initialLogs = ['🚀 Initializing Autonomous AI Machine Learning Pipeline...'];
    if (activeExclusions.length > 0) {
      initialLogs.push(`✅ Applied Feature Exclusions: Dropped ${activeExclusions.length} columns from active memory [${activeExclusions.join(', ')}]`);
    }
    initialLogs.push('=== Stage 1: Pipeline Compilation & Cleaning ===');

    setTrainingProgress({
      status: 'preprocessing',
      logs: initialLogs,
    });

    await sleep(400);

    // 1. Train-test split
    setTrainingProgress(prev => ({
      ...prev,
      logs: [...prev.logs, `✅ Partitioning dataset: 80% Train split, 20% Validation split (deterministic random seed 42)`],
    }));
    const { X_train: rawTrain, X_val: rawVal } = trainTestSplit(workingData, workingData, 0.2);
    
    await sleep(300);

    // 2. Build preprocessor configuration
    setTrainingProgress(prev => ({
      ...prev,
      logs: [...prev.logs, `✅ Building preprocessing configuration based on training statistical profile`],
    }));
    const config = buildPreprocessorConfig(rawTrain, target, problem);
    setPreprocessorConfig(config);

    await sleep(300);

    // 3. Inform user of preprocessing pipeline strategies
    const imputationCount = Object.keys(config.imputationStrategies).length;
    setTrainingProgress(prev => ({
      ...prev,
      logs: [
        ...prev.logs,
        `✅ Applied Missing Values Imputer: Median imputation on numeric features, Mode imputation on categorical features (${imputationCount} variables adjusted)`,
      ],
    }));

    await sleep(250);

    // One hot encoding check
    const encCount = Object.keys(config.encodedColumns).length;
    if (encCount > 0) {
      setTrainingProgress(prev => ({
        ...prev,
        logs: [
          ...prev.logs,
          `✅ Appended One-Hot encoding to ${encCount} categorical columns, mapping them to zero-one vectors`,
        ],
      }));
    }

    await sleep(250);

    // Interaction terms check
    const interactionCount = config.engineeredFeatures.length;
    if (interactionCount > 0) {
      setTrainingProgress(prev => ({
        ...prev,
        logs: [
          ...prev.logs,
          `✅ Feature engineering completed: Synthesized ${interactionCount} feature interaction cross-products (e.g. ${config.engineeredFeatures.map(f => f.newName).join(', ')})`,
        ],
      }));
    }

    await sleep(300);

    // Data transform
    setTrainingProgress(prev => ({
      ...prev,
      logs: [...prev.logs, `✅ Mapping rows into numeric vector matrices: standardized numeric columns to Z-score scaled units`],
    }));

    const trainTransformed = transformData(rawTrain, config, true);
    const valTransformed = transformData(rawVal, config, true);

    const featureCount = config.selectedFeatures.length;
    setTrainingProgress(prev => ({
      ...prev,
      logs: [
        ...prev.logs,
        `📊 Matched shape - Train Set: ${trainTransformed.features.length} rows, ${featureCount} features. Validation Set: ${valTransformed.features.length} rows.`,
        '=== Stage 2: Model Tuning & Convergence ===',
      ],
    }));

    await sleep(500);

    const trainedResults: ModelResult[] = [];

    // Training Regression models
    if (problem === 'regression') {
      const models = [
        { id: 'linear_regression', name: 'L2-Regularized Ridge Regression', cls: LinearRegressionModel },
        { id: 'random_forest', name: 'Random Forest Regressor (Ensemble)', cls: RandomForestRegressorModel },
        { id: 'gradient_boosting', name: 'Gradient Boosting Regressor', cls: GradientBoostingRegressorModel },
        { id: 'support_vector_regression', name: 'Support Vector Regressor', cls: SupportVectorRegressorModel },
        { id: 'knn_regression', name: 'K-Nearest Neighbors Regressor', cls: KNearestNeighborsRegressorModel },
      ];

      for (const mDef of models) {
        setTrainingProgress(prev => ({
          ...prev,
          status: 'training',
          currentModel: mDef.name,
          logs: [...prev.logs, `⚙️ Training Model: ${mDef.name} ...`],
        }));

        const startTime = Date.now();
        const modelObj = new mDef.cls();
        modelObj.fit(trainTransformed.features, trainTransformed.labels);
        const elapsed = Date.now() - startTime;

        // Predict on validation
        const preds = modelObj.predict(valTransformed.features);
        const metrics = evaluateRegression(valTransformed.labels.map(Number), preds);

        trainedResults.push({
          modelId: mDef.id,
          modelName: mDef.name,
          metrics,
          trainingTimeMs: elapsed,
          predictions: preds,
          actuals: valTransformed.labels,
          modelObject: modelObj,
        });

        setTrainingProgress(prev => ({
          ...prev,
          logs: [...prev.logs, `✅ Optimized ${mDef.name} in ${elapsed}ms | validation R²: ${((metrics.r2 ?? 0) * 100).toFixed(1)}% | MAE: ${metrics.mae.toFixed(3)}`],
        }));

        await sleep(400);
      }
    } else {
      // Training Classification models
      const models = [
        { id: 'logistic_regression', name: 'Logistic Regression (One-vs-Rest)', cls: LogisticRegressionModel },
        { id: 'random_forest', name: 'Random Forest Classifier (Ensemble)', cls: RandomForestClassifierModel },
        { id: 'gradient_boosting', name: 'Gradient Boosting Classifier', cls: GradientBoostingClassifierModel },
        { id: 'support_vector_classification', name: 'Support Vector Classifier', cls: SupportVectorClassifierModel },
        { id: 'knn_classification', name: 'K-Nearest Neighbors Classifier', cls: KNearestNeighborsClassifierModel },
      ];

      for (const mDef of models) {
        setTrainingProgress(prev => ({
          ...prev,
          status: 'training',
          currentModel: mDef.name,
          logs: [...prev.logs, `⚙️ Training Model: ${mDef.name} ...`],
        }));

        const startTime = Date.now();
        const modelObj = new mDef.cls();
        modelObj.fit(trainTransformed.features, trainTransformed.labels);
        const elapsed = Date.now() - startTime;

        // Predict on validation
        const preds = modelObj.predict(valTransformed.features);
        const probas = modelObj.predictProba(valTransformed.features);
        const metrics = evaluateClassification(valTransformed.labels, preds, probas, modelObj.classes);

        trainedResults.push({
          modelId: mDef.id,
          modelName: mDef.name,
          metrics,
          trainingTimeMs: elapsed,
          predictions: preds,
          actuals: valTransformed.labels,
          modelObject: modelObj,
        });

        setTrainingProgress(prev => ({
          ...prev,
          logs: [
            ...prev.logs,
            `✅ Optimized ${mDef.name} in ${elapsed}ms | accuracy: ${((metrics.accuracy ?? 0) * 100).toFixed(1)}% | F1: ${((metrics.f1 ?? 0) * 100).toFixed(1)}%`,
          ],
        }));

        await sleep(400);
      }
    }

    setTrainingProgress(prev => ({
      ...prev,
      status: 'evaluating',
      logs: [...prev.logs, '=== Stage 3: Auto-Inference Selection ==='],
    }));

    await sleep(350);

    // Identify best model
    let bestId = trainedResults[0].modelId;
    let bestScore = -Infinity;

    trainedResults.forEach(res => {
      const score = problem === 'classification' ? (res.metrics.f1 ?? 0) : (res.metrics.r2 ?? 0);
      if (score > bestScore) {
        bestScore = score;
        bestId = res.modelId;
      }
    });

    const bestModelName = trainedResults.find(r => r.modelId === bestId)?.modelName;

    setModelResults(trainedResults);
    setSelectedModelId(bestId);

    setTrainingProgress(prev => ({
      ...prev,
      status: 'completed',
      logs: [
        ...prev.logs,
        `👑 Selection Champion: "${bestModelName}" scored highest on validation cross-checks.`,
        `🚀 Autopilot fully compiled. System is ready to deploy predictions in sandbox.`,
      ],
    }));
  };

  return (
    <div id="app-root-container" className="min-h-screen bg-[#E4E3E0] text-[#141414] font-sans flex flex-col">
      {/* 1. Header Navigation Bar */}
      <header className="h-16 border-b border-[#141414] flex items-center justify-between px-8 bg-white text-[#141414] sticky top-0 z-50">
        <div className="flex items-center gap-4">
          <div className="w-8 h-8 bg-[#141414] flex items-center justify-center">
            <span className="text-white font-bold text-xs italic font-display">A.I.</span>
          </div>
          <div>
            <h1 className="text-md sm:text-lg font-black tracking-tight uppercase font-display text-[#141414] flex items-center gap-2">
              Autonomous Data Science Agent
              <span className="text-[10px] font-mono font-normal opacity-50 hidden sm:inline">v1.0.4-PRO</span>
            </h1>
            {fileName && (
              <p className="text-[10px] text-slate-500 font-bold font-mono truncate max-w-[200px] sm:max-w-[320px]">
                FILE: {fileName.toUpperCase()}
              </p>
            )}
          </div>
        </div>

        {/* Stepper Navigation Indicator */}
        <div className="hidden lg:flex items-center gap-6">
          <div className={`text-[10px] font-mono uppercase tracking-widest font-bold ${step === 'upload' ? 'border-b-2 border-[#141414] text-[#141414]' : 'opacity-40 text-[#141414]'}`}>
            01. Upload
          </div>
          <div className={`text-[10px] font-mono uppercase tracking-widest font-bold ${step === 'analyze' ? 'border-b-2 border-[#141414] text-[#141414]' : 'opacity-40 text-[#141414]'}`}>
            02. Profile
          </div>
          <div className={`text-[10px] font-mono uppercase tracking-widest font-bold ${step === 'prune' ? 'border-b-2 border-[#141414] text-[#141414]' : 'opacity-40 text-[#141414]'}`}>
            03. Prune
          </div>
          <div className={`text-[10px] font-mono uppercase tracking-widest font-bold ${step === 'train' ? 'border-b-2 border-[#141414] text-[#141414]' : 'opacity-40 text-[#141414]'}`}>
            04. Train
          </div>
          <div className={`text-[10px] font-mono uppercase tracking-widest font-bold ${step === 'compare' ? 'border-b-2 border-[#141414] text-[#141414]' : 'opacity-40 text-[#141414]'}`}>
            05. Compare
          </div>
          <div className={`text-[10px] font-mono uppercase tracking-widest font-bold ${step === 'predict' ? 'border-b-2 border-[#141414] text-[#141414]' : 'opacity-40 text-[#141414]'}`}>
            06. Predict
          </div>
        </div>

        <div className="flex items-center gap-3">
          {step !== 'upload' && (
            <button
              onClick={resetPipeline}
              id="header-restart-button"
              className="px-4 py-2 border border-[#141414] bg-white text-[#141414] text-xs font-bold uppercase hover:bg-[#141414] hover:text-white transition-colors rounded-none cursor-pointer"
            >
              Reset Sandbox
            </button>
          )}
        </div>
      </header>

      {/* 2. Main Workspace Body */}
      <main className="flex-grow max-w-7xl mx-auto w-full px-6 py-8">
        {error && (
          <div id="global-error-banner" className="mb-6 p-4 border border-[#141414] bg-red-100 text-[#141414] rounded-none flex gap-3">
            <AlertTriangle className="w-5 h-5 text-red-600 flex-shrink-0 mt-0.5" />
            <div className="space-y-1">
              <h4 className="text-xs font-bold font-mono uppercase tracking-wider">System Warning</h4>
              <p className="text-xs text-[#141414] leading-relaxed">{error}</p>
            </div>
          </div>
        )}

        {/* Dynamic Workflow Stage Router */}
        {step === 'upload' && (
          <DatasetUploader onDatasetLoaded={handleDatasetLoaded} onError={handleError} />
        )}

        {step === 'analyze' && summary && (
          <DatasetAnalyzer
            summary={summary}
            onTargetChanged={handleTargetChanged}
            onProblemTypeChanged={handleProblemTypeChanged}
            onProceedToTraining={handleProceedToPruning}
          />
        )}

        {step === 'prune' && summary && (
          <ColumnPruner
            summary={summary}
            onProceed={handlePruningCompleted}
            onBack={() => setStep('analyze')}
          />
        )}

        {step === 'train' && (
          <ModelTrainer
            progress={trainingProgress}
            onProceedToCompare={() => setStep('compare')}
            onRestart={resetPipeline}
          />
        )}

        {step === 'compare' && summary && (
          <ModelComparer
            results={modelResults}
            problemType={summary.problemType!}
            selectedModelId={selectedModelId}
            onSelectModel={setSelectedModelId}
            onProceedToPredict={() => setStep('predict')}
          />
        )}

        {step === 'predict' && preprocessorConfig && selectedModelId && summary && (
          <PredictorDashboard
            config={preprocessorConfig}
            modelResult={modelResults.find(r => r.modelId === selectedModelId)!}
            columnsInfo={summary.columns}
          />
        )}
      </main>

      {/* 3. Footer Bar */}
      <footer className="h-10 border-t border-[#141414] bg-white px-8 flex items-center justify-between text-[10px] font-mono opacity-80 text-[#141414]">
        <div>PROJECT: 2026_Q3_AUTOPILOT_SCIENCE</div>
        <div>ENVIRONMENT: PRODUCTION_SANDBOX_STABLE</div>
        <div>RUNTIME: ONLINE EXECUTION</div>
      </footer>
    </div>
  );
}
