/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 */

import React, { useState } from 'react';
import { PreprocessingConfig, ModelResult, ColumnInfo } from '../types';
import { transformData } from '../ml/preprocessor';
import * as XLSX from 'xlsx';
import {
  Sparkles,
  Play,
  FileSpreadsheet,
  Download,
  Upload,
  RefreshCw,
  HelpCircle,
  TrendingUp,
  Sliders,
  Check,
} from 'lucide-react';
import { motion } from 'motion/react';

interface PredictorDashboardProps {
  config: PreprocessingConfig;
  modelResult: ModelResult;
  columnsInfo: ColumnInfo[];
}

export default function PredictorDashboard({
  config,
  modelResult,
  columnsInfo,
}: PredictorDashboardProps) {
  const { problemType, numericScales, encodedColumns, targetColumn } = config;
  const model = modelResult.modelObject;

  // Initialize input state with average values or first category
  const [inputs, setInputs] = useState<{ [key: string]: any }>(() => {
    const initial: { [key: string]: any } = {};
    
    // Numeric defaults
    Object.entries(numericScales).forEach(([colName, scale]) => {
      initial[colName] = parseFloat(scale.mean.toFixed(2));
    });

    // Categorical defaults
    Object.entries(encodedColumns).forEach(([colName, info]) => {
      initial[colName] = info.categories[0] ?? '';
    });

    return initial;
  });

  const [predictionResult, setPredictionResult] = useState<any | null>(null);
  const [predictionProbas, setPredictionProbas] = useState<{ label: string; probability: number }[] | null>(null);

  // Batch prediction states
  const [batchLoading, setBatchLoading] = useState(false);
  const [batchFile, setBatchFile] = useState<File | null>(null);
  const [batchPreview, setBatchPreview] = useState<any[] | null>(null);
  const [batchOutputData, setBatchOutputData] = useState<any[] | null>(null);
  const [batchError, setBatchError] = useState<string | null>(null);

  const handleInputChange = (colName: string, value: any) => {
    setInputs(prev => ({
      ...prev,
      [colName]: value,
    }));
  };

  const executeSinglePrediction = () => {
    try {
      // Assemble a single raw row object
      const rawRow = { ...inputs };
      
      // Transform single row
      const { features } = transformData([rawRow], config, false);

      // Run predict
      const pred = model.predict(features)[0];
      setPredictionResult(pred);

      // If classification, get probability breakdown if supported
      if (problemType === 'classification' && typeof model.predictProba === 'function') {
        const probasRow = model.predictProba(features)[0];
        const classes = model.classes;
        const mapped = classes.map((cl: any, i: number) => ({
          label: String(cl),
          probability: parseFloat(((probasRow[i] ?? 0) * 100).toFixed(1)),
        })).sort((a: any, b: any) => b.probability - a.probability);
        setPredictionProbas(mapped);
      } else {
        setPredictionProbas(null);
      }
    } catch (err) {
      console.error(err);
    }
  };

  // Batch File Upload handlers
  const handleBatchFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files.length > 0) {
      const file = e.target.files[0];
      setBatchFile(file);
      setBatchError(null);
      setBatchOutputData(null);

      const reader = new FileReader();
      reader.onload = (event) => {
        try {
          const data = new Uint8Array(event.target?.result as ArrayBuffer);
          const workbook = XLSX.read(data, { type: 'array' });
          const firstSheetName = workbook.SheetNames[0];
          const worksheet = workbook.Sheets[firstSheetName];
          const json = XLSX.utils.sheet_to_json(worksheet, { defval: null });

          if (json.length === 0) {
            throw new Error('Spreadsheet has no rows.');
          }

          setBatchPreview(json.slice(0, 5)); // show first 5 rows for preview
          
          // Execute batch predictions
          setBatchLoading(true);
          setTimeout(() => {
            try {
              const { features } = transformData(json, config, false);
              const preds = model.predict(features);
              
              // If classification and supports probabilities
              let probas: number[][] | null = null;
              if (problemType === 'classification' && typeof model.predictProba === 'function') {
                probas = model.predictProba(features);
              }

              // Append predictions to original records
              const enriched = json.map((row, idx) => {
                const output: { [key: string]: any } = Object.assign({}, row as any);
                const predVal = preds[idx];
                output[`predicted_${targetColumn}`] = predVal;

                if (probas && probas[idx]) {
                  model.classes.forEach((cl: any, cIdx: number) => {
                    const pct = parseFloat(((probas![idx][cIdx] ?? 0) * 100).toFixed(1));
                    output[`probability_class_${cl}_pct`] = pct;
                  });
                }
                return output;
              });

              setBatchOutputData(enriched);
            } catch (innerErr: any) {
              setBatchError(innerErr.message || 'Pipeline failed during batch prediction.');
            } finally {
              setBatchLoading(false);
            }
          }, 400);

        } catch (err: any) {
          setBatchError(err.message || 'Error parsing batch predictions sheet.');
        }
      };
      reader.readAsArrayBuffer(file);
    }
  };

  const downloadBatchResults = () => {
    if (!batchOutputData || batchOutputData.length === 0) return;
    
    const ws = XLSX.utils.json_to_sheet(batchOutputData);
    const wb = XLSX.utils.book_new();
    XLSX.utils.book_append_sheet(wb, ws, 'Inference_Predictions');
    
    const fileName = batchFile 
      ? `predicted_${batchFile.name.split('.')[0]}.xlsx`
      : 'batch_predictions.xlsx';
      
    XLSX.writeFile(wb, fileName);
  };

  // Helper to format predictions elegantly
  const formatPrediction = (val: any) => {
    if (val === null || val === undefined) return '—';
    if (problemType === 'regression') {
      const num = Number(val);
      if (targetColumn.toLowerCase().includes('price') || targetColumn.toLowerCase().includes('usd')) {
        return new Intl.NumberFormat('en-US', { style: 'currency', currency: 'USD', maximumFractionDigits: 0 }).format(num);
      }
      return num.toLocaleString('en-US', { maximumFractionDigits: 2 });
    }
    return String(val);
  };

  return (
    <div id="predictor-dashboard-container" className="space-y-8 animate-fade-in text-[#141414]">
      {/* Visual Header */}
      <div className="flex flex-col md:flex-row md:items-center md:justify-between pb-6 border-b border-[#141414] gap-4">
        <div>
          <h2 className="text-xl font-bold font-display uppercase tracking-tight text-[#141414] flex items-center gap-2">
            <Sliders className="w-5.5 h-5.5 text-[#141414]" />
            Live Inference & Deployment Playground
          </h2>
          <p className="text-[11px] font-mono text-slate-500 uppercase tracking-wider mt-1">
            RUNNING ACTIVE PIPELINE ENGINE: <span className="bg-yellow-400 text-[#141414] px-1.5 py-0.5 font-bold border border-[#141414]">{modelResult.modelName.toUpperCase()}</span>
          </p>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-12 gap-8">
        {/* Left column: Manual Form Input (7 cols) */}
        <div className="lg:col-span-7 space-y-6">
          <div className="p-6 border border-[#141414] bg-white rounded-none space-y-6">
            <h3 className="text-xs font-bold uppercase tracking-widest text-slate-500 flex items-center gap-1.5 italic">
              <Sliders className="w-4.5 h-4.5 text-[#141414]" /> 01. Custom Feature Parameters
            </h3>

            <div className="grid grid-cols-1 md:grid-cols-2 gap-5">
              {/* Render original numeric features */}
              {Object.entries(numericScales).map(([colName, scale]) => {
                const colInfo = columnsInfo.find(c => c.name === colName);
                const minVal = colInfo?.min ?? 0;
                const maxVal = colInfo?.max ?? 100;
                return (
                  <div key={colName} className="space-y-2">
                    <label className="text-xs font-bold uppercase tracking-wider text-[#141414] font-mono flex justify-between">
                      <span>{colName}</span>
                      <span className="text-slate-400 font-normal">
                        ({minVal} - {maxVal})
                      </span>
                    </label>
                    <input
                      type="number"
                      value={inputs[colName] ?? ''}
                      onChange={(e) => handleInputChange(colName, e.target.value === '' ? '' : Number(e.target.value))}
                      className="w-full bg-neutral-50 border border-[#141414] rounded-none px-4 py-2.5 text-xs font-mono focus:outline-none"
                    />
                  </div>
                );
              })}

              {/* Render original categorical features */}
              {Object.entries(encodedColumns).map(([colName, info]) => {
                return (
                  <div key={colName} className="space-y-2">
                    <label className="text-xs font-bold uppercase tracking-wider text-[#141414] font-mono">
                      {colName}
                    </label>
                    <select
                      value={inputs[colName] ?? ''}
                      onChange={(e) => handleInputChange(colName, e.target.value)}
                      className="w-full bg-neutral-50 border border-[#141414] rounded-none px-4 py-2.5 text-xs font-mono focus:outline-none"
                    >
                      {info.categories.map(cat => (
                        <option key={String(cat)} value={String(cat)}>
                          {String(cat).toUpperCase()}
                        </option>
                      ))}
                    </select>
                  </div>
                );
              })}
            </div>

            <button
              onClick={executeSinglePrediction}
              id="playground-predict-button"
              className="w-full py-3.5 border border-[#141414] bg-[#141414] hover:bg-transparent hover:text-[#141414] text-white font-bold text-xs uppercase tracking-wider rounded-none transition-colors flex items-center justify-center gap-2 cursor-pointer"
            >
              <Play className="w-4 h-4 fill-current" />
              Compile Live Prediction
            </button>
          </div>
        </div>

        {/* Right column: Prediction Outputs & Batch Predictor (5 cols) */}
        <div className="lg:col-span-5 space-y-6">
          {/* Active Single Prediction Result Card */}
          <div className="p-6 border border-[#141414] bg-white rounded-none flex flex-col justify-between min-h-[180px]">
            <div>
              <h3 className="text-xs font-bold uppercase tracking-widest text-slate-500 mb-4 italic">
                02. Live Prediction Output
              </h3>
              {predictionResult !== null ? (
                <div className="space-y-4">
                  <div className="space-y-2">
                    <p className="text-[10px] font-mono uppercase font-bold text-slate-500">Inferred Output Value:</p>
                    <div className="p-4 bg-yellow-400/10 border border-[#141414] text-center">
                      <p className="text-2xl font-bold font-mono tracking-tight text-[#141414]">
                        {formatPrediction(predictionResult).toUpperCase()}
                      </p>
                    </div>
                  </div>

                  {/* Display Probability Breakdown if classification */}
                  {predictionProbas && predictionProbas.length > 0 && (
                    <div className="space-y-3 mt-4 pt-4 border-t border-dashed border-[#141414]">
                      <p className="text-[10px] font-bold text-slate-500 uppercase tracking-widest font-mono">
                        Prediction Confidences
                      </p>
                      <div className="space-y-2">
                        {predictionProbas.slice(0, 3).map((item, idx) => (
                          <div key={item.label} className="space-y-1 text-xs">
                            <div className="flex justify-between text-[11px] font-mono font-bold">
                              <span className="text-slate-600 uppercase">{item.label}</span>
                              <span className="text-[#141414]">{item.probability}%</span>
                            </div>
                            <div className="h-2 w-full bg-slate-100 border border-[#141414] rounded-none overflow-hidden">
                              <div
                                style={{ width: `${item.probability}%` }}
                                className={`h-full rounded-none ${
                                  idx === 0
                                    ? 'bg-[#141414]'
                                    : idx === 1
                                    ? 'bg-slate-500'
                                    : 'bg-slate-300'
                                }`}
                              ></div>
                            </div>
                          </div>
                        ))}
                      </div>
                    </div>
                  )}
                </div>
              ) : (
                <div className="flex flex-col items-center justify-center py-8 text-center space-y-3">
                  <HelpCircle className="w-10 h-10 text-slate-400" />
                  <p className="text-xs font-mono uppercase text-slate-500 max-w-[220px]">
                    Configure parameters and compile prediction above.
                  </p>
                </div>
              )}
            </div>
          </div>

          {/* Batch Predictions Section */}
          <div id="batch-prediction-section" className="p-6 border border-[#141414] bg-white rounded-none space-y-4">
            <div>
              <h3 className="text-xs font-bold uppercase tracking-widest text-[#141414] flex items-center gap-1.5 italic">
                <FileSpreadsheet className="w-4.5 h-4.5 text-[#141414]" />
                03. Batch Inference Engine
              </h3>
              <p className="text-xs text-slate-500 mt-1">
                Upload a test CSV or Excel spreadsheet to batch-predict values appended as new columns.
              </p>
            </div>

            <div className="relative border border-dashed border-[#141414] rounded-none p-6 text-center hover:bg-slate-50 transition cursor-pointer">
              <input
                type="file"
                accept=".csv, .xlsx, .xls"
                onChange={handleBatchFileChange}
                className="absolute inset-0 opacity-0 cursor-pointer"
                id="batch-uploader-input"
              />
              <div className="space-y-2">
                <Upload className="w-6 h-6 text-[#141414] mx-auto opacity-70" />
                <p className="text-xs font-bold uppercase font-mono text-[#141414]">
                  {batchFile ? batchFile.name : 'Upload test dataset'}
                </p>
                <p className="text-[10px] font-mono text-slate-400 uppercase">CSV or Excel spreadsheet</p>
              </div>
            </div>

            {batchLoading && (
              <div className="flex items-center gap-2 justify-center text-xs font-mono text-slate-500 py-2">
                <RefreshCw className="w-4 h-4 animate-spin text-slate-400" />
                <span>GENERATING BATCH PREDICTIONS...</span>
              </div>
            )}

            {batchError && (
              <div className="p-3 border border-red-600 bg-red-50 text-red-800 text-xs font-mono rounded-none">
                {batchError}
              </div>
            )}

            {batchOutputData && (
              <div className="space-y-3 pt-2">
                <div className="flex items-center gap-1.5 text-xs font-mono text-green-800 bg-green-50 p-3 rounded-none border border-green-600">
                  <Check className="w-4 h-4 text-green-600" />
                  <span>SUCCESS: PREDICTED {batchOutputData.length} RECORDS.</span>
                </div>
                
                <button
                  onClick={downloadBatchResults}
                  id="batch-download-button"
                  className="w-full py-3 border border-[#141414] bg-[#141414] hover:bg-transparent hover:text-[#141414] text-white font-bold text-xs uppercase tracking-wider rounded-none transition-colors flex items-center justify-center gap-1.5 cursor-pointer"
                >
                  <Download className="w-3.5 h-3.5" />
                  Download Predictions (Excel)
                </button>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
