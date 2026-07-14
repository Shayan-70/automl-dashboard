/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 */

import React, { useState } from 'react';
import { ModelResult } from '../types';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';
import { Award, Layers, TrendingUp, CheckCircle, ArrowRight, Play } from 'lucide-react';
import { motion } from 'motion/react';

interface ModelComparerProps {
  results: ModelResult[];
  problemType: 'classification' | 'regression';
  selectedModelId: string | null;
  onSelectModel: (modelId: string) => void;
  onProceedToPredict: () => void;
}

export default function ModelComparer({
  results,
  problemType,
  selectedModelId,
  onSelectModel,
  onProceedToPredict,
}: ModelComparerProps) {
  // Determine primary sorting metric to find the best model
  // Classification: F1-score (or Accuracy)
  // Regression: R-squared (R2) - higher is better
  const getSortedResults = () => {
    return [...results].sort((a, b) => {
      if (problemType === 'classification') {
        return (b.metrics.f1 ?? 0) - (a.metrics.f1 ?? 0);
      } else {
        return (b.metrics.r2 ?? 0) - (a.metrics.r2 ?? 0);
      }
    });
  };

  const sorted = getSortedResults();
  const bestModelId = sorted[0]?.modelId;

  // Chart data formatting
  const chartData = sorted.map(res => {
    if (problemType === 'classification') {
      return {
        name: res.modelName,
        'Accuracy (%)': parseFloat(((res.metrics.accuracy ?? 0) * 100).toFixed(1)),
        'F1-Score (%)': parseFloat(((res.metrics.f1 ?? 0) * 100).toFixed(1)),
        'ROC-AUC (%)': parseFloat(((res.metrics.roc_auc ?? 0) * 100).toFixed(1)),
      };
    } else {
      return {
        name: res.modelName,
        'R-squared (%)': parseFloat(((res.metrics.r2 ?? 0) * 100).toFixed(1)),
        'MAE (scaled)': parseFloat((res.metrics.mae ?? 0).toFixed(3)),
        'RMSE (scaled)': parseFloat((res.metrics.rmse ?? 0).toFixed(3)),
      };
    }
  });

  return (
    <div id="model-comparer-view" className="space-y-8 animate-fade-in text-[#141414]">
      {/* Best Model Banner */}
      <div className="relative overflow-hidden p-6 md:p-8 border border-[#141414] rounded-none bg-white flex flex-col md:flex-row items-center gap-6 justify-between">
        <div className="space-y-2 text-center md:text-left">
          <div className="inline-flex items-center gap-1.5 px-3 py-1 border border-[#141414] bg-yellow-400 text-[#141414] font-bold text-[10px] uppercase tracking-wider rounded-none">
            <Award className="w-4 h-4" /> Champion Inference Model Inferred
          </div>
          <h2 className="text-2xl font-bold font-display uppercase tracking-tight text-[#141414] mt-2">
            {results.find(r => r.modelId === bestModelId)?.modelName || 'Optimized Ensemble'}
          </h2>
          <p className="text-xs font-mono uppercase tracking-wider text-slate-500 max-w-xl">
            {problemType === 'classification'
              ? `ACCURACY SCORE: ${((sorted[0]?.metrics.accuracy ?? 0) * 100).toFixed(1)}% • F1-SCORE: ${((sorted[0]?.metrics.f1 ?? 0) * 100).toFixed(1)}% VALIDATION`
              : `R-SQUARED: ${((sorted[0]?.metrics.r2 ?? 0) * 100).toFixed(1)}% • OPTIMAL SCALE RECOVERY`}
          </p>
        </div>

        <button
          onClick={onProceedToPredict}
          id="comparer-predict-button"
          className="px-6 py-3.5 border border-[#141414] bg-[#141414] hover:bg-transparent hover:text-[#141414] text-white font-bold text-xs uppercase tracking-wider rounded-none transition-colors flex items-center gap-2 group whitespace-nowrap cursor-pointer"
        >
          Deploy Predictions console
          <ArrowRight className="w-4 h-4 transition-transform group-hover:translate-x-1" />
        </button>
      </div>

      {/* Side-by-Side Charts */}
      <div id="comparer-charts" className="grid grid-cols-1 gap-6">
        <div className="p-6 border border-[#141414] bg-white rounded-none">
          <h3 className="text-xs font-bold uppercase tracking-widest text-slate-500 mb-6 flex items-center gap-2 italic">
            <TrendingUp className="w-5 h-5 text-[#141414]" />
            02. Validation Performance Evaluation
          </h3>
          <div className="h-80 w-full">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={chartData} margin={{ top: 20, right: 30, left: 10, bottom: 5 }}>
                <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="#e4e3e0" />
                <XAxis dataKey="name" stroke="#141414" className="text-xs font-mono" />
                <YAxis stroke="#141414" className="text-xs font-mono" />
                <Tooltip
                  contentStyle={{
                    backgroundColor: '#141414',
                    color: '#ffffff',
                    border: '1px solid #141414',
                    borderRadius: '0px',
                    fontSize: '11px',
                    fontFamily: 'monospace',
                  }}
                />
                <Legend wrapperStyle={{ fontSize: '11px', fontFamily: 'monospace', paddingTop: '10px' }} />
                {problemType === 'classification' ? (
                  <>
                    <Bar dataKey="Accuracy (%)" fill="#141414" radius={[0, 0, 0, 0]} />
                    <Bar dataKey="F1-Score (%)" fill="#3b82f6" radius={[0, 0, 0, 0]} />
                    <Bar dataKey="ROC-AUC (%)" fill="#10b981" radius={[0, 0, 0, 0]} />
                  </>
                ) : (
                  <>
                    <Bar dataKey="R-squared (%)" fill="#141414" radius={[0, 0, 0, 0]} />
                    <Bar dataKey="MAE (scaled)" fill="#f59e0b" radius={[0, 0, 0, 0]} />
                    <Bar dataKey="RMSE (scaled)" fill="#ef4444" radius={[0, 0, 0, 0]} />
                  </>
                )}
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>
      </div>

      {/* Model Selection Cards */}
      <div id="comparer-cards" className="space-y-4">
        <div className="flex items-center gap-2 text-[#141414]">
          <Layers className="w-4.5 h-4.5 text-[#141414] opacity-70" />
          <h3 className="text-xs font-bold font-mono uppercase tracking-widest text-slate-500 italic">03. Candidate Model Selection</h3>
        </div>

        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-5 gap-4">
          {sorted.map(res => {
            const isBest = res.modelId === bestModelId;
            const isSelected = res.modelId === selectedModelId;
            return (
              <div
                key={res.modelId}
                onClick={() => onSelectModel(res.modelId)}
                className={`p-6 border transition-all cursor-pointer flex flex-col justify-between relative overflow-hidden rounded-none ${
                  isSelected
                    ? 'border-2 border-[#141414] bg-neutral-50 shadow-sm'
                    : 'border border-[#141414] bg-white hover:bg-slate-50'
                }`}
              >
                {/* Badges */}
                <div className="flex items-center justify-between mb-4">
                  <span className="text-xs font-bold text-slate-500 font-mono">
                    {res.modelId.toUpperCase()}
                  </span>
                  <div className="flex gap-1.5">
                    {isBest && (
                      <span className="text-[9px] font-bold px-2 py-0.5 bg-yellow-400 text-[#141414] border border-[#141414] rounded-none uppercase tracking-wider">
                        CHAMP
                      </span>
                    )}
                    {isSelected && (
                      <span className="text-[9px] font-bold px-2 py-0.5 bg-[#141414] text-white rounded-none uppercase tracking-wider">
                        DEPLOYED
                      </span>
                    )}
                  </div>
                </div>

                <div className="space-y-1">
                  <h4 className="font-bold text-[#141414] font-display text-base uppercase tracking-tight">
                    {res.modelName}
                  </h4>
                  <p className="text-[10px] font-mono text-slate-400 uppercase">
                    Took {res.trainingTimeMs}ms to compile
                  </p>
                </div>

                {/* Metrics detail */}
                <div className="mt-6 border-t border-dashed border-[#141414] pt-4 space-y-2">
                  {problemType === 'classification' ? (
                    <>
                      <div className="flex justify-between text-xs font-mono">
                        <span className="text-slate-500">ACCURACY:</span>
                        <span className="font-bold text-[#141414]">
                          {((res.metrics.accuracy ?? 0) * 100).toFixed(1)}%
                        </span>
                      </div>
                      <div className="flex justify-between text-xs font-mono">
                        <span className="text-slate-500">MACRO F1:</span>
                        <span className="font-bold text-[#141414]">
                          {((res.metrics.f1 ?? 0) * 100).toFixed(1)}%
                        </span>
                      </div>
                      <div className="flex justify-between text-xs font-mono">
                        <span className="text-slate-500">ROC-AUC:</span>
                        <span className="font-bold text-[#141414]">
                          {((res.metrics.roc_auc ?? 0) * 100).toFixed(1)}%
                        </span>
                      </div>
                    </>
                  ) : (
                    <>
                      <div className="flex justify-between text-xs font-mono">
                        <span className="text-slate-500">R-SQUARED:</span>
                        <span className="font-bold text-[#141414]">
                          {((res.metrics.r2 ?? 0) * 100).toFixed(1)}%
                        </span>
                      </div>
                      <div className="flex justify-between text-xs font-mono">
                        <span className="text-slate-500">MAE Score:</span>
                        <span className="font-bold text-[#141414]">
                          {(res.metrics.mae ?? 0).toFixed(3)}
                        </span>
                      </div>
                      <div className="flex justify-between text-xs font-mono">
                        <span className="text-slate-500">RMSE Score:</span>
                        <span className="font-bold text-[#141414]">
                          {(res.metrics.rmse ?? 0).toFixed(3)}
                        </span>
                      </div>
                    </>
                  )}
                </div>

                <div className="mt-5 pt-3 border-t border-dashed border-[#141414]">
                  <span className={`text-xs font-bold font-mono uppercase flex items-center gap-1.5 ${
                    isSelected ? 'text-[#141414]' : 'text-slate-400'
                  }`}>
                    <CheckCircle className={`w-4 h-4 ${isSelected ? 'opacity-100' : 'opacity-30'}`} />
                    {isSelected ? 'Active for Inference' : 'Select candidate'}
                  </span>
                </div>
              </div>
            );
          })}
        </div>
      </div>
    </div>
  );
}
