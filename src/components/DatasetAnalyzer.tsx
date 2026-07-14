/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 */

import React from 'react';
import { ColumnInfo, DatasetSummary } from '../types';
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  Cell,
} from 'recharts';
import {
  Eye,
  CheckCircle,
  AlertTriangle,
  FileText,
  TrendingUp,
  Settings,
  ArrowRight,
  Sparkles,
} from 'lucide-react';

interface DatasetAnalyzerProps {
  summary: DatasetSummary;
  onTargetChanged: (newTarget: string) => void;
  onProblemTypeChanged: (newType: 'classification' | 'regression') => void;
  onProceedToTraining: () => void;
}

export default function DatasetAnalyzer({
  summary,
  onTargetChanged,
  onProblemTypeChanged,
  onProceedToTraining,
}: DatasetAnalyzerProps) {
  const { rowCount, columnCount, columns, targetColumn, problemType, classImbalance } = summary;

  // Find warnings
  const warnings: string[] = [];
  columns.forEach(col => {
    if (col.missingPercentage > 30) {
      warnings.push(`Column "${col.name}" has high missingness (${col.missingPercentage.toFixed(1)}%). Standard imputation will apply, but results may vary.`);
    }
    if (col.uniqueValuesCount === 1) {
      warnings.push(`Column "${col.name}" has only 1 unique value and will be automatically filtered out during training as it holds zero variance.`);
    }
  });

  if (rowCount < 50) {
    warnings.push('This dataset is quite small (under 50 rows). Machine learning models may overfit; simpler models like Linear/Logistic regression are highly recommended.');
  }

  if (problemType === 'classification' && classImbalance && classImbalance.length > 0) {
    const minPct = classImbalance[classImbalance.length - 1].percentage;
    const maxPct = classImbalance[0].percentage;
    if (maxPct / minPct > 3) {
      warnings.push(`Severe class imbalance detected. The majority class is ${maxPct}% vs the minority class at ${minPct}%. Gini and Boosting models will automatically adjust target thresholds.`);
    }
  }

  // Pre-selected colors for charts matching Swiss/brutalist theme
  const CHART_COLORS = ['#141414', '#3b82f6', '#10b981', '#f59e0b', '#ef4444', '#8b5cf6'];

  return (
    <div id="dataset-analyzer-view" className="space-y-8 animate-fade-in text-[#141414]">
      {/* 1. Header Overview & Configuration */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        <div className="p-6 bg-white rounded-none border border-[#141414] flex flex-col justify-between lg:col-span-1">
          <div className="space-y-4">
            <h3 className="text-xs font-bold uppercase tracking-widest text-slate-500 italic">01. Dataset Snapshot</h3>
            <div className="space-y-3">
              <div className="flex justify-between items-end border-b border-dashed border-[#141414] pb-1">
                <span className="text-xs uppercase font-mono">Total Rows:</span>
                <span className="text-sm font-mono font-bold">{rowCount}</span>
              </div>
              <div className="flex justify-between items-end border-b border-dashed border-[#141414] pb-1">
                <span className="text-xs uppercase font-mono">Features:</span>
                <span className="text-sm font-mono font-bold">{columnCount}</span>
              </div>
              <div className="flex justify-between items-end border-b border-dashed border-[#141414] pb-1">
                <span className="text-xs uppercase font-mono">Inferred Task:</span>
                <span className="text-[10px] px-2 py-0.5 bg-blue-100 text-blue-800 rounded-none font-bold uppercase tracking-wider font-mono">
                  {problemType}
                </span>
              </div>
            </div>
          </div>
          
          <div className="mt-6 border-t border-[#141414] pt-4 space-y-2">
            <div className="flex items-center justify-between text-xs font-mono">
              <span>PROFILER STATUS:</span>
              <span className="flex items-center gap-1 text-green-600 font-bold uppercase">
                <CheckCircle className="w-3.5 h-3.5" /> Ready
              </span>
            </div>
          </div>
        </div>

        {/* Target and Problem configuration */}
        <div className="p-6 bg-white rounded-none border border-[#141414] lg:col-span-2 space-y-5">
          <div className="flex items-center gap-2 mb-1">
            <Settings className="w-4.5 h-4.5 text-[#141414]" />
            <h3 className="text-xs font-bold uppercase tracking-widest text-[#141414] italic">02. AI Workflow Customization</h3>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            <div className="space-y-2">
              <label className="text-xs font-bold uppercase tracking-wider text-[#141414] font-mono">
                Target Variable (Label Column)
              </label>
              <select
                id="analyzer-target-select"
                value={targetColumn || ''}
                onChange={(e) => onTargetChanged(e.target.value)}
                className="w-full bg-white border border-[#141414] rounded-none px-4 py-3 text-slate-800 focus:outline-none text-xs font-mono"
              >
                {columns.map(col => (
                  <option key={col.name} value={col.name} disabled={!col.isTargetPossible}>
                    {col.name.toUpperCase()} {!col.isTargetPossible ? '(INVALID)' : ''}
                  </option>
                ))}
              </select>
              <p className="text-[11px] text-slate-500 font-mono">
                The target column which our autonomous models will predict.
              </p>
            </div>

            <div className="space-y-2">
              <label className="text-xs font-bold uppercase tracking-wider text-[#141414] font-mono">
                Problem Type (Inferred)
              </label>
              <div className="flex gap-2">
                <button
                  type="button"
                  id="analyzer-problem-classification"
                  onClick={() => onProblemTypeChanged('classification')}
                  className={`flex-1 py-3 text-xs font-bold uppercase tracking-wider rounded-none border border-[#141414] transition ${
                    problemType === 'classification'
                      ? 'bg-[#141414] text-white'
                      : 'bg-white text-[#141414] hover:bg-slate-50'
                  }`}
                >
                  Classification
                </button>
                <button
                  type="button"
                  id="analyzer-problem-regression"
                  onClick={() => onProblemTypeChanged('regression')}
                  className={`flex-1 py-3 text-xs font-bold uppercase tracking-wider rounded-none border border-[#141414] transition ${
                    problemType === 'regression'
                      ? 'bg-[#141414] text-white'
                      : 'bg-white text-[#141414] hover:bg-slate-50'
                  }`}
                >
                  Regression
                </button>
              </div>
              <p className="text-[11px] text-slate-500 font-mono">
                {problemType === 'classification'
                  ? 'Predicting discrete classes / categories.'
                  : 'Predicting continuous scalar numbers.'}
              </p>
            </div>
          </div>
        </div>
      </div>

      {/* 2. Target Distribution (Only for classification) */}
      {problemType === 'classification' && classImbalance && classImbalance.length > 0 && (
          <div id="target-distribution-chart" className="p-6 border border-[#141414] bg-white rounded-none">
            <div className="flex items-center justify-between mb-4">
              <h4 className="text-xs font-bold uppercase tracking-widest text-[#141414] italic flex items-center gap-2">
                <TrendingUp className="w-5 h-5 text-[#141414]" />
                03. Class Distribution: "{targetColumn?.toUpperCase()}"
              </h4>
            </div>
            <div className="h-64 w-full">
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={classImbalance} margin={{ top: 20, right: 30, left: 20, bottom: 5 }}>
                  <CartesianGrid strokeDasharray="3 3" vertical={false} />
                  <XAxis dataKey="label" stroke="#141414" className="text-xs font-mono" />
                  <YAxis stroke="#141414" unit="%" className="text-xs font-mono" />
                  <Tooltip
                    formatter={(value: any, name: any, props: any) => [
                      `${value}% (${props.payload.count} rows)`,
                      'Frequency',
                    ]}
                    contentStyle={{
                      backgroundColor: '#141414',
                      color: '#ffffff',
                      border: '1px solid #141414',
                      borderRadius: '0px',
                      fontSize: '11px',
                      fontFamily: 'monospace',
                    }}
                  />
                  <Bar dataKey="percentage" radius={[0, 0, 0, 0]} maxBarSize={60}>
                    {classImbalance.map((entry, index) => (
                      <Cell key={`cell-${index}`} fill={CHART_COLORS[index % CHART_COLORS.length]} />
                    ))}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            </div>
          </div>
      )}

      {/* 3. Alerts & Warnings */}
      {warnings.length > 0 && (
        <div id="profiler-warnings" className="p-5 border border-[#141414] bg-yellow-100 text-[#141414] rounded-none flex gap-3.5">
          <AlertTriangle className="w-5.5 h-5.5 text-yellow-600 flex-shrink-0 mt-0.5" />
          <div className="space-y-2">
            <h4 className="font-bold text-xs uppercase tracking-wider font-mono">
              Data Science Agent Alerts ({warnings.length})
            </h4>
            <ul className="list-disc pl-4 text-xs space-y-1.5 leading-relaxed font-mono">
              {warnings.map((warn, i) => (
                <li key={i}>{warn}</li>
              ))}
            </ul>
          </div>
        </div>
      )}

      {/* 4. Feature Profile Table */}
      <div id="feature-profile-table-container" className="border border-[#141414] bg-white rounded-none overflow-hidden">
        <div className="p-6 border-b border-[#141414] flex items-center gap-2">
          <FileText className="w-5 h-5 text-[#141414]" />
          <h4 className="text-xs font-bold uppercase tracking-widest text-[#141414] italic">04. Dataset Column Profiler</h4>
        </div>

        <div className="overflow-x-auto">
          <table className="w-full text-left text-xs border-collapse">
            <thead>
              <tr className="bg-[#141414] text-white text-[10px] uppercase font-bold tracking-widest font-mono">
                <th className="px-6 py-4">Column</th>
                <th className="px-6 py-4">Inferred Type</th>
                <th className="px-6 py-4">Missingness</th>
                <th className="px-6 py-4">Unique Values</th>
                <th className="px-6 py-4">Mean / Std</th>
                <th className="px-6 py-4">Min / Max</th>
                <th className="px-6 py-4">Outliers (IQR)</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-[#141414]/30">
              {columns.map(col => {
                const isTarget = col.name === targetColumn;
                return (
                  <tr
                    key={col.name}
                    className={`hover:bg-slate-50 transition-colors ${
                      isTarget ? 'bg-yellow-100/40 font-semibold' : ''
                    }`}
                  >
                    <td className="px-6 py-4 font-mono font-bold text-[#141414] flex items-center gap-1.5 uppercase">
                      {col.name}
                      {isTarget && (
                        <span className="text-[9px] font-bold px-1.5 py-0.5 bg-[#141414] text-white rounded-none tracking-widest">
                          TARGET
                        </span>
                      )}
                    </td>
                    <td className="px-6 py-4">
                      <span className={`text-[10px] px-2 py-0.5 rounded-none font-bold uppercase font-mono ${
                        col.type === 'numeric'
                          ? 'bg-blue-100 text-blue-800'
                          : col.type === 'categorical'
                          ? 'bg-purple-100 text-purple-800'
                          : 'bg-neutral-100 text-neutral-800'
                      }`}>
                        {col.type}
                      </span>
                    </td>
                    <td className="px-6 py-4 font-mono">
                      {col.missingCount > 0 ? (
                        <span className="text-red-600 font-bold">
                          {col.missingCount} ({col.missingPercentage.toFixed(1)}%)
                        </span>
                      ) : (
                        <span className="text-green-700">0%</span>
                      )}
                    </td>
                    <td className="px-6 py-4 font-mono">
                      {col.uniqueValuesCount}
                    </td>
                    <td className="px-6 py-4 font-mono text-slate-500">
                      {col.type === 'numeric' ? (
                        <>
                          {col.mean} <span className="text-slate-300">/</span> {col.std}
                        </>
                      ) : (
                        '—'
                      )}
                    </td>
                    <td className="px-6 py-4 font-mono text-slate-500">
                      {col.type === 'numeric' ? (
                        <>
                          {col.min} <span className="text-slate-300">/</span> {col.max}
                        </>
                      ) : (
                        '—'
                      )}
                    </td>
                    <td className="px-6 py-4 font-mono">
                      {col.type === 'numeric' ? (
                        col.outliersCount && col.outliersCount > 0 ? (
                          <span className="text-amber-600 font-bold">
                            {col.outliersCount}
                          </span>
                        ) : (
                          <span className="text-slate-400">0</span>
                        )
                      ) : (
                        '—'
                      )}
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </div>

      {/* Proceed buttons */}
      <div id="proceed-actions" className="flex flex-col sm:flex-row sm:items-center sm:justify-between p-6 bg-white border border-[#141414] rounded-none gap-4">
        <div className="flex items-center gap-2 text-[#141414] text-xs font-mono">
          <Eye className="w-4 h-4" />
          <span className="uppercase font-bold">CONFIGURATION COMPILED. READY TO COMPILE ML MODELS.</span>
        </div>
        <button
          onClick={onProceedToTraining}
          id="analyzer-proceed-button"
          className="px-6 py-3 border border-[#141414] bg-[#141414] hover:bg-transparent hover:text-[#141414] text-white text-xs font-bold uppercase tracking-wider rounded-none transition-colors flex items-center justify-center gap-2 group cursor-pointer"
        >
          Compile Machine Learning Models
          <ArrowRight className="w-4 h-4 transition-transform group-hover:translate-x-1" />
        </button>
      </div>
    </div>
  );
}
