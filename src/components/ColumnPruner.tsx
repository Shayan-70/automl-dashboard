/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 */

import React, { useState } from 'react';
import { DatasetSummary } from '../types';
import {
  Trash2,
  Check,
  AlertOctagon,
  ArrowRight,
  ArrowLeft,
  Columns,
  CheckSquare,
  Square,
  HelpCircle
} from 'lucide-react';

interface ColumnPrunerProps {
  summary: DatasetSummary;
  onProceed: (excluded: string[]) => void;
  onBack: () => void;
}

export default function ColumnPruner({
  summary,
  onProceed,
  onBack,
}: ColumnPrunerProps) {
  const { columns, targetColumn } = summary;

  // Selected columns to EXCLUDE/DROP
  const [selectedExclusions, setSelectedExclusions] = useState<string[]>([]);
  const [errorMsg, setErrorMsg] = useState<string | null>(null);
  const [confirmed, setConfirmed] = useState<boolean>(false);

  // Toggle selection for a column
  const handleToggleColumn = (colName: string) => {
    setErrorMsg(null);
    
    if (colName === targetColumn) {
      setErrorMsg(`⚠️ Error: The target column "${targetColumn}" cannot be dropped as it is the training label.`);
      return;
    }

    if (selectedExclusions.includes(colName)) {
      setSelectedExclusions(prev => prev.filter(c => c !== colName));
    } else {
      setSelectedExclusions(prev => [...prev, colName]);
    }
  };

  // Select all columns except the target
  const handleSelectAll = () => {
    setErrorMsg(null);
    const nonTargetCols = columns
      .map(c => c.name)
      .filter(name => name !== targetColumn);
    setSelectedExclusions(nonTargetCols);
  };

  // Deselect all columns
  const handleDeselectAll = () => {
    setErrorMsg(null);
    setSelectedExclusions([]);
  };

  // Attempt to confirm the selection
  const handleConfirm = () => {
    setErrorMsg(null);

    // Double check if target is in selected exclusions (just in case)
    if (targetColumn && selectedExclusions.includes(targetColumn)) {
      setErrorMsg(`⚠️ Severe Error: The target column "${targetColumn}" cannot be dropped from the dataset.`);
      return;
    }

    // Check if the user dropped all columns
    if (selectedExclusions.length >= columns.length - 1 && columns.length > 1) {
      setErrorMsg('⚠️ Error: You must keep at least one feature column (in addition to the target column) for model training.');
      return;
    }

    setConfirmed(true);
  };

  const handleProceedToTraining = () => {
    onProceed(selectedExclusions);
  };

  return (
    <div id="column-pruning-view" className="space-y-8 animate-fade-in text-[#141414]">
      {/* Visual Header */}
      <div className="flex flex-col md:flex-row md:items-center md:justify-between pb-6 border-b border-[#141414] gap-4">
        <div>
          <h2 className="text-xl font-bold font-display uppercase tracking-tight text-[#141414] flex items-center gap-2">
            <Columns className="w-5.5 h-5.5 text-[#141414]" />
            Manual Column-Removal Pipeline Step
          </h2>
          <p className="text-[11px] font-mono text-slate-500 uppercase tracking-wider mt-1">
            03. SELECT COLUMNS TO DISCARD OR IGNORE BEFORE COMPILATION
          </p>
        </div>
        
        <button
          onClick={onBack}
          className="px-4 py-2 border border-[#141414] bg-white text-[#141414] hover:bg-[#141414] hover:text-white transition-colors font-bold text-xs uppercase tracking-wider rounded-none self-start md:self-auto flex items-center gap-1.5 cursor-pointer"
        >
          <ArrowLeft className="w-4 h-4" />
          Back to Profile
        </button>
      </div>

      {errorMsg && (
        <div id="pruning-error" className="p-4 border border-red-600 bg-red-50 text-red-800 text-xs font-mono uppercase rounded-none flex gap-3 animate-pulse">
          <AlertOctagon className="w-5 h-5 text-red-600 flex-shrink-0" />
          <p className="leading-relaxed font-bold">{errorMsg}</p>
        </div>
      )}

      {/* Main interactive panel */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
        
        {/* Left column: Action description & checklist controls */}
        <div className="lg:col-span-1 space-y-6">
          <div className="p-6 bg-white border border-[#141414] rounded-none space-y-6">
            <h3 className="text-xs font-bold uppercase tracking-widest text-slate-500 italic">01. Pruning Directives</h3>
            <div className="text-xs space-y-4 leading-relaxed text-slate-600 font-mono">
              <p>
                Removing irrelevant, redundant, or leaking variables is standard practice in feature engineering. 
              </p>
              <p>
                Choose variables to exclude below. The target label column <span className="bg-yellow-400 text-black px-1 py-0.5 border border-black font-bold font-sans">{targetColumn}</span> is automatically preserved and locked.
              </p>
              <p className="text-[#141414] font-bold">
                Exclusions will persist consistently across training partition splits and real-time inference predictions.
              </p>
            </div>

            <div className="pt-4 border-t border-dashed border-[#141414] space-y-3">
              <button
                onClick={handleSelectAll}
                className="w-full py-2.5 border border-[#141414] bg-white hover:bg-neutral-50 text-[#141414] text-xs font-bold uppercase tracking-wider rounded-none transition-colors flex items-center justify-center gap-1.5 cursor-pointer"
              >
                <CheckSquare className="w-3.5 h-3.5" />
                Exclude All Features
              </button>
              
              <button
                onClick={handleDeselectAll}
                className="w-full py-2.5 border border-[#141414] bg-white hover:bg-neutral-50 text-[#141414] text-xs font-bold uppercase tracking-wider rounded-none transition-colors flex items-center justify-center gap-1.5 cursor-pointer"
              >
                <Square className="w-3.5 h-3.5" />
                Keep All Features
              </button>
            </div>
          </div>

          {/* Locked target banner */}
          <div className="p-5 border border-[#141414] bg-neutral-100 rounded-none space-y-1">
            <p className="text-[10px] font-mono font-bold text-slate-500 uppercase">LOCKED MODEL TARGET LABEL</p>
            <p className="text-sm font-bold font-mono text-[#141414] flex items-center gap-2">
              🎯 {targetColumn}
              <span className="text-[9px] font-bold px-1.5 py-0.5 border border-green-600 bg-green-50 text-green-800 uppercase rounded-none tracking-wider font-sans">Required</span>
            </p>
          </div>
        </div>

        {/* Right column: Checklist scroll area */}
        <div className="lg:col-span-2 bg-white border border-[#141414] rounded-none p-6 space-y-4">
          <div className="flex justify-between items-center pb-3 border-b border-[#141414]">
            <h3 className="text-xs font-bold uppercase tracking-widest text-slate-500 italic">02. Column Index Check</h3>
            <span className="text-[10px] font-mono bg-neutral-100 border border-[#141414] px-2 py-1 font-bold">
              DROPPING: {selectedExclusions.length} / {columns.length - 1} FEATURES
            </span>
          </div>

          <div className="max-h-[380px] overflow-y-auto divide-y divide-dashed divide-[#141414] pr-2">
            {columns.map(col => {
              const isTarget = col.name === targetColumn;
              const isExcluded = selectedExclusions.includes(col.name);
              
              return (
                <div
                  key={col.name}
                  onClick={() => handleToggleColumn(col.name)}
                  className={`py-4 px-3 flex items-center justify-between transition-colors cursor-pointer select-none ${
                    isTarget 
                      ? 'bg-neutral-50 cursor-not-allowed opacity-65'
                      : isExcluded 
                      ? 'bg-red-50/50 hover:bg-red-50' 
                      : 'hover:bg-neutral-50'
                  }`}
                >
                  <div className="flex items-center gap-4">
                    {/* Checkbox state */}
                    <div className="flex-shrink-0">
                      {isTarget ? (
                        <div className="w-5 h-5 border border-[#141414] bg-yellow-400 text-[#141414] flex items-center justify-center font-bold text-xs rounded-none">
                          ★
                        </div>
                      ) : isExcluded ? (
                        <div className="w-5 h-5 border border-red-600 bg-red-600 text-white flex items-center justify-center rounded-none font-bold">
                          <Check className="w-4 h-4 stroke-[3]" />
                        </div>
                      ) : (
                        <div className="w-5 h-5 border border-[#141414] bg-white rounded-none"></div>
                      )}
                    </div>

                    <div className="space-y-1">
                      <p className={`text-sm font-bold font-mono tracking-tight ${isExcluded ? 'line-through text-red-800' : 'text-[#141414]'}`}>
                        {col.name}
                      </p>
                      <div className="flex gap-2 items-center text-[10px] font-mono text-slate-500">
                        <span className="uppercase font-bold border border-neutral-300 px-1">{col.type}</span>
                        <span>•</span>
                        <span>{col.uniqueValuesCount} UNIQUES</span>
                        {col.missingCount > 0 && (
                          <>
                            <span>•</span>
                            <span className="text-red-600 font-bold">{col.missingPercentage.toFixed(1)}% MISSING</span>
                          </>
                        )}
                      </div>
                    </div>
                  </div>

                  <div className="text-right flex items-center gap-3">
                    {isTarget ? (
                      <span className="text-[10px] font-bold font-mono text-slate-500 uppercase tracking-wider bg-yellow-400/20 px-2 py-0.5 border border-yellow-500">
                        TARGET
                      </span>
                    ) : isExcluded ? (
                      <span className="text-[10px] font-bold font-mono text-red-800 uppercase tracking-wider bg-red-100 px-2 py-0.5 border border-red-400 flex items-center gap-1">
                        <Trash2 className="w-3 h-3" /> EXCLUDED
                      </span>
                    ) : (
                      <span className="text-[10px] font-mono text-slate-400 uppercase">
                        ACTIVE
                      </span>
                    )}
                  </div>
                </div>
              );
            })}
          </div>
        </div>
      </div>

      {/* Confirmation and proceed actions */}
      <div id="pruner-actions" className="p-6 bg-white border border-[#141414] rounded-none space-y-4">
        {!confirmed ? (
          <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-4">
            <div className="flex items-center gap-2 text-[#141414] text-xs font-mono">
              <HelpCircle className="w-4 h-4" />
              <span className="uppercase">VALIDATE THE COLUMN SELECTION TO FINALIZE EXCLUSIONS.</span>
            </div>
            
            <button
              onClick={handleConfirm}
              id="confirm-exclusions-button"
              className="px-6 py-3 border border-[#141414] bg-[#141414] hover:bg-transparent hover:text-[#141414] text-white text-xs font-bold uppercase tracking-wider rounded-none transition-colors flex items-center justify-center gap-2 cursor-pointer"
            >
              Confirm Columns & Exclude
              <ArrowRight className="w-4 h-4" />
            </button>
          </div>
        ) : (
          <div className="space-y-4">
            {/* Show short confirmation message */}
            <div id="pruning-confirmation-banner" className="p-4 border border-green-600 bg-green-50 text-green-800 text-xs font-mono uppercase rounded-none flex items-center gap-2">
              <Check className="w-4 h-4 text-green-600 stroke-[3]" />
              <span>
                ✅ {selectedExclusions.length} columns removed: [{selectedExclusions.length > 0 ? selectedExclusions.join(', ') : 'None'}]
              </span>
            </div>

            <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-4">
              <button
                onClick={() => setConfirmed(false)}
                className="text-xs font-bold font-mono uppercase text-slate-500 hover:text-[#141414] underline cursor-pointer"
              >
                Modify exclusions
              </button>

              <button
                onClick={handleProceedToTraining}
                id="pruner-proceed-button"
                className="px-6 py-3 border border-green-600 bg-green-600 hover:bg-transparent hover:text-green-800 text-white text-xs font-bold uppercase tracking-wider rounded-none transition-colors flex items-center justify-center gap-2 cursor-pointer"
              >
                Proceed to Model Preprocessing & Training
                <ArrowRight className="w-4 h-4" />
              </button>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
