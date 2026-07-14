/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 */

import React from 'react';
import { TrainingProgress } from '../types';
import { Cpu, Terminal, ArrowRight, CheckCircle, RefreshCw } from 'lucide-react';
import { motion } from 'motion/react';

interface ModelTrainerProps {
  progress: TrainingProgress;
  onProceedToCompare: () => void;
  onRestart: () => void;
}

export default function ModelTrainer({ progress, onProceedToCompare, onRestart }: ModelTrainerProps) {
  const { status, currentModel, logs } = progress;

  return (
    <div id="model-trainer-view" className="space-y-8 max-w-3xl mx-auto animate-fade-in text-[#141414]">
      {/* Visual Header */}
      <div className="text-center space-y-2">
        <div className="inline-flex p-3 border border-[#141414] bg-white text-[#141414] rounded-none mb-2 shadow-sm animate-pulse">
          <Cpu className="w-8 h-8" />
        </div>
        <h2 className="text-2xl font-bold font-display uppercase tracking-tight text-[#141414]">
          {status === 'preprocessing' && 'Preprocessing & Engineering Data...'}
          {status === 'training' && `Optimizing ${currentModel || 'Models'}...`}
          {status === 'evaluating' && 'Measuring Cross-Validation Performance...'}
          {status === 'completed' && 'Pipeline Trained successfully!'}
          {status === 'failed' && 'Training Failed'}
          {status === 'idle' && 'Waiting to start training...'}
        </h2>
        <p className="text-xs font-mono uppercase tracking-wider text-slate-500">
          DATA CLEANING • SCALING • FEATURE COUPLING • ENSEMBLE OPTIMIZATION
        </p>
      </div>

      {/* Main progress panel */}
      <div className="border border-[#141414] bg-white rounded-none overflow-hidden">
        {/* Progress bar */}
        <div className="h-2 w-full bg-slate-100 relative border-b border-[#141414]">
          <motion.div
            initial={{ width: '0%' }}
            animate={{
              width:
                status === 'preprocessing'
                  ? '25%'
                  : status === 'training'
                  ? '65%'
                  : status === 'evaluating'
                  ? '85%'
                  : status === 'completed'
                  ? '100%'
                  : '0%',
            }}
            transition={{ duration: 0.5, ease: 'easeInOut' }}
            className="h-full bg-[#141414]"
          ></motion.div>
        </div>

        {/* Live Terminal */}
        <div className="p-6 bg-slate-950 text-slate-300 font-mono text-xs leading-relaxed space-y-2.5 min-h-[300px] max-h-[400px] overflow-y-auto custom-scrollbar">
          <div className="flex items-center gap-2 text-slate-500 border-b border-slate-900 pb-3 mb-3">
            <Terminal className="w-4 h-4" />
            <span className="uppercase text-[10px] font-bold tracking-wider">Agent Compilation Console</span>
          </div>

          {logs.map((log, index) => {
            const isCompleted = log.startsWith('✅');
            const isFailed = log.startsWith('❌');
            const isHeader = log.startsWith('---') || log.startsWith('===');
            
            return (
              <motion.div
                initial={{ opacity: 0, x: -5 }}
                animate={{ opacity: 1, x: 0 }}
                transition={{ duration: 0.2 }}
                key={index}
                className={`${
                  isCompleted
                    ? 'text-emerald-400 font-bold'
                    : isFailed
                    ? 'text-red-400 font-bold'
                    : isHeader
                    ? 'text-blue-400 font-bold'
                    : 'text-slate-300'
                }`}
              >
                {log}
              </motion.div>
            );
          })}

          {status !== 'completed' && status !== 'failed' && (
            <div className="flex items-center gap-1.5 text-slate-500 animate-pulse mt-3">
              <RefreshCw className="w-3.5 h-3.5 animate-spin" />
              <span>Agent is calculating...</span>
            </div>
          )}
        </div>
      </div>

      {/* Finish actions */}
      {status === 'completed' && (
        <motion.div
          initial={{ opacity: 0, y: 10 }}
          animate={{ opacity: 1, y: 0 }}
          className="flex flex-col sm:flex-row sm:items-center sm:justify-between p-6 bg-white border border-[#141414] rounded-none gap-4"
        >
          <div className="flex items-center gap-2.5 text-[#141414] text-xs font-mono uppercase font-bold">
            <CheckCircle className="w-5 h-5 text-green-600 flex-shrink-0" />
            <span>Success: Multiple models trained. View side-by-side performance.</span>
          </div>
          <button
            onClick={onProceedToCompare}
            id="trainer-compare-button"
            className="px-6 py-3 border border-[#141414] bg-[#141414] hover:bg-transparent hover:text-[#141414] text-white text-xs font-bold uppercase tracking-wider rounded-none transition-colors flex items-center justify-center gap-2 group cursor-pointer"
          >
            Compare Performance
            <ArrowRight className="w-4 h-4 transition-transform group-hover:translate-x-1" />
          </button>
        </motion.div>
      )}

      {status === 'failed' && (
        <motion.div
          initial={{ opacity: 0, y: 10 }}
          animate={{ opacity: 1, y: 0 }}
          className="flex flex-col sm:flex-row sm:items-center sm:justify-between p-6 bg-white border border-[#141414] rounded-none gap-4"
        >
          <p className="text-[#141414] text-xs font-mono uppercase font-bold">
            Compilation Failed. Check dataset structure and size.
          </p>
          <button
            onClick={onRestart}
            id="trainer-restart-button"
            className="px-6 py-3 border border-[#141414] bg-[#141414] hover:bg-transparent hover:text-[#141414] text-white text-xs font-bold uppercase tracking-wider rounded-none transition-colors cursor-pointer"
          >
            Upload New File
          </button>
        </motion.div>
      )}
    </div>
  );
}
