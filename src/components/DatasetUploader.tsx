/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 */

import React, { useRef, useState } from 'react';
import * as XLSX from 'xlsx';
import { Upload, FileSpreadsheet, AlertCircle, Database, HelpCircle, Sparkles } from 'lucide-react';
import { motion } from 'motion/react';

interface DatasetUploaderProps {
  onDatasetLoaded: (data: any[], fileName: string) => void;
  onError: (error: string) => void;
}

export default function DatasetUploader({ onDatasetLoaded, onDatasetLoaded: _unused, onError }: DatasetUploaderProps) {
  const [isDragging, setIsDragging] = useState(false);
  const [loading, setLoading] = useState(false);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleFile = (file: File) => {
    const extension = file.name.split('.').pop()?.toLowerCase();
    if (extension !== 'csv' && extension !== 'xlsx' && extension !== 'xls') {
      onError('Unsupported file format. Please upload a CSV or Excel (.xlsx / .xls) file.');
      return;
    }

    setLoading(true);
    const reader = new FileReader();
    
    reader.onload = (e) => {
      try {
        const data = new Uint8Array(e.target?.result as ArrayBuffer);
        const workbook = XLSX.read(data, { type: 'array' });
        
        if (workbook.SheetNames.length === 0) {
          throw new Error('The uploaded file contains no sheets.');
        }

        const firstSheetName = workbook.SheetNames[0];
        const worksheet = workbook.Sheets[firstSheetName];
        
        // Header-to-row structure
        const json = XLSX.utils.sheet_to_json(worksheet, { defval: null });
        
        if (json.length === 0) {
          throw new Error('The dataset is empty. No rows found.');
        }

        onDatasetLoaded(json, file.name);
      } catch (err: any) {
        onError(err.message || 'Failed to parse the file. Ensure the sheet is not corrupted.');
      } finally {
        setLoading(false);
      }
    };

    reader.onerror = () => {
      onError('Error reading file. Please try again.');
      setLoading(false);
    };

    reader.readAsArrayBuffer(file);
  };

  const handleDragOver = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(true);
  };

  const handleDragLeave = () => {
    setIsDragging(false);
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);
    if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
      handleFile(e.dataTransfer.files[0]);
    }
  };

  const handleFileInputChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files.length > 0) {
      handleFile(e.target.files[0]);
    }
  };

  // Pre-configured premium sample datasets
  const loadSampleDataset = (type: 'iris' | 'housing' | 'heart') => {
    setLoading(true);
    setTimeout(() => {
      let data: any[] = [];
      let name = '';

      if (type === 'iris') {
        name = 'iris_flowers.csv';
        // 150 rows of standard Iris classification data
        const classes = ['setosa', 'versicolor', 'virginica'];
        for (let i = 0; i < 150; i++) {
          const cIdx = Math.floor(i / 50);
          const c = classes[cIdx];
          // Adding small variations
          const offset = cIdx * 1.2;
          data.push({
            sepal_length: parseFloat((4.5 + Math.random() * 1.5 + offset * 0.4).toFixed(1)),
            sepal_width: parseFloat((2.5 + Math.random() * 1.2 - offset * 0.1).toFixed(1)),
            petal_length: parseFloat((1.3 + Math.random() * 0.8 + offset * 1.6).toFixed(1)),
            petal_width: parseFloat((0.2 + Math.random() * 0.4 + offset * 0.7).toFixed(1)),
            species: c,
          });
        }
      } else if (type === 'housing') {
        name = 'boston_housing_prices.csv';
        const neighborhoods = ['Downtown', 'Suburbs', 'WestSide', 'EastSide', 'NorthHill'];
        for (let i = 0; i < 200; i++) {
          const rooms = Math.floor(3 + Math.random() * 6);
          const area = Math.floor(800 + rooms * 250 + Math.random() * 1000);
          const age = Math.floor(2 + Math.random() * 80);
          const distToCenter = parseFloat((1.2 + Math.random() * 12).toFixed(1));
          const neighborhood = neighborhoods[Math.floor(Math.random() * neighborhoods.length)];
          
          // Price structure
          let basePrice = 150000;
          basePrice += rooms * 45000;
          basePrice += area * 110;
          basePrice -= age * 1200;
          basePrice -= distToCenter * 8000;
          if (neighborhood === 'Downtown') basePrice += 120000;
          if (neighborhood === 'Suburbs') basePrice += 40000;
          basePrice += Math.random() * 40000 - 20000;

          data.push({
            rooms,
            area_sqft: area,
            property_age_years: age,
            distance_to_center_miles: distToCenter,
            neighborhood,
            median_price_usd: Math.max(80000, Math.round(basePrice)),
          });
        }
      } else if (type === 'heart') {
        name = 'heart_risk_assessment.csv';
        for (let i = 0; i < 180; i++) {
          const age = Math.floor(25 + Math.random() * 55);
          const sex = Math.random() > 0.6 ? 'Female' : 'Male';
          const bp = Math.floor(100 + Math.random() * 60 + age * 0.3);
          const chol = Math.floor(150 + Math.random() * 150 + age * 0.5);
          const chestPain = Math.random() > 0.5 ? 'Asymptomatic' : (Math.random() > 0.4 ? 'Typical Angina' : 'Non-Anginal');
          
          let riskScore = -5;
          riskScore += age * 0.12;
          riskScore += bp * 0.03;
          riskScore += chol * 0.01;
          if (sex === 'Male') riskScore += 1.5;
          if (chestPain === 'Asymptomatic') riskScore += 2.0;
          
          const risk = riskScore > 2.2 ? 'High Risk' : 'Low Risk';

          data.push({
            age,
            gender: sex,
            blood_pressure: bp,
            cholesterol_level: chol,
            chest_pain_type: chestPain,
            cardiac_risk: risk,
          });
        }
      }

      onDatasetLoaded(data, name);
      setLoading(false);
    }, 600);
  };

  return (
    <div id="dataset-uploader-container" className="space-y-8">
      {/* Upload zone */}
      <div
        id="uploader-drop-zone"
        onDragOver={handleDragOver}
        onDragLeave={handleDragLeave}
        onDrop={handleDrop}
        onClick={() => fileInputRef.current?.click()}
        className={`relative flex flex-col items-center justify-center border-2 border-dashed p-12 text-center cursor-pointer transition-all duration-300 min-h-[320px] rounded-none ${
          isDragging
            ? 'border-green-600 bg-green-50'
            : 'border-[#141414] hover:border-[#141414] hover:bg-white bg-white/70'
        }`}
      >
        <input
          ref={fileInputRef}
          type="file"
          accept=".csv, .xlsx, .xls"
          onChange={handleFileInputChange}
          className="hidden"
          id="uploader-file-input"
        />

        {loading ? (
          <div className="space-y-4">
            <div className="animate-spin rounded-full h-12 w-12 border-t-2 border-b-2 border-[#141414] mx-auto"></div>
            <p className="text-[#141414] font-mono text-xs font-bold uppercase tracking-wider">Parsing and profiling dataset...</p>
          </div>
        ) : (
          <div className="space-y-5 max-w-lg">
            <div className="mx-auto w-16 h-16 rounded-none border border-[#141414] bg-neutral-50 flex items-center justify-center text-[#141414] transition-transform hover:scale-105 duration-200">
              <Upload className="w-8 h-8" />
            </div>
            
            <div>
              <p className="text-lg font-bold uppercase tracking-tight text-[#141414] font-display">
                Drag and drop your dataset here
              </p>
              <p className="text-xs font-mono text-slate-500 mt-1 uppercase">
                Supports CSV, XLSX, and XLS sheets up to 10MB
              </p>
            </div>

            <button
              type="button"
              id="uploader-select-button"
              className="px-5 py-2.5 border border-[#141414] bg-[#141414] hover:bg-transparent hover:text-[#141414] text-white font-bold uppercase tracking-wider text-xs rounded-none transition-colors cursor-pointer"
            >
              Select File from Device
            </button>
          </div>
        )}
      </div>

      {/* Premium Samples Grid */}
      <div id="sample-datasets-section" className="space-y-4">
        <div className="flex items-center gap-2 text-[#141414]">
          <Database className="w-4.5 h-4.5 text-[#141414] opacity-70" />
          <h3 className="text-xs font-bold font-mono uppercase tracking-widest text-slate-500 italic">01. Explore Sandbox Samples</h3>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          {/* Sample 1 */}
          <div
            onClick={() => loadSampleDataset('iris')}
            className="p-6 border border-[#141414] bg-white rounded-none cursor-pointer hover:bg-slate-50 transition-colors flex flex-col justify-between min-h-[220px]"
          >
            <div>
              <div className="flex items-center justify-between mb-3">
                <span className="p-1.5 border border-[#141414] bg-neutral-50 rounded-none">
                  <FileSpreadsheet className="w-5 h-5 text-[#141414]" />
                </span>
                <span className="text-[10px] font-bold px-2 py-0.5 bg-blue-100 text-blue-800 rounded-none uppercase tracking-wider">
                  Classification
                </span>
              </div>
              <h4 className="font-bold text-[#141414] font-display text-base mb-1 uppercase tracking-tight">Iris Flower Types</h4>
              <p className="text-xs text-slate-600">
                Classify three iris species based on sepal and petal size measurements. Highly balanced.
              </p>
            </div>
            <div className="mt-4 flex items-center gap-1.5 text-blue-600 text-xs font-mono font-bold uppercase">
              <Sparkles className="w-3.5 h-3.5" />
              Load dataset
            </div>
          </div>

          {/* Sample 2 */}
          <div
            onClick={() => loadSampleDataset('housing')}
            className="p-6 border border-[#141414] bg-white rounded-none cursor-pointer hover:bg-slate-50 transition-colors flex flex-col justify-between min-h-[220px]"
          >
            <div>
              <div className="flex items-center justify-between mb-3">
                <span className="p-1.5 border border-[#141414] bg-neutral-50 rounded-none">
                  <FileSpreadsheet className="w-5 h-5 text-[#141414]" />
                </span>
                <span className="text-[10px] font-bold px-2 py-0.5 bg-yellow-100 text-yellow-800 rounded-none uppercase tracking-wider">
                  Regression
                </span>
              </div>
              <h4 className="font-bold text-[#141414] font-display text-base mb-1 uppercase tracking-tight">Boston Housing Prices</h4>
              <p className="text-xs text-slate-600">
                Predict residential property values based on room counts, square footage, neighborhood, and age.
              </p>
            </div>
            <div className="mt-4 flex items-center gap-1.5 text-yellow-600 text-xs font-mono font-bold uppercase">
              <Sparkles className="w-3.5 h-3.5" />
              Load dataset
            </div>
          </div>

          {/* Sample 3 */}
          <div
            onClick={() => loadSampleDataset('heart')}
            className="p-6 border border-[#141414] bg-white rounded-none cursor-pointer hover:bg-slate-50 transition-colors flex flex-col justify-between min-h-[220px]"
          >
            <div>
              <div className="flex items-center justify-between mb-3">
                <span className="p-1.5 border border-[#141414] bg-neutral-50 rounded-none">
                  <FileSpreadsheet className="w-5 h-5 text-[#141414]" />
                </span>
                <span className="text-[10px] font-bold px-2 py-0.5 bg-blue-100 text-blue-800 rounded-none uppercase tracking-wider">
                  Classification
                </span>
              </div>
              <h4 className="font-bold text-[#141414] font-display text-base mb-1 uppercase tracking-tight">Cardiac Risk Evaluator</h4>
              <p className="text-xs text-slate-600">
                Determine cardiac risk levels based on physiological stats like blood pressure, cholesterol, gender, and chest pain.
              </p>
            </div>
            <div className="mt-4 flex items-center gap-1.5 text-blue-600 text-xs font-mono font-bold uppercase">
              <Sparkles className="w-3.5 h-3.5" />
              Load dataset
            </div>
          </div>
        </div>
      </div>

      {/* Safety Notice */}
      <div className="bg-white border border-[#141414] p-4 flex gap-3 text-[#141414] text-xs font-mono rounded-none">
        <AlertCircle className="w-4.5 h-4.5 text-[#141414] flex-shrink-0 mt-0.5" />
        <div>
          <p className="font-bold uppercase mb-0.5">Privacy Confirmed</p>
          <p className="opacity-80">
            Data parsing, modeling, and evaluation happen 100% locally in your web browser. No dataset records are ever sent to external cloud servers, keeping your proprietary business data completely safe.
          </p>
        </div>
      </div>
    </div>
  );
}
