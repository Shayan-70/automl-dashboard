/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 */

export type ColumnType = 'numeric' | 'categorical' | 'datetime' | 'unknown';

export interface ColumnInfo {
  name: string;
  type: ColumnType;
  missingCount: number;
  missingPercentage: number;
  mean?: number;
  median?: number;
  min?: number;
  max?: number;
  std?: number;
  uniqueValuesCount: number;
  uniqueValuesSample: any[];
  outliersCount?: number;
  isTargetPossible: boolean;
}

export interface DatasetSummary {
  rowCount: number;
  columnCount: number;
  columns: ColumnInfo[];
  targetColumn: string | null;
  problemType: 'classification' | 'regression' | null;
  classImbalance?: { label: string; count: number; percentage: number }[];
}

export interface PreprocessingConfig {
  targetColumn: string;
  problemType: 'classification' | 'regression';
  imputationStrategies: { [col: string]: 'mean' | 'median' | 'mode' };
  imputationValues: { [col: string]: any };
  encodedColumns: { [col: string]: { categories: any[] } };
  numericScales: { [col: string]: { mean: number; std: number } };
  engineeredFeatures: {
    type: 'interaction';
    colA: string;
    colB: string;
    newName: string;
  }[];
  selectedFeatures: string[]; // Final columns after feature engineering and selection
}

export interface ModelResult {
  modelId: string;
  modelName: string;
  metrics: { [metricName: string]: number };
  trainingTimeMs: number;
  predictions: any[];
  actuals: any[];
  modelObject: any; // Stored JS model for live inference
}

export interface TrainingProgress {
  status: 'idle' | 'preprocessing' | 'training' | 'evaluating' | 'completed' | 'failed';
  currentModel?: string;
  logs: string[];
}
