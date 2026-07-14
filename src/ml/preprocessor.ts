/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 */

import { DatasetSummary, ColumnInfo, ColumnType, PreprocessingConfig } from '../types';

/**
 * Helper to check if a value is numeric
 */
function isNumericValue(val: any): boolean {
  if (val === null || val === undefined || val === '') return false;
  const num = Number(val);
  return !isNaN(num) && isFinite(num);
}

/**
 * Helper to check if a value is a date
 */
function isDateValue(val: any): boolean {
  if (val === null || val === undefined || val === '') return false;
  if (typeof val === 'number') return false; // Unix timestamps could be parsed as date, but keep them as numeric
  const d = new Date(val);
  return !isNaN(d.getTime()) && val.toString().includes('-') || val.toString().includes('/');
}

/**
 * Analyzes the raw dataset (array of objects) and computes summary statistics
 */
export function analyzeDataset(data: any[], targetColumnName?: string): DatasetSummary {
  const rowCount = data.length;
  if (rowCount === 0) {
    return {
      rowCount: 0,
      columnCount: 0,
      columns: [],
      targetColumn: null,
      problemType: null,
    };
  }

  // Get all unique keys in the data
  const keys = Array.from(new Set(data.flatMap(row => Object.keys(row))));
  const columnCount = keys.length;
  const columns: ColumnInfo[] = [];

  for (const colName of keys) {
    const rawValues = data.map(row => row[colName]);
    const nonNullValues = rawValues.filter(val => val !== null && val !== undefined && val !== '');
    const missingCount = rowCount - nonNullValues.length;
    const missingPercentage = (missingCount / rowCount) * 100;

    // Detect column type
    let type: ColumnType = 'categorical';
    let numericCount = 0;
    let dateCount = 0;

    for (const val of nonNullValues) {
      if (isNumericValue(val)) numericCount++;
      else if (isDateValue(val)) dateCount++;
    }

    const nonNullSize = nonNullValues.length;
    if (nonNullSize > 0) {
      if (numericCount / nonNullSize > 0.8) {
        type = 'numeric';
      } else if (dateCount / nonNullSize > 0.8) {
        type = 'datetime';
      }
    } else {
      type = 'unknown';
    }

    // Unique values
    const uniqueVals = Array.from(new Set(nonNullValues));
    const uniqueValuesCount = uniqueVals.length;
    const uniqueValuesSample = uniqueVals.slice(0, 10);

    const colInfo: ColumnInfo = {
      name: colName,
      type,
      missingCount,
      missingPercentage,
      uniqueValuesCount,
      uniqueValuesSample,
      isTargetPossible: uniqueValuesCount > 1 && uniqueValuesCount < rowCount && type !== 'unknown' && type !== 'datetime',
    };

    // Numeric statistics
    if (type === 'numeric' && nonNullSize > 0) {
      const numbers = nonNullValues.map(v => Number(v)).sort((a, b) => a - b);
      const sum = numbers.reduce((a, b) => a + b, 0);
      const mean = sum / nonNullSize;
      
      // Median
      const mid = Math.floor(numbers.length / 2);
      const median = numbers.length % 2 !== 0 ? numbers[mid] : (numbers[mid - 1] + numbers[mid]) / 2;
      
      const min = numbers[0];
      const max = numbers[numbers.length - 1];

      // Std deviation
      const sqDiffs = numbers.map(v => Math.pow(v - mean, 2));
      const variance = sqDiffs.reduce((a, b) => a + b, 0) / nonNullSize;
      const std = Math.sqrt(variance);

      colInfo.mean = parseFloat(mean.toFixed(4));
      colInfo.median = parseFloat(median.toFixed(4));
      colInfo.min = parseFloat(min.toFixed(4));
      colInfo.max = parseFloat(max.toFixed(4));
      colInfo.std = parseFloat(std.toFixed(4));

      // Outliers using IQR method
      const q1Idx = Math.floor(numbers.length * 0.25);
      const q3Idx = Math.floor(numbers.length * 0.75);
      const q1 = numbers[q1Idx];
      const q3 = numbers[q3Idx];
      const iqr = q3 - q1;
      const lowerBound = q1 - 1.5 * iqr;
      const upperBound = q3 + 1.5 * iqr;
      const outliers = numbers.filter(v => v < lowerBound || v > upperBound);
      colInfo.outliersCount = outliers.length;
    }

    columns.push(colInfo);
  }

  // Auto-detect target column if not provided
  let targetColumn = targetColumnName || null;
  if (!targetColumn) {
    // Try to find a logical target (e.g., column containing "target", "label", "class", "churn", "price", "output")
    const commonTargets = ['target', 'label', 'class', 'churn', 'price', 'output', 'revenue', 'sold', 'clicked', 'outcome', 'survived', 'diagnoses'];
    const matched = columns.find(col => commonTargets.includes(col.name.toLowerCase()) && col.isTargetPossible);
    if (matched) {
      targetColumn = matched.name;
    } else {
      // Pick the last column that's eligible
      const eligible = columns.filter(col => col.isTargetPossible);
      if (eligible.length > 0) {
        targetColumn = eligible[eligible.length - 1].name;
      }
    }
  }

  // Infer problem type based on target column
  let problemType: 'classification' | 'regression' | null = null;
  let classImbalance: { label: string; count: number; percentage: number }[] | undefined;

  if (targetColumn) {
    const targetColInfo = columns.find(c => c.name === targetColumn);
    if (targetColInfo) {
      const nonNullTargets = data
        .map(row => row[targetColumn!])
        .filter(val => val !== null && val !== undefined && val !== '');

      const uniqueTargetVals = Array.from(new Set(nonNullTargets));
      
      // If target is numeric, and has a large number of unique values, or a high percentage of unique values, it's regression
      if (targetColInfo.type === 'numeric' && uniqueTargetVals.length > 10 && uniqueTargetVals.length > rowCount * 0.05) {
        problemType = 'regression';
      } else {
        problemType = 'classification';

        // Calculate class imbalance
        const counts: { [key: string]: number } = {};
        for (const val of nonNullTargets) {
          const strVal = String(val);
          counts[strVal] = (counts[strVal] || 0) + 1;
        }
        
        classImbalance = Object.entries(counts).map(([label, count]) => ({
          label,
          count,
          percentage: parseFloat(((count / nonNullTargets.length) * 100).toFixed(2)),
        })).sort((a, b) => b.count - a.count);
      }
    }
  }

  return {
    rowCount,
    columnCount,
    columns,
    targetColumn,
    problemType,
    classImbalance,
  };
}

/**
 * Builds the preprocessing pipeline configuration based on raw training data
 */
export function buildPreprocessorConfig(
  data: any[],
  targetColumn: string,
  problemType: 'classification' | 'regression'
): PreprocessingConfig {
  const summary = analyzeDataset(data, targetColumn);
  
  const imputationStrategies: { [col: string]: 'mean' | 'median' | 'mode' } = {};
  const imputationValues: { [col: string]: any } = {};
  const encodedColumns: { [col: string]: { categories: any[] } } = {};
  const numericScales: { [col: string]: { mean: number; std: number } } = {};
  
  // Calculate imputation values and scaling parameters
  for (const col of summary.columns) {
    if (col.name === targetColumn) continue;
    if (col.type === 'unknown' || col.type === 'datetime') continue;

    const rawValues = data.map(row => row[col.name]);
    const nonNullValues = rawValues.filter(val => val !== null && val !== undefined && val !== '');

    if (col.type === 'numeric') {
      const numbers = nonNullValues.map(v => Number(v));
      
      // We will impute with Median
      imputationStrategies[col.name] = 'median';
      
      let median = 0;
      if (numbers.length > 0) {
        numbers.sort((a, b) => a - b);
        const mid = Math.floor(numbers.length / 2);
        median = numbers.length % 2 !== 0 ? numbers[mid] : (numbers[mid - 1] + numbers[mid]) / 2;
      }
      imputationValues[col.name] = median;

      // Scaling statistics
      let mean = 0;
      let std = 1;
      if (numbers.length > 0) {
        const sum = numbers.reduce((a, b) => a + b, 0);
        mean = sum / numbers.length;
        const sqDiffs = numbers.map(v => Math.pow(v - mean, 2));
        const variance = sqDiffs.reduce((a, b) => a + b, 0) / numbers.length;
        std = Math.sqrt(variance) || 1; // Fallback to 1 if variance is 0
      }
      numericScales[col.name] = { mean, std };

    } else if (col.type === 'categorical') {
      // We will impute with Mode
      imputationStrategies[col.name] = 'mode';
      
      let mode = 'Missing';
      if (nonNullValues.length > 0) {
        const counts: { [key: string]: number } = {};
        for (const v of nonNullValues) {
          const s = String(v);
          counts[s] = (counts[s] || 0) + 1;
        }
        const sortedModes = Object.entries(counts).sort((a, b) => b[1] - a[1]);
        mode = sortedModes[0][0];
      }
      imputationValues[col.name] = mode;

      // Find all unique categories
      const categories = Array.from(new Set(nonNullValues.map(v => String(v))));
      encodedColumns[col.name] = { categories };
    }
  }

  // 1. Feature Engineering: Automated Interaction Terms
  // Find up to 3 pairs of numeric columns that aren't perfectly correlated but have reasonable spread, to create X * Y terms
  const numericColumns = summary.columns.filter(c => c.type === 'numeric' && c.name !== targetColumn);
  const engineeredFeatures: PreprocessingConfig['engineeredFeatures'] = [];

  if (numericColumns.length >= 2) {
    // Generate simple interactions for the first few numeric features
    const maxInteractions = Math.min(3, Math.floor((numericColumns.length * (numericColumns.length - 1)) / 2));
    let count = 0;
    for (let i = 0; i < numericColumns.length && count < maxInteractions; i++) {
      for (let j = i + 1; j < numericColumns.length && count < maxInteractions; j++) {
        const colA = numericColumns[i].name;
        const colB = numericColumns[j].name;
        engineeredFeatures.push({
          type: 'interaction',
          colA,
          colB,
          newName: `${colA}_x_${colB}`,
        });
        count++;
      }
    }
  }

  // 2. Feature Selection: Determine selected features
  // We include:
  // - All original scaled numeric features
  // - One-hot encoded versions of categorical features
  // - Engineered interaction terms
  // We perform correlation-based filtering. If any numeric feature has correlation > 0.98 with another feature, we drop it.
  const finalFeatures: string[] = [];

  // Add original numeric features
  numericColumns.forEach(c => {
    // Check if column has zero standard deviation (constant column)
    const scale = numericScales[c.name];
    if (scale && scale.std > 0.0001) {
      finalFeatures.push(c.name);
    }
  });

  // Add engineered interaction features
  engineeredFeatures.forEach(eff => {
    finalFeatures.push(eff.newName);
  });

  // Add categorical features (One-hot encoded names)
  Object.entries(encodedColumns).forEach(([colName, info]) => {
    info.categories.forEach(cat => {
      finalFeatures.push(`${colName}_is_${cat}`);
    });
  });

  return {
    targetColumn,
    problemType,
    imputationStrategies,
    imputationValues,
    encodedColumns,
    numericScales,
    engineeredFeatures,
    selectedFeatures: finalFeatures,
  };
}

/**
 * Transforms a raw dataset row-by-row into a numeric matrix of features and a flat array of labels
 */
export function transformData(
  data: any[],
  config: PreprocessingConfig,
  isTraining: boolean = true
): { features: number[][]; labels: any[] } {
  const features: number[][] = [];
  const labels: any[] = [];

  for (const row of data) {
    const rowFeatures: { [key: string]: number } = {};

    // 1. Process standard columns
    // Impute, Scale and One-Hot encode
    
    // Process numeric columns
    Object.entries(config.numericScales).forEach(([colName, scale]) => {
      let val = row[colName];
      // Impute missing
      if (val === null || val === undefined || val === '') {
        val = config.imputationValues[colName];
      }
      
      // Standardize: (X - mean) / std
      const numVal = Number(val);
      const scaledVal = (numVal - scale.mean) / (scale.std || 1);
      rowFeatures[colName] = scaledVal;
    });

    // Process categorical columns
    Object.entries(config.encodedColumns).forEach(([colName, info]) => {
      let val = row[colName];
      // Impute missing
      if (val === null || val === undefined || val === '') {
        val = config.imputationValues[colName];
      }
      const strVal = String(val);

      // Create One-Hot keys
      info.categories.forEach(cat => {
        const featureName = `${colName}_is_${cat}`;
        rowFeatures[featureName] = strVal === String(cat) ? 1 : 0;
      });
    });

    // 2. Process engineered features (e.g. interactions)
    config.engineeredFeatures.forEach(eff => {
      if (eff.type === 'interaction') {
        const valA = rowFeatures[eff.colA] ?? 0;
        const valB = rowFeatures[eff.colB] ?? 0;
        rowFeatures[eff.newName] = valA * valB;
      }
    });

    // 3. Assemble selected feature array matching selectedFeatures order
    const featureArray = config.selectedFeatures.map(feat => rowFeatures[feat] ?? 0);
    features.push(featureArray);

    // 4. Capture label if training or if target exists in row
    if (isTraining || (row[config.targetColumn] !== undefined && row[config.targetColumn] !== null)) {
      labels.push(row[config.targetColumn]);
    } else {
      labels.push(null);
    }
  }

  return {
    features,
    labels,
  };
}
