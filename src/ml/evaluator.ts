/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 */

/**
 * Split dataset into training and validation sets
 */
export function trainTestSplit<T, K>(
  X: T[],
  y: K[],
  testSize: number = 0.2,
  seed: number = 42
): { X_train: T[]; X_val: T[]; y_train: K[]; y_val: K[] } {
  const n = X.length;
  const indices = Array.from({ length: n }, (_, i) => i);

  // Deterministic shuffle using a simple LCG random generator
  let state = seed;
  const random = () => {
    state = (state * 1664525 + 1013904223) % 4294967296;
    return state / 4294967296;
  };

  for (let i = n - 1; i > 0; i--) {
    const j = Math.floor(random() * (i + 1));
    const temp = indices[i];
    indices[i] = indices[j];
    indices[j] = temp;
  }

  const splitIdx = Math.floor(n * (1 - testSize));
  const trainIndices = indices.slice(0, splitIdx);
  const valIndices = indices.slice(splitIdx);

  return {
    X_train: trainIndices.map(i => X[i]),
    X_val: valIndices.map(i => X[i]),
    y_train: trainIndices.map(i => y[i]),
    y_val: valIndices.map(i => y[i]),
  };
}

/**
 * Calculates Area Under the ROC Curve (ROC-AUC) for binary classification
 */
function calculateBinaryAuc(actuals: number[], probabilities: number[]): number {
  const n = actuals.length;
  if (n === 0) return 0.5;

  // Pair up actuals and probabilities, then sort by probability descending
  const paired = actuals.map((act, i) => ({ act, prob: probabilities[i] }));
  paired.sort((a, b) => b.prob - a.prob);

  const numPositives = actuals.reduce((a, b) => a + b, 0);
  const numNegatives = n - numPositives;

  if (numPositives === 0 || numNegatives === 0) {
    return 1.0; // Perfect score if only one class is present
  }

  let tp = 0;
  let fp = 0;
  let prevTp = 0;
  let prevFp = 0;
  let area = 0;
  let prevProb = -1;

  for (let i = 0; i < n; i++) {
    const { act, prob } = paired[i];
    if (prob !== prevProb) {
      // Calculate trapezoid area under curve
      area += trapezoidArea(fp, prevFp, tp, prevTp);
      prevProb = prob;
      prevFp = fp;
      prevTp = tp;
    }
    if (act === 1) {
      tp++;
    } else {
      fp++;
    }
  }

  area += trapezoidArea(numNegatives, prevFp, numPositives, prevTp);
  
  // Normalize
  return area / (numPositives * numNegatives);
}

function trapezoidArea(x1: number, x2: number, y1: number, y2: number): number {
  const base = Math.abs(x1 - x2);
  const avgHeight = (y1 + y2) / 2;
  return base * avgHeight;
}

/**
 * Evaluates Classification metrics (Accuracy, Macro-F1, ROC-AUC)
 */
export function evaluateClassification(
  actuals: any[],
  predictions: any[],
  probabilities: number[][],
  classes: any[]
): { accuracy: number; f1: number; roc_auc: number } {
  const n = actuals.length;
  if (n === 0) return { accuracy: 0, f1: 0, roc_auc: 0.5 };

  // 1. Accuracy
  let correctCount = 0;
  for (let i = 0; i < n; i++) {
    if (String(actuals[i]) === String(predictions[i])) {
      correctCount++;
    }
  }
  const accuracy = correctCount / n;

  // 2. Macro F1-Score
  let sumF1 = 0;
  for (const currentClass of classes) {
    const strClass = String(currentClass);
    let tp = 0;
    let fp = 0;
    let fn = 0;

    for (let i = 0; i < n; i++) {
      const act = String(actuals[i]);
      const pred = String(predictions[i]);

      if (act === strClass && pred === strClass) tp++;
      else if (act !== strClass && pred === strClass) fp++;
      else if (act === strClass && pred !== strClass) fn++;
    }

    const precision = tp + fp > 0 ? tp / (tp + fp) : 0;
    const recall = tp + fn > 0 ? tp / (tp + fn) : 0;
    const f1 = precision + recall > 0 ? (2 * precision * recall) / (precision + recall) : 0;
    sumF1 += f1;
  }
  const f1 = sumF1 / classes.length;

  // 3. Multi-class ROC-AUC (using average of One-vs-Rest AUCs)
  let sumAuc = 0;
  for (let cIdx = 0; cIdx < classes.length; cIdx++) {
    const currentClass = classes[cIdx];
    const binaryActuals = actuals.map(val => (String(val) === String(currentClass) ? 1 : 0));
    const binaryProbs = probabilities.map(row => row[cIdx] ?? 0);
    
    const classAuc = calculateBinaryAuc(binaryActuals, binaryProbs);
    sumAuc += classAuc;
  }
  const roc_auc = sumAuc / classes.length;

  return {
    accuracy: parseFloat(accuracy.toFixed(4)),
    f1: parseFloat(f1.toFixed(4)),
    roc_auc: parseFloat(roc_auc.toFixed(4)),
  };
}

/**
 * Evaluates Regression metrics (RMSE, MAE, R-squared)
 */
export function evaluateRegression(
  actuals: number[],
  predictions: number[]
): { rmse: number; mae: number; r2: number } {
  const n = actuals.length;
  if (n === 0) return { rmse: 0, mae: 0, r2: 0 };

  let sumAbsoluteError = 0;
  let sumSquaredError = 0;
  let sumActual = 0;

  for (let i = 0; i < n; i++) {
    const act = actuals[i];
    const pred = predictions[i];
    const diff = act - pred;

    sumAbsoluteError += Math.abs(diff);
    sumSquaredError += diff * diff;
    sumActual += act;
  }

  const mae = sumAbsoluteError / n;
  const rmse = Math.sqrt(sumSquaredError / n);

  // R-squared
  const meanActual = sumActual / n;
  let sumTotalSquaredError = 0;
  for (let i = 0; i < n; i++) {
    const diff = actuals[i] - meanActual;
    sumTotalSquaredError += diff * diff;
  }

  const r2 = sumTotalSquaredError > 0 ? 1 - sumSquaredError / sumTotalSquaredError : 0;

  return {
    mae: parseFloat(mae.toFixed(4)),
    rmse: parseFloat(rmse.toFixed(4)),
    r2: parseFloat(r2.toFixed(4)),
  };
}
