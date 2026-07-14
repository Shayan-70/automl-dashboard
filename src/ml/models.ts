/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 */

// ============================================================================
// Types & General Helpers
// ============================================================================

export interface MLModel {
  fit(X: number[][], y: any[]): void;
  predict(X: number[][]): any[];
}

export interface ClassifierModel extends MLModel {
  predictProba(X: number[][]): number[][];
  classes: any[];
}

function sigmoid(x: number): number {
  return 1 / (1 + Math.exp(-Math.max(-15, Math.min(15, x))));
}

function dotProduct(a: number[], b: number[]): number {
  let sum = 0;
  for (let i = 0; i < a.length; i++) {
    sum += a[i] * b[i];
  }
  return sum;
}

// ============================================================================
// 1. Regression Models
// ============================================================================

/**
 * Ridge Linear Regression with Gradient Descent
 */
export class LinearRegressionModel implements MLModel {
  private weights: number[] = [];
  private bias: number = 0;
  private learningRate: number = 0.05;
  private iterations: number = 250;
  private lambda: number = 0.1; // L2 Regularization parameter

  fit(X: number[][], y: any[]): void {
    if (X.length === 0) return;
    const numSamples = X.length;
    const numFeatures = X[0].length;
    
    this.weights = new Array(numFeatures).fill(0);
    this.bias = 0;

    const targets = y.map(Number);

    for (let iter = 0; iter < this.iterations; iter++) {
      let dW = new Array(numFeatures).fill(0);
      let dB = 0;

      for (let i = 0; i < numSamples; i++) {
        const pred = dotProduct(X[i], this.weights) + this.bias;
        const diff = pred - targets[i];

        for (let j = 0; j < numFeatures; j++) {
          dW[j] += diff * X[i][j];
        }
        dB += diff;
      }

      // Update weights with regularized gradient descent
      for (let j = 0; j < numFeatures; j++) {
        dW[j] = dW[j] / numSamples + (this.lambda * this.weights[j]);
        this.weights[j] -= this.learningRate * dW[j];
      }
      dB /= numSamples;
      this.bias -= this.learningRate * dB;
    }
  }

  predict(X: number[][]): number[] {
    return X.map(row => dotProduct(row, this.weights) + this.bias);
  }
}

/**
 * Decision Tree Node
 */
interface TreeNode {
  featureIndex: number;
  threshold: number;
  left: TreeNode | null;
  right: TreeNode | null;
  value: any; // Average target for regression, class probabilities/label for classification
  isLeaf: boolean;
}

/**
 * Regressor Decision Tree (MSE Split Criterion)
 */
class RegressionTree {
  private root: TreeNode | null = null;
  private maxDepth: number;
  private minSamplesSplit: number;

  constructor(maxDepth: number = 5, minSamplesSplit: number = 5) {
    this.maxDepth = maxDepth;
    this.minSamplesSplit = minSamplesSplit;
  }

  fit(X: number[][], y: number[]): void {
    this.root = this.buildTree(X, y, 0);
  }

  private buildTree(X: number[][], y: number[], depth: number): TreeNode {
    const numSamples = X.length;
    const numFeatures = X[0]?.length || 0;

    // Base cases
    if (depth >= this.maxDepth || numSamples < this.minSamplesSplit || numFeatures === 0) {
      return {
        featureIndex: -1,
        threshold: 0,
        left: null,
        right: null,
        value: this.mean(y),
        isLeaf: true,
      };
    }

    // Find best split
    let bestFeature = -1;
    let bestThreshold = 0;
    let bestMseReduction = -1;
    let leftX: number[][] = [];
    let leftY: number[] = [];
    let rightX: number[][] = [];
    let rightY: number[] = [];

    const currentMse = this.variance(y);

    // Feature subset selection (for Random Forest)
    const featuresToTry = Array.from({ length: numFeatures }, (_, i) => i);
    
    for (const fIdx of featuresToTry) {
      const values = X.map(r => r[fIdx]);
      // Limit candidate splits to speed up computation
      const uniqueValues = Array.from(new Set(values)).sort((a, b) => a - b);
      const candidates = uniqueValues.length > 15 
        ? Array.from({ length: 15 }, (_, i) => uniqueValues[Math.floor((i + 1) * uniqueValues.length / 16)])
        : uniqueValues;

      for (const threshold of candidates) {
        const lY: number[] = [];
        const rY: number[] = [];

        for (let i = 0; i < numSamples; i++) {
          if (X[i][fIdx] <= threshold) {
            lY.push(y[i]);
          } else {
            rY.push(y[i]);
          }
        }

        if (lY.length === 0 || rY.length === 0) continue;

        const leftWeight = lY.length / numSamples;
        const rightWeight = rY.length / numSamples;
        const splitMse = (leftWeight * this.variance(lY)) + (rightWeight * this.variance(rY));
        const mseReduction = currentMse - splitMse;

        if (mseReduction > bestMseReduction) {
          bestMseReduction = mseReduction;
          bestFeature = fIdx;
          bestThreshold = threshold;
        }
      }
    }

    if (bestFeature === -1 || bestMseReduction <= 0) {
      return {
        featureIndex: -1,
        threshold: 0,
        left: null,
        right: null,
        value: this.mean(y),
        isLeaf: true,
      };
    }

    // Split data
    for (let i = 0; i < numSamples; i++) {
      if (X[i][bestFeature] <= bestThreshold) {
        leftX.push(X[i]);
        leftY.push(y[i]);
      } else {
        rightX.push(X[i]);
        rightY.push(y[i]);
      }
    }

    return {
      featureIndex: bestFeature,
      threshold: bestThreshold,
      left: this.buildTree(leftX, leftY, depth + 1),
      right: this.buildTree(rightX, rightY, depth + 1),
      value: this.mean(y),
      isLeaf: false,
    };
  }

  predictRow(row: number[]): number {
    let node = this.root;
    while (node && !node.isLeaf) {
      if (row[node.featureIndex] <= node.threshold) {
        node = node.left;
      } else {
        node = node.right;
      }
    }
    return node ? node.value : 0;
  }

  private mean(vals: number[]): number {
    if (vals.length === 0) return 0;
    return vals.reduce((a, b) => a + b, 0) / vals.length;
  }

  private variance(vals: number[]): number {
    if (vals.length === 0) return 0;
    const m = this.mean(vals);
    return vals.reduce((acc, v) => acc + Math.pow(v - m, 2), 0) / vals.length;
  }
}

/**
 * Random Forest Regressor
 */
export class RandomForestRegressorModel implements MLModel {
  private trees: RegressionTree[] = [];
  private numTrees: number;
  private maxDepth: number;

  constructor(numTrees: number = 10, maxDepth: number = 6) {
    this.numTrees = numTrees;
    this.maxDepth = maxDepth;
  }

  fit(X: number[][], y: any[]): void {
    const numSamples = X.length;
    const targets = y.map(Number);
    this.trees = [];

    for (let t = 0; t < this.numTrees; t++) {
      // Bootstrap sampling (sampling with replacement)
      const bootX: number[][] = [];
      const bootY: number[] = [];
      for (let i = 0; i < numSamples; i++) {
        const randIdx = Math.floor(Math.random() * numSamples);
        bootX.push(X[randIdx]);
        bootY.push(targets[randIdx]);
      }

      const tree = new RegressionTree(this.maxDepth);
      tree.fit(bootX, bootY);
      this.trees.push(tree);
    }
  }

  predict(X: number[][]): number[] {
    return X.map(row => {
      const treePreds = this.trees.map(tree => tree.predictRow(row));
      return treePreds.reduce((a, b) => a + b, 0) / this.trees.length;
    });
  }
}

/**
 * Gradient Boosting Regressor
 */
export class GradientBoostingRegressorModel implements MLModel {
  private trees: RegressionTree[] = [];
  private learningRate: number;
  private numTrees: number;
  private maxDepth: number;
  private initialPrediction: number = 0;

  constructor(numTrees: number = 12, learningRate: number = 0.1, maxDepth: number = 4) {
    this.numTrees = numTrees;
    this.learningRate = learningRate;
    this.maxDepth = maxDepth;
  }

  fit(X: number[][], y: any[]): void {
    if (X.length === 0) return;
    const numSamples = X.length;
    const targets = y.map(Number);
    this.trees = [];

    // 1. Initial constant prediction (mean of targets)
    this.initialPrediction = targets.reduce((a, b) => a + b, 0) / numSamples;
    
    // Tracks sum of predictions
    const predictions = new Array(numSamples).fill(this.initialPrediction);

    for (let t = 0; t < this.numTrees; t++) {
      // 2. Compute pseudo-residuals
      const residuals = targets.map((target, idx) => target - predictions[idx]);

      // 3. Fit a regression tree on residual
      const tree = new RegressionTree(this.maxDepth, 4);
      tree.fit(X, residuals);

      // 4. Update model predictions with learning rate shrink
      for (let i = 0; i < numSamples; i++) {
        predictions[i] += this.learningRate * tree.predictRow(X[i]);
      }

      this.trees.push(tree);
    }
  }

  predict(X: number[][]): number[] {
    return X.map(row => {
      let pred = this.initialPrediction;
      for (const tree of this.trees) {
        pred += this.learningRate * tree.predictRow(row);
      }
      return pred;
    });
  }
}

// ============================================================================
// 2. Classification Models
// ============================================================================

/**
 * One-vs-Rest Logistic Regression Classifier (handles binary and multiclass)
 */
export class LogisticRegressionModel implements ClassifierModel {
  classes: any[] = [];
  private weightsPerClass: { [classLabel: string]: number[] } = {};
  private biasPerClass: { [classLabel: string]: number } = {};
  private learningRate: number = 0.1;
  private iterations: number = 200;

  fit(X: number[][], y: any[]): void {
    if (X.length === 0) return;
    const numSamples = X.length;
    const numFeatures = X[0].length;

    // Get unique classes
    this.classes = Array.from(new Set(y));
    this.weightsPerClass = {};
    this.biasPerClass = {};

    // For each class, fit a binary logistic regression (One-vs-Rest)
    for (const classLabel of this.classes) {
      // Map targets to 1 if match classLabel, else 0
      const binaryTargets = y.map(val => (val === classLabel ? 1 : 0));
      
      let weights = new Array(numFeatures).fill(0);
      let bias = 0;

      for (let iter = 0; iter < this.iterations; iter++) {
        let dW = new Array(numFeatures).fill(0);
        let dB = 0;

        for (let i = 0; i < numSamples; i++) {
          const rawScore = dotProduct(X[i], weights) + bias;
          const predProb = sigmoid(rawScore);
          const error = predProb - binaryTargets[i];

          for (let j = 0; j < numFeatures; j++) {
            dW[j] += error * X[i][j];
          }
          dB += error;
        }

        // Gradient update
        for (let j = 0; j < numFeatures; j++) {
          weights[j] -= this.learningRate * (dW[j] / numSamples);
        }
        bias -= this.learningRate * (dB / numSamples);
      }

      this.weightsPerClass[classLabel] = weights;
      this.biasPerClass[classLabel] = bias;
    }
  }

  predictProba(X: number[][]): number[][] {
    return X.map(row => {
      if (this.classes.length === 2) {
        // Standard Binary Logistic Regression
        const class0 = this.classes[0];
        const class1 = this.classes[1];
        const weights1 = this.weightsPerClass[class1] || [];
        const bias1 = this.biasPerClass[class1] ?? 0;
        
        const prob1 = sigmoid(dotProduct(row, weights1) + bias1);
        const prob0 = 1 - prob1;
        return [prob0, prob1];
      } else {
        // Multi-class: compute raw sigmoid logits, then normalize
        const probs = this.classes.map(cl => {
          const w = this.weightsPerClass[cl] || [];
          const b = this.biasPerClass[cl] ?? 0;
          return sigmoid(dotProduct(row, w) + b);
        });

        const sum = probs.reduce((a, b) => a + b, 0) || 1;
        return probs.map(p => p / sum);
      }
    });
  }

  predict(X: number[][]): any[] {
    const probas = this.predictProba(X);
    return probas.map(probRow => {
      let maxIdx = 0;
      let maxP = -1;
      for (let i = 0; i < probRow.length; i++) {
        if (probRow[i] > maxP) {
          maxP = probRow[i];
          maxIdx = i;
        }
      }
      return this.classes[maxIdx];
    });
  }
}

/**
 * Classifier Decision Tree (Gini split criterion)
 */
class ClassificationTree {
  private root: TreeNode | null = null;
  private maxDepth: number;
  private minSamplesSplit: number;
  private classes: any[] = [];

  constructor(maxDepth: number = 6, minSamplesSplit: number = 5) {
    this.maxDepth = maxDepth;
    this.minSamplesSplit = minSamplesSplit;
  }

  fit(X: number[][], y: any[], classes: any[]): void {
    this.classes = classes;
    this.root = this.buildTree(X, y, 0);
  }

  private buildTree(X: number[][], y: any[], depth: number): TreeNode {
    const numSamples = X.length;
    const numFeatures = X[0]?.length || 0;

    const classProbs = this.computeClassProbs(y);

    // Base cases
    if (depth >= this.maxDepth || numSamples < this.minSamplesSplit || numFeatures === 0) {
      return {
        featureIndex: -1,
        threshold: 0,
        left: null,
        right: null,
        value: classProbs,
        isLeaf: true,
      };
    }

    let bestFeature = -1;
    let bestThreshold = 0;
    let bestGiniGain = -1;
    let leftX: number[][] = [];
    let leftY: any[] = [];
    let rightX: number[][] = [];
    let rightY: any[] = [];

    const currentGini = this.gini(y);

    // Feature subset selection
    const featuresToTry = Array.from({ length: numFeatures }, (_, i) => i);

    for (const fIdx of featuresToTry) {
      const values = X.map(r => r[fIdx]);
      const uniqueValues = Array.from(new Set(values)).sort((a, b) => a - b);
      const candidates = uniqueValues.length > 15 
        ? Array.from({ length: 15 }, (_, i) => uniqueValues[Math.floor((i + 1) * uniqueValues.length / 16)])
        : uniqueValues;

      for (const threshold of candidates) {
        const lY: any[] = [];
        const rY: any[] = [];

        for (let i = 0; i < numSamples; i++) {
          if (X[i][fIdx] <= threshold) {
            lY.push(y[i]);
          } else {
            rY.push(y[i]);
          }
        }

        if (lY.length === 0 || rY.length === 0) continue;

        const leftWeight = lY.length / numSamples;
        const rightWeight = rY.length / numSamples;
        const splitGini = (leftWeight * this.gini(lY)) + (rightWeight * this.gini(rY));
        const giniGain = currentGini - splitGini;

        if (giniGain > bestGiniGain) {
          bestGiniGain = giniGain;
          bestFeature = fIdx;
          bestThreshold = threshold;
        }
      }
    }

    if (bestFeature === -1 || bestGiniGain <= 0) {
      return {
        featureIndex: -1,
        threshold: 0,
        left: null,
        right: null,
        value: classProbs,
        isLeaf: true,
      };
    }

    // Split data
    for (let i = 0; i < numSamples; i++) {
      if (X[i][bestFeature] <= bestThreshold) {
        leftX.push(X[i]);
        leftY.push(y[i]);
      } else {
        rightX.push(X[i]);
        rightY.push(y[i]);
      }
    }

    return {
      featureIndex: bestFeature,
      threshold: bestThreshold,
      left: this.buildTree(leftX, leftY, depth + 1),
      right: this.buildTree(rightX, rightY, depth + 1),
      value: classProbs,
      isLeaf: false,
    };
  }

  predictRowProbs(row: number[]): number[] {
    let node = this.root;
    while (node && !node.isLeaf) {
      if (row[node.featureIndex] <= node.threshold) {
        node = node.left;
      } else {
        node = node.right;
      }
    }
    return node ? node.value : new Array(this.classes.length).fill(0);
  }

  private gini(vals: any[]): number {
    if (vals.length === 0) return 0;
    const counts: { [key: string]: number } = {};
    for (const v of vals) {
      counts[v] = (counts[v] || 0) + 1;
    }
    let sumSqProb = 0;
    for (const cl of this.classes) {
      const p = (counts[cl] || 0) / vals.length;
      sumSqProb += p * p;
    }
    return 1 - sumSqProb;
  }

  private computeClassProbs(vals: any[]): number[] {
    const probs = new Array(this.classes.length).fill(0);
    if (vals.length === 0) return probs;
    
    const counts: { [key: string]: number } = {};
    for (const v of vals) {
      counts[v] = (counts[v] || 0) + 1;
    }
    for (let i = 0; i < this.classes.length; i++) {
      probs[i] = (counts[this.classes[i]] || 0) / vals.length;
    }
    return probs;
  }
}

/**
 * Random Forest Classifier
 */
export class RandomForestClassifierModel implements ClassifierModel {
  classes: any[] = [];
  private trees: ClassificationTree[] = [];
  private numTrees: number;
  private maxDepth: number;

  constructor(numTrees: number = 10, maxDepth: number = 6) {
    this.numTrees = numTrees;
    this.maxDepth = maxDepth;
  }

  fit(X: number[][], y: any[]): void {
    const numSamples = X.length;
    this.classes = Array.from(new Set(y));
    this.trees = [];

    for (let t = 0; t < this.numTrees; t++) {
      const bootX: number[][] = [];
      const bootY: any[] = [];
      for (let i = 0; i < numSamples; i++) {
        const randIdx = Math.floor(Math.random() * numSamples);
        bootX.push(X[randIdx]);
        bootY.push(y[randIdx]);
      }

      const tree = new ClassificationTree(this.maxDepth);
      tree.fit(bootX, bootY, this.classes);
      this.trees.push(tree);
    }
  }

  predictProba(X: number[][]): number[][] {
    return X.map(row => {
      const sumProbs = new Array(this.classes.length).fill(0);
      for (const tree of this.trees) {
        const treeProbs = tree.predictRowProbs(row);
        for (let i = 0; i < this.classes.length; i++) {
          sumProbs[i] += treeProbs[i];
        }
      }
      return sumProbs.map(p => p / this.trees.length);
    });
  }

  predict(X: number[][]): any[] {
    const probas = this.predictProba(X);
    return probas.map(probRow => {
      let maxIdx = 0;
      let maxP = -1;
      for (let i = 0; i < probRow.length; i++) {
        if (probRow[i] > maxP) {
          maxP = probRow[i];
          maxIdx = i;
        }
      }
      return this.classes[maxIdx];
    });
  }
}

/**
 * Gradient Boosting Classifier (Binary and multi-class support via OVR boosting)
 */
export class GradientBoostingClassifierModel implements ClassifierModel {
  classes: any[] = [];
  private modelsPerClass: { [classLabel: string]: GradientBoostingRegressorModel } = {};

  constructor(numTrees: number = 10, learningRate: number = 0.1, maxDepth: number = 4) {
    // We will build a multi-class Gradient Boosting algorithm using an OVR approach of Regressors.
    // Each class gets a Gradient Boosting Regressor fitting residual targets of (1 - prob) or (0 - prob).
  }

  fit(X: number[][], y: any[]): void {
    if (X.length === 0) return;
    this.classes = Array.from(new Set(y));
    this.modelsPerClass = {};

    // For each class, fit a GradientBoostingRegressorModel to fit residuals (1 if match classLabel, else 0)
    for (const classLabel of this.classes) {
      const binaryTargets = y.map(val => (val === classLabel ? 1 : 0));
      const model = new GradientBoostingRegressorModel(10, 0.1, 4);
      model.fit(X, binaryTargets);
      this.modelsPerClass[classLabel] = model;
    }
  }

  predictProba(X: number[][]): number[][] {
    return X.map(row => {
      const scores = this.classes.map(cl => {
        const regModel = this.modelsPerClass[cl];
        if (!regModel) return 0;
        // The regModel predicts a continuous scale. Sigmoid maps it to a valid probability indicator
        const raw = regModel.predict([row])[0];
        return sigmoid(raw);
      });

      const sum = scores.reduce((a, b) => a + b, 0) || 1;
      return scores.map(s => s / sum);
    });
  }

  predict(X: number[][]): any[] {
    const probas = this.predictProba(X);
    return probas.map(probRow => {
      let maxIdx = 0;
      let maxP = -1;
      for (let i = 0; i < probRow.length; i++) {
        if (probRow[i] > maxP) {
          maxP = probRow[i];
          maxIdx = i;
        }
      }
      return this.classes[maxIdx];
    });
  }
}

/**
 * Support Vector Classifier (One-vs-Rest)
 */
export class SupportVectorClassifierModel implements ClassifierModel {
  classes: any[] = [];
  private weightsPerClass: { [classLabel: string]: number[] } = {};
  private biasPerClass: { [classLabel: string]: number } = {};
  private learningRate: number = 0.05;
  private iterations: number = 200;
  private lambda: number = 0.01;

  fit(X: number[][], y: any[]): void {
    if (X.length === 0) return;
    const numSamples = X.length;
    const numFeatures = X[0].length;

    this.classes = Array.from(new Set(y));
    this.weightsPerClass = {};
    this.biasPerClass = {};

    for (const classLabel of this.classes) {
      const binaryTargets = y.map(val => (val === classLabel ? 1 : -1));
      
      let weights = new Array(numFeatures).fill(0);
      let bias = 0;

      for (let iter = 0; iter < this.iterations; iter++) {
        let dW = new Array(numFeatures).fill(0);
        let dB = 0;

        for (let i = 0; i < numSamples; i++) {
          const score = dotProduct(X[i], weights) + bias;
          const margin = binaryTargets[i] * score;

          if (margin < 1) {
            for (let j = 0; j < numFeatures; j++) {
              dW[j] += -binaryTargets[i] * X[i][j];
            }
            dB += -binaryTargets[i];
          }
        }

        for (let j = 0; j < numFeatures; j++) {
          dW[j] = dW[j] / numSamples + this.lambda * weights[j];
          weights[j] -= this.learningRate * dW[j];
        }
        bias -= this.learningRate * (dB / numSamples);
      }

      this.weightsPerClass[classLabel] = weights;
      this.biasPerClass[classLabel] = bias;
    }
  }

  predictProba(X: number[][]): number[][] {
    return X.map(row => {
      const scores = this.classes.map(cl => {
        const w = this.weightsPerClass[cl] || [];
        const b = this.biasPerClass[cl] ?? 0;
        return sigmoid(dotProduct(row, w) + b);
      });

      const sum = scores.reduce((a, b) => a + b, 0) || 1;
      return scores.map(s => s / sum);
    });
  }

  predict(X: number[][]): any[] {
    const probas = this.predictProba(X);
    return probas.map(probRow => {
      let maxIdx = 0;
      let maxP = -1;
      for (let i = 0; i < probRow.length; i++) {
        if (probRow[i] > maxP) {
          maxP = probRow[i];
          maxIdx = i;
        }
      }
      return this.classes[maxIdx];
    });
  }
}

/**
 * K-Nearest Neighbors Classifier
 */
export class KNearestNeighborsClassifierModel implements ClassifierModel {
  classes: any[] = [];
  private X_train: number[][] = [];
  private y_train: any[] = [];
  private k: number;

  constructor(k: number = 5) {
    this.k = k;
  }

  fit(X: number[][], y: any[]): void {
    this.X_train = X;
    this.y_train = y;
    this.classes = Array.from(new Set(y));
  }

  predictProba(X: number[][]): number[][] {
    return X.map(row => {
      const distances = this.X_train.map((trainRow, idx) => {
        let sumSq = 0;
        for (let j = 0; j < row.length; j++) {
          const diff = row[j] - trainRow[j];
          sumSq += diff * diff;
        }
        return { distance: Math.sqrt(sumSq), label: this.y_train[idx] };
      });

      distances.sort((a, b) => a.distance - b.distance);
      const neighbors = distances.slice(0, Math.min(this.k, this.X_train.length));
      
      const counts: { [key: string]: number } = {};
      neighbors.forEach(n => {
        counts[n.label] = (counts[n.label] || 0) + 1;
      });

      const probs = this.classes.map(cl => {
        return (counts[cl] || 0) / Math.max(1, neighbors.length);
      });

      const sum = probs.reduce((a, b) => a + b, 0) || 1;
      return probs.map(p => p / sum);
    });
  }

  predict(X: number[][]): any[] {
    const probas = this.predictProba(X);
    return probas.map(probRow => {
      let maxIdx = 0;
      let maxP = -1;
      for (let i = 0; i < probRow.length; i++) {
        if (probRow[i] > maxP) {
          maxP = probRow[i];
          maxIdx = i;
        }
      }
      return this.classes[maxIdx];
    });
  }
}

/**
 * Support Vector Regressor (Epsilon-Insensitive Hinge Loss)
 */
export class SupportVectorRegressorModel implements MLModel {
  private weights: number[] = [];
  private bias: number = 0;
  private learningRate: number = 0.05;
  private iterations: number = 200;
  private lambda: number = 0.01;
  private epsilon: number = 0.1;

  fit(X: number[][], y: any[]): void {
    if (X.length === 0) return;
    const numSamples = X.length;
    const numFeatures = X[0].length;

    this.weights = new Array(numFeatures).fill(0);
    this.bias = 0;

    const targets = y.map(Number);

    for (let iter = 0; iter < this.iterations; iter++) {
      let dW = new Array(numFeatures).fill(0);
      let dB = 0;

      for (let i = 0; i < numSamples; i++) {
        const pred = dotProduct(X[i], this.weights) + this.bias;
        const diff = targets[i] - pred;
        const absDiff = Math.abs(diff);

        if (absDiff > this.epsilon) {
          const sign = diff > 0 ? 1 : -1;
          for (let j = 0; j < numFeatures; j++) {
            dW[j] += -sign * X[i][j];
          }
          dB += -sign;
        }
      }

      for (let j = 0; j < numFeatures; j++) {
        dW[j] = dW[j] / numSamples + this.lambda * this.weights[j];
        this.weights[j] -= this.learningRate * dW[j];
      }
      this.bias -= this.learningRate * (dB / numSamples);
    }
  }

  predict(X: number[][]): number[] {
    return X.map(row => dotProduct(row, this.weights) + this.bias);
  }
}

/**
 * K-Nearest Neighbors Regressor
 */
export class KNearestNeighborsRegressorModel implements MLModel {
  private X_train: number[][] = [];
  private y_train: number[] = [];
  private k: number;

  constructor(k: number = 5) {
    this.k = k;
  }

  fit(X: number[][], y: any[]): void {
    this.X_train = X;
    this.y_train = y.map(Number);
  }

  predict(X: number[][]): number[] {
    return X.map(row => {
      const distances = this.X_train.map((trainRow, idx) => {
        let sumSq = 0;
        for (let j = 0; j < row.length; j++) {
          const diff = row[j] - trainRow[j];
          sumSq += diff * diff;
        }
        return { distance: Math.sqrt(sumSq), val: this.y_train[idx] };
      });

      distances.sort((a, b) => a.distance - b.distance);
      const neighbors = distances.slice(0, Math.min(this.k, this.X_train.length));
      
      const sum = neighbors.reduce((acc, n) => acc + n.val, 0);
      return sum / Math.max(1, neighbors.length);
    });
  }
}
