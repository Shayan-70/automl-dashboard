# Autonomous Data Science Agent

An elegant, browser-native machine learning pipeline and data profiling sandbox. This platform allows users to upload custom datasets, perform comprehensive statistical profiling, prune unwanted columns, train multiple candidate machine learning models simultaneously, compare cross-validation metrics, and deploy models in real-time—**all running 100% locally in the browser**.

---

## 🚀 Architectural Design: Local vs. API-Based

This application is **completely local-first (not API-based)**. It operates entirely as a client-side Single Page Application (SPA).

### Why a Browser-Native Architecture?
1. **Total Data Privacy**: Your proprietary CSV and Excel sheets are parsed and processed directly in your device's memory. No dataset records are ever uploaded to external cloud APIs or servers.
2. **Instant Response Times**: Local model training and prediction execution happen without network latency or round-trip times.
3. **Zero Cost & Cloud-Free**: Running machine learning algorithms in the browser via native TypeScript mathematical models eliminates the need for expensive cloud compute servers or external token-based APIs.

---

## 🎨 Visual System: Professional Polish

The application implements a premium, high-contrast, editorial design style reflecting Swiss design principles and minimalist typography:
- **Clean Palette**: A signature `#E4E3E0` off-white warm canvas paired with deep `#141414` slate-charcoal text.
- **Intentional Spacing**: Balanced negative space combined with raw, dashed bordered containers.
- **Unified Font Stack**: Clean "Inter" body text paired with "Space Grotesk" for display headings and "JetBrains Mono" for statistical tables and console lines.
- **No Clutter**: Stripped of generic shadows and standard pastel gradients, offering a tactile, professional-grade visual environment.

---

## 🛠️ Main Features & Pipeline Flow

### 1. 📂 01. Dataset Upload
- Supports custom **CSV** and **Excel** (.xlsx, .xls) sheets up to 10MB.
- Dynamic drag-and-drop interface with browser-native file parsing.
- Includes pre-loaded sandbox datasets (e.g., Iris, Boston Housing, Cardiac Risk Evaluator) to easily test workflows.

### 2. 📊 02. Dataset Profiling
- Infers dataset shape (Total Rows, Features count).
- Auto-detects the machine learning task type (**Classification** or **Regression**).
- Renders live target class distribution charts using Recharts.
- Computes detailed missing values, cardinality, standard deviations, and IQR outlier counts.

### 3. ✂️ 03. Column Pruning (Feature Selection)
- Displays an interactive checklist of all detected columns (name + inferred type) immediately after data profiling.
- Allows users to manually select and exclude irrelevant, redundant, or leaking columns from training.
- **Label Protection**: Automatically locks and prevents the selected target label column from being dropped.
- **Consistently Apprehended**: Excluded column configurations are persisted and applied seamlessly to downstream test data during inference.

### 4. ⚙️ 04. Model Training
- Cleans data, handles missingness, and performs categorical one-hot encoding on the active feature dataframe.
- Optimizes **5 standard models in parallel** for both ML tasks:
  - **Classification**: Logistic Regression (One-vs-Rest), Random Forest, Gradient Boosting, Support Vector Machine (SVC), and K-Nearest Neighbors (KNN).
  - **Regression**: Ridge Linear Regression, Random Forest, Gradient Boosting, Support Vector Regressor (SVR), and K-Nearest Neighbors Regressor.
- Outlines execution state details live in an interactive console block.

### 5. ⚖️ 05. Model Comparison Matrix
- Displays interactive validation charts for comparative analysis.
- Summarizes performance side-by-side using key metrics (Accuracy, F1-Score, and ROC-AUC for classifiers; R-squared, MAE, and RMSE for regressors).
- Ranks candidate models by primary metrics to surface optimal models and lets users deploy any candidate of choice.

### 6. 🎛️ 06. Deployed Inference Console
- **Manual Inference**: Enter custom parameters via dynamically generated form inputs (with excluded columns automatically hidden to maintain consistent feature alignment).
- **Batch Inference**: Upload fresh testing spreadsheets to automatically generate and download predictions appended as a new column.

---

## 💻 Technical Stack

- **Framework**: React 18+ & Vite
- **Language**: TypeScript
- **Styling**: Tailwind CSS
- **Parser**: SheetJS (`xlsx`) for local spreadsheet processing
- **Visuals & Charts**: Lucide React & Recharts
- **Animations**: Framer Motion (`motion/react`)
