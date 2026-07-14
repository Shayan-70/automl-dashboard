import streamlit as st
import pandas as pd
import numpy as np
import time
import io
import plotly.express as px
import plotly.graph_objects as go

# Scikit-learn imports
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score,
    r2_score, mean_absolute_error, mean_squared_error
)

# Classifiers
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

# Regressors
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor

# ----------------------------------------------------
# Layout & Page Config
# ----------------------------------------------------
st.set_page_config(
    page_title="Autonomous Data Science Agent",
    page_icon="⚙️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Inject Swiss Editorial Minimalist Styling
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=Space+Grotesk:wght@500;700&family=JetBrains+Mono:wght@400;500&display=swap');
    
    /* Main body background & color override */
    html, body, [data-testid="stAppViewContainer"] {
        background-color: #E4E3E0 !important;
        color: #141414 !important;
        font-family: "Inter", sans-serif !important;
    }

    [data-testid="stHeader"] {
        background-color: rgba(228, 227, 224, 0.8) !important;
    }

    /* Titles and Headings with Space Grotesk */
    h1, h2, h3, h4, h5, h6 {
        font-family: "Space Grotesk", sans-serif !important;
        color: #141414 !important;
        font-weight: 700 !important;
        letter-spacing: -0.02em !important;
    }

    /* Monospace formatting for stats and codes */
    code, pre, .mono-text, .stDataFrame, div[data-testid="metric-container"] {
        font-family: "JetBrains Mono", monospace !important;
    }

    /* Premium minimalist flat cards */
    .editorial-card {
        background-color: #FFFFFF;
        border: 1px solid #141414;
        padding: 24px;
        margin-bottom: 24px;
        border-radius: 0px !important;
    }

    .dashed-card {
        background-color: #FFFFFF;
        border: 1px dashed #141414;
        padding: 24px;
        margin-bottom: 24px;
        border-radius: 0px !important;
    }

    /* Custom Buttons - flat, bold, highly clickable */
    .stButton > button {
        background-color: #141414 !important;
        color: #E4E3E0 !important;
        border: 1px solid #141414 !important;
        border-radius: 0px !important;
        font-family: "Space Grotesk", sans-serif !important;
        font-weight: 700 !important;
        text-transform: uppercase !important;
        letter-spacing: 0.05em !important;
        padding: 12px 24px !important;
        transition: all 0.2s ease-in-out !important;
        cursor: pointer;
    }

    .stButton > button:hover {
        background-color: transparent !important;
        color: #141414 !important;
        border: 1px solid #141414 !important;
    }

    /* Back buttons */
    .back-btn-container .stButton > button {
        background-color: transparent !important;
        color: #141414 !important;
        border: 1px solid #141414 !important;
    }
    .back-btn-container .stButton > button:hover {
        background-color: #141414 !important;
        color: #E4E3E0 !important;
    }

    /* Custom form elements styling */
    div[data-baseweb="select"] {
        border-radius: 0px !important;
        border: 1px solid #141414 !important;
    }

    input {
        border-radius: 0px !important;
        border: 1px solid #141414 !important;
    }

    /* Header Stepper Component styling */
    .stepper-container {
        display: flex;
        justify-content: space-between;
        align-items: center;
        border-bottom: 1px solid #141414;
        padding-bottom: 12px;
        margin-bottom: 30px;
        flex-wrap: wrap;
        gap: 10px;
    }

    .step-item {
        font-family: "JetBrains Mono", monospace;
        font-size: 11px;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.08em;
    }

    .step-active {
        color: #141414;
        border-bottom: 2px solid #141414;
        padding-bottom: 2px;
    }

    .step-inactive {
        color: #141414;
        opacity: 0.4;
    }

    /* Metrics and indicators */
    .pill-green {
        background-color: rgba(34, 197, 94, 0.15);
        color: rgb(21, 128, 61);
        border: 1px solid rgb(34, 197, 94);
        padding: 2px 8px;
        font-size: 10px;
        font-weight: 700;
        text-transform: uppercase;
    }

    .pill-yellow {
        background-color: rgba(234, 179, 8, 0.15);
        color: rgb(161, 98, 7);
        border: 1px solid rgb(234, 179, 8);
        padding: 2px 8px;
        font-size: 10px;
        font-weight: 700;
        text-transform: uppercase;
    }

    /* Interactive console log */
    .console-container {
        background-color: #141414;
        color: #00FF66;
        padding: 16px;
        font-family: "JetBrains Mono", monospace;
        font-size: 12px;
        height: 300px;
        overflow-y: auto;
        border: 1px solid #141414;
        margin-bottom: 20px;
    }
</style>
""", unsafe_allow_html=True)

# ----------------------------------------------------
# Session State Initialization
# ----------------------------------------------------
if "step" not in st.session_state:
    st.session_state.step = "upload"
if "raw_data" not in st.session_state:
    st.session_state.raw_data = None
if "file_name" not in st.session_state:
    st.session_state.file_name = ""
if "target_column" not in st.session_state:
    st.session_state.target_column = None
if "problem_type" not in st.session_state:
    st.session_state.problem_type = "classification"
if "excluded_columns" not in st.session_state:
    st.session_state.excluded_columns = []
if "trained_models" not in st.session_state:
    st.session_state.trained_models = None
if "selected_model_id" not in st.session_state:
    st.session_state.selected_model_id = None
if "pipeline_preprocessor" not in st.session_state:
    st.session_state.pipeline_preprocessor = None
if "model_classes" not in st.session_state:
    st.session_state.model_classes = {}

# ----------------------------------------------------
# Helper Functions: Sandbox Datasets Generator
# ----------------------------------------------------
def get_sandbox_dataset(dataset_type):
    np.random.seed(42)
    if dataset_type == "iris":
        classes = ["setosa", "versicolor", "virginica"]
        rows = []
        for i in range(150):
            cls_idx = i // 50
            cls_name = classes[cls_idx]
            # Add randomized sepal/petal dimensions with distinct means
            sepal_len = np.round(np.random.normal(5.0 + cls_idx * 0.9, 0.4), 1)
            sepal_wid = np.round(np.random.normal(3.4 - cls_idx * 0.3, 0.3), 1)
            petal_len = np.round(np.random.normal(1.5 + cls_idx * 1.8, 0.5), 1)
            petal_wid = np.round(np.random.normal(0.2 + cls_idx * 0.8, 0.2), 1)
            rows.append({
                "sepal_length": sepal_len,
                "sepal_width": sepal_wid,
                "petal_length": petal_len,
                "petal_width": petal_wid,
                "species": cls_name
            })
        return pd.DataFrame(rows), "iris_flowers.csv"

    elif dataset_type == "housing":
        rows = []
        for i in range(300):
            # Boston/California housing simulation
            med_inc = np.round(np.random.uniform(1.5, 8.0), 4)
            house_age = int(np.random.randint(5, 52))
            ave_rooms = np.round(np.random.normal(5.5 + med_inc * 0.2, 0.8), 2)
            ave_bedrms = np.round(np.random.normal(1.0 + np.random.uniform(0, 0.1), 0.1), 2)
            population = int(np.random.normal(1400, 800))
            population = max(100, population)
            ave_occup = np.round(np.random.uniform(1.8, 4.5), 2)
            latitude = np.round(34.0 + np.random.uniform(-1.5, 1.5), 2)
            longitude = np.round(-118.0 + np.random.uniform(-1.5, 1.5), 2)
            
            # Predict target value based on synthetic weights + noise
            target = np.round(
                med_inc * 0.4 + house_age * -0.002 + ave_rooms * 0.05 + np.random.normal(0.5, 0.25),
                2
            )
            target = max(0.1, min(5.0, target))
            
            rows.append({
                "MedInc": med_inc,
                "HouseAge": house_age,
                "AveRooms": ave_rooms,
                "AveBedrms": ave_bedrms,
                "Population": population,
                "AveOccup": ave_occup,
                "Latitude": latitude,
                "Longitude": longitude,
                "MedHouseValue": target
            })
        return pd.DataFrame(rows), "california_housing.csv"

    elif dataset_type == "heart":
        rows = []
        for i in range(250):
            age = int(np.random.randint(29, 78))
            sex = int(np.random.choice([0, 1]))
            chest_pain = int(np.random.choice([0, 1, 2, 3]))
            cholesterol = int(np.random.normal(240, 50))
            cholesterol = max(100, cholesterol)
            resting_bp = int(np.random.normal(130, 15))
            max_hr = int(np.random.normal(200 - age * 0.8, 15))
            ex_angina = int(np.random.choice([0, 1]))
            
            # Risk formula
            risk_score = (age * 0.02) + (sex * 0.5) + (chest_pain * -0.3) + (cholesterol * 0.003) + (ex_angina * 0.8) - 1.5
            p = 1 / (1 + np.exp(-risk_score))
            target = 1 if np.random.random() < p else 0
            
            rows.append({
                "Age": age,
                "Sex_Male": sex,
                "ChestPainType": chest_pain,
                "Cholesterol": cholesterol,
                "RestingBP": resting_bp,
                "MaxHR": max_hr,
                "ExerciseAngina": ex_angina,
                "CardiacRisk": target
            })
        return pd.DataFrame(rows), "cardiac_risk_evaluator.csv"

# ----------------------------------------------------
# Layout Component: Top Minimal Stepper
# ----------------------------------------------------
def render_stepper():
    steps = [
        ("upload", "01. Upload"),
        ("analyze", "02. Profile"),
        ("prune", "03. Prune"),
        ("train", "04. Train"),
        ("compare", "05. Compare"),
        ("predict", "06. Predict"),
    ]
    
    stepper_html = '<div class="stepper-container">'
    for code, label in steps:
        active_class = "step-active" if st.session_state.step == code else "step-inactive"
        stepper_html += f'<div class="step-item {active_class}">{label}</div>'
    stepper_html += '</div>'
    
    st.markdown(stepper_html, unsafe_allow_html=True)

# ----------------------------------------------------
# MAIN RENDERER (BASED ON CURRENT STEP)
# ----------------------------------------------------

render_stepper()

# Add a clean container for layout spacing
main_container = st.container()

# ----------------------------------------------------
# STEP 1: Upload Dataset
# ----------------------------------------------------
if st.session_state.step == "upload":
    with main_container:
        st.markdown('<h2 style="margin-top:0px;">01. Import Active Datasearch Pipeline</h2>', unsafe_allow_html=True)
        st.markdown("<p style='font-size:13px; color:#555;'>Upload custom datasets or boot standard sandbox files directly into local partition memory.</p>", unsafe_allow_html=True)
        
        col1, col2 = st.columns([2, 1], gap="large")
        
        with col1:
            st.markdown('<div class="dashed-card">', unsafe_allow_html=True)
            uploaded_file = st.file_uploader(
                "Upload custom spreadsheets (CSV, XLSX, XLS)",
                type=["csv", "xlsx", "xls"],
                key="uploader_input",
                label_visibility="collapsed"
            )
            
            if uploaded_file is not None:
                try:
                    ext = uploaded_file.name.split(".")[-1].lower()
                    if ext == "csv":
                        df = pd.read_csv(uploaded_file)
                    else:
                        df = pd.read_excel(uploaded_file)
                    
                    if len(df) == 0:
                        st.error("The uploaded dataset contains zero rows. Please select a valid dataset.")
                    else:
                        st.session_state.raw_data = df
                        st.session_state.file_name = uploaded_file.name
                        # Reset child state parameters
                        st.session_state.target_column = None
                        st.session_state.excluded_columns = []
                        st.session_state.trained_models = None
                        st.session_state.step = "analyze"
                        st.rerun()
                except Exception as e:
                    st.error(f"Failed to parse target file: {str(e)}")
            st.markdown('</div>', unsafe_allow_html=True)
            
        with col2:
            st.markdown('<div class="editorial-card" style="height:100%;">', unsafe_allow_html=True)
            st.markdown("<h5>⭐ Preloaded Sandboxes</h5>", unsafe_allow_html=True)
            st.markdown("<p style='font-size:12px; color:#666;'>Try sandbox datasets to test features instantly with deterministic distributions.</p>", unsafe_allow_html=True)
            
            if st.button("🌸 Load Iris Flowers (Classification)"):
                df, name = get_sandbox_dataset("iris")
                st.session_state.raw_data = df
                st.session_state.file_name = name
                st.session_state.target_column = None
                st.session_state.excluded_columns = []
                st.session_state.trained_models = None
                st.session_state.step = "analyze"
                st.rerun()
                
            if st.button("🏡 Load California Housing (Regression)"):
                df, name = get_sandbox_dataset("housing")
                st.session_state.raw_data = df
                st.session_state.file_name = name
                st.session_state.target_column = None
                st.session_state.excluded_columns = []
                st.session_state.trained_models = None
                st.session_state.step = "analyze"
                st.rerun()
                
            if st.button("❤️ Load Heart Disease Risk (Classification)"):
                df, name = get_sandbox_dataset("heart")
                st.session_state.raw_data = df
                st.session_state.file_name = name
                st.session_state.target_column = None
                st.session_state.excluded_columns = []
                st.session_state.trained_models = None
                st.session_state.step = "analyze"
                st.rerun()
            st.markdown('</div>', unsafe_allow_html=True)

# ----------------------------------------------------
# STEP 2: Dataset Profiling & Auto-Config
# ----------------------------------------------------
elif st.session_state.step == "analyze":
    df = st.session_state.raw_data
    if df is None:
        st.session_state.step = "upload"
        st.rerun()
        
    with main_container:
        # Header actions
        col_title, col_action = st.columns([3, 1])
        with col_title:
            st.markdown(f'<h2 style="margin-top:0px;">02. Statistical Dataset Profile: {st.session_state.file_name}</h2>', unsafe_allow_html=True)
        with col_action:
            st.markdown('<div class="back-btn-container" style="text-align:right;">', unsafe_allow_html=True)
            if st.button("⬅️ Change File"):
                st.session_state.step = "upload"
                st.rerun()
            st.markdown('</div>', unsafe_allow_html=True)
            
        # Summary dimensions
        total_rows, total_cols = df.shape
        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        cat_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()
        
        st.markdown(f"""
        <div class="mono-text" style="font-size:12px; color:#555; text-transform:uppercase; margin-bottom:20px; border-bottom:1px solid #141414; padding-bottom:8px;">
            ROWS: {total_rows} &nbsp;&bull;&nbsp; COLUMNS: {total_cols} &nbsp;&bull;&nbsp; NUMERICAL: {len(num_cols)} &nbsp;&bull;&nbsp; CATEGORICAL: {len(cat_cols)}
        </div>
        """, unsafe_allow_html=True)
        
        # Grid layout for Profiler configuration & Target Selection
        col_left, col_right = st.columns([1, 2], gap="large")
        
        with col_left:
            st.markdown('<div class="editorial-card">', unsafe_allow_html=True)
            st.markdown("<h4>🎯 Model Target Configuration</h4>", unsafe_allow_html=True)
            st.markdown("<p style='font-size:12px; color:#666;'>Select your target prediction label column and optimization task mode.</p>", unsafe_allow_html=True)
            
            # Select default target column if not set
            all_cols = df.columns.tolist()
            default_target_idx = 0
            if st.session_state.target_column in all_cols:
                default_target_idx = all_cols.index(st.session_state.target_column)
            else:
                # Try to guess label targets: 'species', 'target', 'class', 'CardiacRisk', 'MedHouseValue'
                lower_cols = [c.lower() for c in all_cols]
                for guess in ["species", "cardiacrisk", "medhousevalue", "target", "class", "label"]:
                    if guess in lower_cols:
                        default_target_idx = lower_cols.index(guess)
                        break
                        
            target_col = st.selectbox(
                "Select Target Label (Y)",
                options=all_cols,
                index=default_target_idx
            )
            st.session_state.target_column = target_col
            
            # Guess problem type
            target_series = df[target_col]
            inferred_problem = "classification"
            # If target has floats, or has high unique cardinalities relative to size, assume regression
            unique_cnt = target_series.nunique()
            is_float = pd.api.types.is_float_dtype(target_series)
            if is_float or (unique_cnt > 15 and pd.api.types.is_numeric_dtype(target_series)):
                inferred_problem = "regression"
                
            problem_type_options = ["classification", "regression"]
            default_prob_idx = problem_type_options.index(inferred_problem)
            
            prob_type = st.radio(
                "Inferred Problem Type",
                options=problem_type_options,
                index=default_prob_idx,
                format_func=lambda x: "🎯 Classification (Label Prediction)" if x == "classification" else "📈 Regression (Continuous Value)"
            )
            st.session_state.problem_type = prob_type
            
            # Action button to proceed
            st.markdown("<br>", unsafe_allow_html=True)
            if st.button("Proceed to Column Exclusions ➡️"):
                st.session_state.step = "prune"
                st.rerun()
                
            st.markdown('</div>', unsafe_allow_html=True)
            
        with col_right:
            st.markdown('<div class="editorial-card">', unsafe_allow_html=True)
            st.markdown("<h4>📋 Dataset Column Index Summary</h4>", unsafe_allow_html=True)
            
            # Generate table of summary stats for all columns
            col_stats = []
            for col in all_cols:
                series = df[col]
                missing_cnt = series.isnull().sum()
                missing_pct = (missing_cnt / total_rows) * 100
                unique_vals = series.nunique()
                col_type = "Numerical" if pd.api.types.is_numeric_dtype(series) else "Categorical"
                
                col_stats.append({
                    "Column Name": col,
                    "Inferred Type": col_type,
                    "Unique Values": unique_vals,
                    "Missing Percentage": f"{missing_pct:.1f}%",
                    "Role": "🎯 Target" if col == target_col else "⚡ Feature"
                })
                
            stats_df = pd.DataFrame(col_stats)
            st.dataframe(stats_df, use_container_width=True, hide_index=True)
            st.markdown('</div>', unsafe_allow_html=True)
            
        # Target Distribution plots
        st.markdown('<div class="editorial-card">', unsafe_allow_html=True)
        st.markdown("<h4>📊 Target Distribution Visualization</h4>", unsafe_allow_html=True)
        
        target_series = df[target_col].dropna()
        if prob_type == "classification":
            counts = target_series.value_counts().reset_index()
            counts.columns = [target_col, "Count"]
            fig = px.bar(
                counts,
                x=target_col,
                y="Count",
                text="Count",
                color_discrete_sequence=["#141414"]
            )
            fig.update_layout(
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font_family="JetBrains Mono, monospace",
                font_color="#141414",
                margin=dict(l=20, r=20, t=10, b=20),
                height=250
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            fig = px.histogram(
                df,
                x=target_col,
                nbins=30,
                color_discrete_sequence=["#141414"]
            )
            fig.update_layout(
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font_family="JetBrains Mono, monospace",
                font_color="#141414",
                margin=dict(l=20, r=20, t=10, b=20),
                height=250
            )
            st.plotly_chart(fig, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

# ----------------------------------------------------
# STEP 3: Column Pruner / Feature Selector
# ----------------------------------------------------
elif st.session_state.step == "prune":
    df = st.session_state.raw_data
    if df is None:
        st.session_state.step = "upload"
        st.rerun()
        
    target_col = st.session_state.target_column
    
    with main_container:
        # Header actions
        col_title, col_action = st.columns([3, 1])
        with col_title:
            st.markdown('<h2 style="margin-top:0px;">03. Column-Removal Pipeline Step</h2>', unsafe_allow_html=True)
            st.markdown("<p style='font-size:11px; font-family: \"JetBrains Mono\"; text-transform:uppercase; color:#555;'>Discard redundant, irrelevant, or leakage columns before pipeline training</p>", unsafe_allow_html=True)
        with col_action:
            st.markdown('<div class="back-btn-container" style="text-align:right;">', unsafe_allow_html=True)
            if st.button("⬅️ Profile Dashboard"):
                st.session_state.step = "analyze"
                st.rerun()
            st.markdown('</div>', unsafe_allow_html=True)
            
        all_feature_cols = [c for c in df.columns.tolist() if c != target_col]
        
        col_p1, col_p2 = st.columns([1, 2], gap="large")
        
        with col_p1:
            st.markdown('<div class="editorial-card">', unsafe_allow_html=True)
            st.markdown("<h5>⚙️ Pruning Directives</h5>", unsafe_allow_html=True)
            st.markdown(f"""
            <div style="font-size:12px; line-height:1.6; color:#555;">
                <p>Removing columns containing highly sparse, duplicate, or index values maximizes the generalization properties of candidate classifiers.</p>
                <p>The locked target variable <span class="pill-yellow">{target_col}</span> is automatically preserved in pipeline memory.</p>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("<br>", unsafe_allow_html=True)
            # Select all or deselect all buttons
            if st.button("Exclude All Features"):
                st.session_state.excluded_columns = all_feature_cols.copy()
                st.rerun()
            if st.button("Keep All Features"):
                st.session_state.excluded_columns = []
                st.rerun()
                
            st.markdown('</div>', unsafe_allow_html=True)
            
        with col_p2:
            st.markdown('<div class="editorial-card">', unsafe_allow_html=True)
            st.markdown("<h5>📋 Feature Checklists</h5>", unsafe_allow_html=True)
            
            new_exclusions = []
            for col in all_feature_cols:
                is_excluded = col in st.session_state.excluded_columns
                checked = st.checkbox(
                    f"⛔ Exclude Column: **{col}**",
                    value=is_excluded,
                    key=f"check_excl_{col}"
                )
                if checked:
                    new_exclusions.append(col)
                    
            st.session_state.excluded_columns = new_exclusions
            st.markdown('</div>', unsafe_allow_html=True)
            
        # Global Confirm Banner
        active_feature_count = len(all_feature_cols) - len(st.session_state.excluded_columns)
        st.markdown('<div class="editorial-card">', unsafe_allow_html=True)
        col_left_b, col_right_b = st.columns([2, 1])
        with col_left_b:
            st.markdown(f"""
            <div style="font-family: 'JetBrains Mono', monospace; font-size:13px; font-weight:700;">
                ACTIVE TRAINING COLUMNS: {active_feature_count} &nbsp;&bull;&nbsp; EXCLUDED: {len(st.session_state.excluded_columns)}
            </div>
            """, unsafe_allow_html=True)
        with col_right_b:
            if active_feature_count <= 0:
                st.error("⚠️ Error: You must preserve at least one active feature for training.")
            else:
                if st.button("Proceed to Pipeline Training ⚙️"):
                    st.session_state.step = "train"
                    st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)

# ----------------------------------------------------
# STEP 4: Preprocessing & Parallel Pipeline Training
# ----------------------------------------------------
elif st.session_state.step == "train":
    df = st.session_state.raw_data
    if df is None:
        st.session_state.step = "upload"
        st.rerun()
        
    target_col = st.session_state.target_column
    prob_type = st.session_state.problem_type
    exclusions = st.session_state.excluded_columns
    
    with main_container:
        st.markdown('<h2 style="margin-top:0px;">04. Running Autonomous ML Pipeline</h2>', unsafe_allow_html=True)
        st.markdown("<p style='font-size:12px; font-family: \"JetBrains Mono\"; text-transform:uppercase; color:#555;'>Compiling active slices, performing deterministic imputation, and training 5 custom models...</p>", unsafe_allow_html=True)
        
        # We will render a fake scrolling log first to match visual fidelity, then actually run the scikit-learn models.
        log_placeholder = st.empty()
        
        logs = [
            "🚀 Initializing Autonomous AI Machine Learning Pipeline...",
            f"✅ Applied Feature Exclusions: Dropped {len(exclusions)} columns from active memory: {exclusions if exclusions else 'None'}",
            "=== Stage 1: Pipeline Compilation & Cleaning ===",
            "✅ Partitioning dataset: 80% Train split, 20% Validation split (deterministic random seed 42)",
        ]
        
        # Display logs progressively
        for i in range(1, len(logs) + 1):
            log_text = "<div class='console-container'>" + "".join([f"<div>{line}</div>" for line in logs[:i]]) + "</div>"
            log_placeholder.markdown(log_text, unsafe_allow_html=True)
            time.sleep(0.3)
            
        # Preprocessing dataset
        # 1. Slice working data
        working_cols = [c for c in df.columns if c not in exclusions]
        working_df = df[working_cols].copy()
        
        X = working_df.drop(columns=[target_col])
        y = working_df[target_col]
        
        # Convert classification target labels to strings/integers if needed, or regression target to numeric
        if prob_type == "classification":
            y = y.astype(str)
        else:
            y = pd.to_numeric(y, errors='coerce')
            
        # Determine numerical and categorical feature pipelines
        num_feats = X.select_dtypes(include=[np.number]).columns.tolist()
        cat_feats = X.select_dtypes(exclude=[np.number]).columns.tolist()
        
        # Transformers
        num_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])
        
        cat_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
            ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
        ])
        
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', num_transformer, num_feats),
                ('cat', cat_transformer, cat_feats)
            ]
        )
        
        logs.append("⚙️ Fitting Column Imputation & One-Hot Scaling Preprocessors...")
        log_text = "<div class='console-container'>" + "".join([f"<div>{line}</div>" for line in logs]) + "</div>"
        log_placeholder.markdown(log_text, unsafe_allow_html=True)
        time.sleep(0.4)
        
        # Train-validation split
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Fit preprocessor on training data
        X_train_trans = preprocessor.fit_transform(X_train)
        X_val_trans = preprocessor.transform(X_val)
        
        st.session_state.pipeline_preprocessor = preprocessor
        
        logs.append(f"📊 Feature transformation complete: {X_train_trans.shape[1]} active inputs formatted.")
        logs.append("=== Stage 2: Parallel Candidate Estimator Evaluation ===")
        log_text = "<div class='console-container'>" + "".join([f"<div>{line}</div>" for line in logs]) + "</div>"
        log_placeholder.markdown(log_text, unsafe_allow_html=True)
        time.sleep(0.4)
        
        # Prepare list of models to train
        results = []
        trained_models_instances = {}
        
        if prob_type == "classification":
            models_to_train = [
                ("logistic_regression", "Logistic Regression (One-vs-Rest)", LogisticRegression(max_iter=1000, random_state=42)),
                ("random_forest", "Random Forest Classifier (Ensemble)", RandomForestClassifier(n_estimators=100, max_depth=8, random_state=42)),
                ("gradient_boosting", "Gradient Boosting Classifier", GradientBoostingClassifier(n_estimators=100, max_depth=4, random_state=42)),
                ("support_vector_classification", "Support Vector Classifier (SVC)", SVC(probability=True, random_state=42, max_iter=2000)),
                ("knn_classification", "K-Nearest Neighbors Classifier", KNeighborsClassifier(n_neighbors=5))
            ]
            
            for m_id, m_name, model in models_to_train:
                logs.append(f"🏋️ training estimator: {m_name}...")
                log_text = "<div class='console-container'>" + "".join([f"<div>{line}</div>" for line in logs]) + "</div>"
                log_placeholder.markdown(log_text, unsafe_allow_html=True)
                
                start_t = time.time()
                model.fit(X_train_trans, y_train)
                elapsed = time.time() - start_t
                
                # Predictions
                y_pred = model.predict(X_val_trans)
                y_prob = model.predict_proba(X_val_trans)
                
                # Metrics
                acc = accuracy_score(y_val, y_pred)
                
                # F1 Score
                unique_classes = np.unique(y_val)
                if len(unique_classes) > 2:
                    f1 = f1_score(y_val, y_pred, average='weighted')
                else:
                    # binary
                    f1 = f1_score(y_val, y_pred, pos_label=unique_classes[-1])
                    
                # ROC-AUC
                try:
                    if len(unique_classes) == 2:
                        roc = roc_auc_score(y_val, y_prob[:, 1])
                    else:
                        roc = roc_auc_score(y_val, y_prob, multi_class='ovr')
                except Exception:
                    roc = 0.5  # fallback if single class present
                    
                results.append({
                    "modelId": m_id,
                    "modelName": m_name,
                    "accuracy": acc,
                    "f1": f1,
                    "roc": roc,
                    "time": elapsed
                })
                trained_models_instances[m_id] = model
                logs.append(f"   ↳ Finished in {elapsed:.3f}s. Validation Acc: {acc:.4f} &bull; F1: {f1:.4f}")
                time.sleep(0.2)
                
        else:
            # Regression task
            models_to_train = [
                ("linear_regression", "L2-Regularized Ridge Regression", Ridge(random_state=42)),
                ("random_forest", "Random Forest Regressor (Ensemble)", RandomForestRegressor(n_estimators=100, max_depth=8, random_state=42)),
                ("gradient_boosting", "Gradient Boosting Regressor", GradientBoostingRegressor(n_estimators=100, max_depth=4, random_state=42)),
                ("support_vector_regression", "Support Vector Regressor", SVR(max_iter=2000)),
                ("knn_regression", "K-Nearest Neighbors Regressor", KNeighborsRegressor(n_neighbors=5))
            ]
            
            for m_id, m_name, model in models_to_train:
                logs.append(f"🏋️ training estimator: {m_name}...")
                log_text = "<div class='console-container'>" + "".join([f"<div>{line}</div>" for line in logs]) + "</div>"
                log_placeholder.markdown(log_text, unsafe_allow_html=True)
                
                start_t = time.time()
                model.fit(X_train_trans, y_train)
                elapsed = time.time() - start_t
                
                # Predictions
                y_pred = model.predict(X_val_trans)
                
                # Metrics
                r2 = r2_score(y_val, y_pred)
                mae = mean_absolute_error(y_val, y_pred)
                rmse = np.sqrt(mean_squared_error(y_val, y_pred))
                
                results.append({
                    "modelId": m_id,
                    "modelName": m_name,
                    "r2": r2,
                    "mae": mae,
                    "rmse": rmse,
                    "time": elapsed
                })
                trained_models_instances[m_id] = model
                logs.append(f"   ↳ Finished in {elapsed:.3f}s. Validation R²: {r2:.4f} &bull; MAE: {mae:.4f}")
                time.sleep(0.2)
                
        logs.append("=== Stage 3: Optimization & Auto-Selection Complete ===")
        logs.append("🎉 Pipeline execution successful. Directing to Comparative Matrix.")
        log_text = "<div class='console-container'>" + "".join([f"<div>{line}</div>" for line in logs]) + "</div>"
        log_placeholder.markdown(log_text, unsafe_allow_html=True)
        time.sleep(1.0)
        
        # Auto-select the top model by primary metric
        # Classification primary: F1-score
        # Regression primary: R2-score
        if prob_type == "classification":
            sorted_res = sorted(results, key=lambda x: x["f1"], reverse=True)
        else:
            sorted_res = sorted(results, key=lambda x: x["r2"], reverse=True)
            
        st.session_state.trained_models = results
        st.session_state.model_classes = trained_models_instances
        st.session_state.selected_model_id = sorted_res[0]["modelId"]
        
        # Proceed
        st.session_state.step = "compare"
        st.rerun()

# ----------------------------------------------------
# STEP 5: Performance Comparative Matrix
# ----------------------------------------------------
elif st.session_state.step == "compare":
    results = st.session_state.trained_models
    prob_type = st.session_state.problem_type
    
    if results is None:
        st.session_state.step = "train"
        st.rerun()
        
    # Sort results
    if prob_type == "classification":
        sorted_results = sorted(results, key=lambda x: x["f1"], reverse=True)
        primary_metric = "f1"
        primary_label = "F1-Score"
    else:
        sorted_results = sorted(results, key=lambda x: x["r2"], reverse=True)
        primary_metric = "r2"
        primary_label = "R²-Score"
        
    best_model_id = sorted_results[0]["modelId"]
    
    with main_container:
        st.markdown('<h2 style="margin-top:0px;">05. Pipeline Comparison Matrix</h2>', unsafe_allow_html=True)
        st.markdown("<p style='font-size:12px; font-family: \"JetBrains Mono\"; text-transform:uppercase; color:#555;'>Interactive benchmarking validation across all 5 candidate estimators</p>", unsafe_allow_html=True)
        
        # Validation Chart (Horizontal Bar Chart)
        st.markdown('<div class="editorial-card">', unsafe_allow_html=True)
        st.markdown(f"<h4>📊 Validation Model Rankings (Sorted by {primary_label})</h4>", unsafe_allow_html=True)
        
        chart_data = pd.DataFrame(sorted_results)
        fig = px.bar(
            chart_data,
            x=primary_metric,
            y="modelName",
            orientation='h',
            text=primary_metric,
            color_discrete_sequence=["#141414"]
        )
        fig.update_traces(texttemplate='%{text:.4f}', textposition='outside')
        fig.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font_family="JetBrains Mono, monospace",
            font_color="#141414",
            margin=dict(l=20, r=20, t=10, b=20),
            height=250,
            xaxis=dict(title=primary_label, range=[0, 1.1] if prob_type == "classification" else None),
            yaxis=dict(title="")
        )
        st.plotly_chart(fig, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Benchmark grid cards
        st.markdown("<h4>📋 Pipeline Model Indices</h4>", unsafe_allow_html=True)
        cols_grid = st.columns(5, gap="small")
        
        for idx, res in enumerate(sorted_results):
            m_id = res["modelId"]
            is_best = m_id == best_model_id
            is_selected = m_id == st.session_state.selected_model_id
            
            with cols_grid[idx]:
                card_style = "border: 2px solid #141414;" if is_selected else "border: 1px solid #141414;"
                st.markdown(f"""
                <div class="editorial-card" style="margin-bottom:12px; {card_style} min-height: 250px; display: flex; flex-direction: column; justify-content: space-between;">
                    <div>
                        <div style="font-family:'JetBrains Mono'; font-size:10px; color:#555; text-transform:uppercase; font-weight:700; margin-bottom:4px;">
                            RANK {idx+1} { '⭐️ BEST' if is_best else '' }
                        </div>
                        <h5 style="margin-top:0px; font-size:14px; line-height:1.2;">{res["modelName"]}</h5>
                    </div>
                """, unsafe_allow_html=True)
                
                # Show key metrics in mono-text format
                if prob_type == "classification":
                    st.markdown(f"""
                    <div style="font-family:'JetBrains Mono'; font-size:12px; line-height:1.6; margin-top:10px; margin-bottom:15px;">
                        <div>F1: {res["f1"]:.4f}</div>
                        <div>ACC: {res["accuracy"]:.4f}</div>
                        <div>AUC: {res["roc"]:.4f}</div>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div style="font-family:'JetBrains Mono'; font-size:12px; line-height:1.6; margin-top:10px; margin-bottom:15px;">
                        <div>R²: {res["r2"]:.4f}</div>
                        <div>MAE: {res["mae"]:.4f}</div>
                        <div>RMSE: {res["rmse"]:.4f}</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Set selection buttons
                if is_selected:
                    st.markdown('<span class="pill-green">ACTIVE DEPLOYMENT</span>', unsafe_allow_html=True)
                else:
                    if st.button("Deploy Model", key=f"sel_btn_{m_id}"):
                        st.session_state.selected_model_id = m_id
                        st.rerun()
                st.markdown("</div>", unsafe_allow_html=True)
                
        # Action Proceed Section
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<div class="editorial-card">', unsafe_allow_html=True)
        col_la, col_ra = st.columns([2, 1])
        with col_la:
            deployed_name = next(r["modelName"] for r in results if r["modelId"] == st.session_state.selected_model_id)
            st.markdown(f"""
            <div style="font-family: 'JetBrains Mono', monospace; font-size:13px; font-weight:700;">
                ACTIVE SYSTEM PIPELINE: <span class="pill-yellow">{deployed_name}</span>
            </div>
            """, unsafe_allow_html=True)
        with col_ra:
            if st.button("Proceed to Prediction Console ➡️"):
                st.session_state.step = "predict"
                st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)

# ----------------------------------------------------
# STEP 6: Real-time Deployed Inference
# ----------------------------------------------------
elif st.session_state.step == "predict":
    df = st.session_state.raw_data
    exclusions = st.session_state.excluded_columns
    target_col = st.session_state.target_column
    prob_type = st.session_state.problem_type
    
    if df is None or st.session_state.selected_model_id is None:
        st.session_state.step = "upload"
        st.rerun()
        
    model = st.session_state.model_classes[st.session_state.selected_model_id]
    preprocessor = st.session_state.pipeline_preprocessor
    
    with main_container:
        # Header actions
        col_title, col_action = st.columns([3, 1])
        with col_title:
            st.markdown('<h2 style="margin-top:0px;">06. Live Deployed Prediction Console</h2>', unsafe_allow_html=True)
            st.markdown("<p style='font-size:12px; font-family: \"JetBrains Mono\"; text-transform:uppercase; color:#555;'>Perform manual single-record or bulk-batch inferences</p>", unsafe_allow_html=True)
        with col_action:
            st.markdown('<div class="back-btn-container" style="text-align:right;">', unsafe_allow_html=True)
            if st.button("⬅️ Compare Models"):
                st.session_state.step = "compare"
                st.rerun()
            st.markdown('</div>', unsafe_allow_html=True)
            
        col_form, col_batch = st.columns([1, 1], gap="large")
        
        with col_form:
            st.markdown('<div class="editorial-card">', unsafe_allow_html=True)
            st.markdown("<h4>📋 Single-Record Manual Inference</h4>", unsafe_allow_html=True)
            st.markdown("<p style='font-size:11px; color:#666; font-family:\"JetBrains Mono\";'>Input custom sample parameters. Pruned/excluded columns are automatically handled.</p>", unsafe_allow_html=True)
            
            # Auto-generate input fields for features
            features_df = df.drop(columns=[target_col] + list(exclusions), errors='ignore')
            
            form_inputs = {}
            for col in features_df.columns:
                series = features_df[col]
                if pd.api.types.is_numeric_dtype(series):
                    # Numerical field slider/number box
                    min_val = float(series.min())
                    max_val = float(series.max())
                    mean_val = float(series.mean())
                    
                    form_inputs[col] = st.number_input(
                        f"⚡ Feature: {col}",
                        min_value=min_val,
                        max_value=max_val,
                        value=mean_val,
                        key=f"input_{col}"
                    )
                else:
                    # Categorical dropdown
                    unique_vals = series.dropna().unique().tolist()
                    form_inputs[col] = st.selectbox(
                        f"⚡ Feature: {col}",
                        options=unique_vals,
                        key=f"input_{col}"
                    )
                    
            if st.button("Generate Real-time Prediction ⚡"):
                # Format into single row dataframe
                input_df = pd.DataFrame([form_inputs])
                
                # Apply preprocessor scaling & imputation
                input_trans = preprocessor.transform(input_df)
                
                # Predict
                pred = model.predict(input_trans)[0]
                
                st.markdown("<br><hr style='border-color:#141414;'><br>", unsafe_allow_html=True)
                st.markdown("<h5>🎯 Prediction Output Summary</h5>", unsafe_allow_html=True)
                if prob_type == "classification":
                    # Display confidence probabilities
                    y_prob = model.predict_proba(input_trans)[0]
                    classes = model.classes_
                    st.success(f"PREDICTED CLASS LABEL: **{pred}**")
                    
                    prob_df = pd.DataFrame({
                        "Class": classes,
                        "Probability": y_prob
                    })
                    st.dataframe(prob_df, use_container_width=True, hide_index=True)
                else:
                    st.success(f"PREDICTED VALUE: **{pred:.4f}**")
            st.markdown('</div>', unsafe_allow_html=True)
            
        with col_batch:
            st.markdown('<div class="editorial-card">', unsafe_allow_html=True)
            st.markdown("<h4>📂 Bulk Batch Spreadsheet Testing</h4>", unsafe_allow_html=True)
            st.markdown("<p style='font-size:11px; color:#666; font-family:\"JetBrains Mono\";'>Upload raw data files. We will automatically pre-clean features, match exclusions, generate predictions, and return a downloadable CSV sheet.</p>", unsafe_allow_html=True)
            
            batch_file = st.file_uploader(
                "Upload a testing file (CSV/XLSX)",
                type=["csv", "xlsx", "xls"],
                key="batch_file_uploader"
            )
            
            if batch_file is not None:
                try:
                    ext = batch_file.name.split(".")[-1].lower()
                    if ext == "csv":
                        batch_df = pd.read_csv(batch_file)
                    else:
                        batch_df = pd.read_excel(batch_file)
                        
                    # Drop exclusions if they exist
                    pred_df = batch_df.copy()
                    
                    # Target column is optional in test data. If it exists, drop it to match feature set
                    eval_test_df = batch_df.drop(columns=[target_col], errors='ignore')
                    
                    # Ensure all required features are present
                    missing_features = [c for c in features_df.columns if c not in eval_test_df.columns]
                    if len(missing_features) > 0:
                        st.error(f"⚠️ Test dataset is missing the following active features: {missing_features}")
                    else:
                        # Reorder columns to match fit preprocessor X format exactly
                        test_features_X = eval_test_df[features_df.columns]
                        
                        # Preprocess
                        test_trans = preprocessor.transform(test_features_X)
                        
                        # Predict
                        batch_predictions = model.predict(test_trans)
                        
                        pred_df[f"Predicted_{target_col}"] = batch_predictions
                        
                        st.success("✅ Batch predictions compiled successfully.")
                        st.dataframe(pred_df.head(10), use_container_width=True)
                        
                        # Export to CSV
                        csv_buffer = io.StringIO()
                        pred_df.to_csv(csv_buffer, index=False)
                        st.download_button(
                            label="⬇️ Download Prediction Spreadsheet (CSV)",
                            data=csv_buffer.getvalue(),
                            file_name=f"predictions_{batch_file.name}",
                            mime="text/csv"
                        )
                except Exception as e:
                    st.error(f"Batch prediction failed: {str(e)}")
            st.markdown('</div>', unsafe_allow_html=True)
