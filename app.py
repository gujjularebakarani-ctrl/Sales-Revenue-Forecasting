"""
app.py
======
Main Streamlit Application for Final-Year B.Tech Major Project:
Sales and Revenue Forecasting with Seasonality via Bayesian Hierarchical Modeling
for Uncertainty Quantification, Applied to Retail Demand Planning and Inventory Optimization.

Project Guide: Dr. P. Sudhakar
Technology Stack: Python, Pandas, NumPy, PyMC, ArviZ, Scikit-learn, Plotly, Streamlit
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import os
import sys
import io

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.data_processing import (
    load_and_validate_data,
    aggregate_sales_data,
    engineer_forecasting_features,
    split_train_test,
    format_date
)
from src.bayesian_model import BayesianHierarchicalForecaster
from src.baseline_models import (
    RidgeBaselineForecaster,
    RandomForestBaselineForecaster,
    HoltWintersBaselineForecaster
)
from src.seasonality import (
    perform_seasonal_decomposition,
    compute_monthly_seasonality,
    compute_quarterly_seasonality,
    compute_fourier_spectrum,
    compute_holiday_promotional_lift
)
from src.inventory import (
    calculate_classical_inventory_parameters,
    calculate_bayesian_inventory_parameters,
    simulate_inventory_trajectory,
    generate_service_level_tradeoff_curve
)
from src.evaluation import compute_forecast_metrics, compare_multiple_models
from src.visualization import (
    plot_forecast_with_credible_intervals,
    plot_model_comparison,
    plot_seasonality_decomposition,
    plot_monthly_seasonality,
    plot_inventory_trajectory,
    plot_service_level_cost_curve,
    plot_hierarchical_shrinkage,
    plot_lead_time_demand_distribution,
    apply_chart_theme
)

# ---------------------------------------------------------
# Streamlit Page Configuration
# ---------------------------------------------------------
st.set_page_config(
    page_title="Bayesian Retail Forecasting & Inventory Optimization",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------------------------------------------------
# Custom Modern CSS (Midnight Slate / Dark Tech Theme)
# ---------------------------------------------------------
st.markdown("""
<style>
    /* Main container and font */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    
    /* Top banner */
    .hero-banner {
        background: linear-gradient(135deg, #1E293B 0%, #0F172A 100%);
        border: 1px solid #334155;
        border-radius: 12px;
        padding: 24px;
        margin-bottom: 24px;
        box-shadow: 0 4px 20px -2px rgba(0, 0, 0, 0.4);
    }
    .hero-title {
        color: #F8FAFC;
        font-size: 26px;
        font-weight: 700;
        margin-bottom: 8px;
        line-height: 1.3;
    }
    .hero-subtitle {
        color: #94A3B8;
        font-size: 14px;
        margin-bottom: 12px;
    }
    .guide-badge {
        display: inline-block;
        background: rgba(59, 130, 246, 0.15);
        color: #60A5FA;
        border: 1px solid #3B82F6;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 12px;
        font-weight: 600;
    }
    
    /* KPI Metric Cards */
    .kpi-card {
        background-color: #1E293B;
        border: 1px solid #334155;
        border-radius: 10px;
        padding: 18px 20px;
        text-align: left;
        margin-bottom: 16px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.2);
    }
    .kpi-title {
        color: #94A3B8;
        font-size: 12px;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 6px;
        font-weight: 600;
    }
    .kpi-value {
        color: #F8FAFC;
        font-size: 26px;
        font-weight: 700;
        margin-bottom: 4px;
    }
    .kpi-subtext {
        color: #10B981;
        font-size: 12px;
        font-weight: 500;
    }
    
    /* Callout & Math boxes */
    .math-box {
        background-color: #0F172A;
        border-left: 4px solid #3B82F6;
        padding: 14px 18px;
        border-radius: 0 8px 8px 0;
        margin: 14px 0;
        font-size: 14px;
        color: #E2E8F0;
    }
    
    .academic-box {
        background-color: rgba(139, 92, 246, 0.08);
        border: 1px solid #8B5CF6;
        border-radius: 8px;
        padding: 16px;
        margin: 16px 0;
    }
    
    /* Sidebar branding */
    .sidebar-brand {
        padding: 12px 0 20px 0;
        text-align: center;
        border-bottom: 1px solid #334155;
        margin-bottom: 16px;
    }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# Data Caching & Management
# ---------------------------------------------------------
@st.cache_data
def get_default_dataset() -> pd.DataFrame:
    """Load and cache default benchmark synthetic retail dataset."""
    csv_path = os.path.join(os.path.dirname(__file__), "data", "retail_sales_sample.csv")
    if not os.path.exists(csv_path):
        from data.generate_dataset import generate_retail_sales_data
        df = generate_retail_sales_data(num_weeks=156, random_seed=42)
        df.to_csv(csv_path, index=False)
        return df
    df, _ = load_and_validate_data(csv_path)
    return df

# Initialize Session State
if "raw_df" not in st.session_state:
    st.session_state.raw_df = get_default_dataset()
if "bayes_model" not in st.session_state:
    st.session_state.bayes_model = None
if "forecast_results" not in st.session_state:
    st.session_state.forecast_results = None
if "baseline_results" not in st.session_state:
    st.session_state.baseline_results = {}
if "selected_granularity" not in st.session_state:
    st.session_state.selected_granularity = "Enterprise Total"

# ---------------------------------------------------------
# Sidebar Navigation
# ---------------------------------------------------------
with st.sidebar:
    st.markdown("""
    <div class="sidebar-brand">
        <h3 style="color:#3B82F6; margin:0; font-size:20px;">⚡ RetailBayes AI</h3>
        <p style="color:#94A3B8; margin:4px 0 0 0; font-size:12px;">Hierarchical Forecasting & Inventory</p>
    </div>
    """, unsafe_allow_html=True)
    
    page = st.radio(
        "Navigation",
        [
            "Home",
            "Data Overview",
            "Exploratory Analysis",
            "Sales Forecasting",
            "Bayesian Analysis",
            "Seasonality Analysis",
            "Inventory Optimization",
            "Model Evaluation",
            "About Project"
        ],
        index=0
    )
    
    st.markdown("---")
    st.markdown("""
    **Project Guide:**  
    <span style="color:#60A5FA; font-weight:600;">Dr. P. Sudhakar</span>  
    *Final-Year B.Tech Major Project*
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    st.markdown("""
    <div style="font-size:11px; color:#64748B;">
        <b>Technology Stack:</b><br>
        PyMC 6 • ArviZ • Scikit-learn<br>
        Pandas • NumPy • Plotly • Streamlit
    </div>
    """, unsafe_allow_html=True)

# ---------------------------------------------------------
# PAGE 1: HOME
# ---------------------------------------------------------
if page == "Home":
    st.markdown("""
    <div class="hero-banner">
        <span class="guide-badge">Major Project Guide: Dr. P. Sudhakar</span>
        <div class="hero-title">
            Sales & Revenue Forecasting with Seasonality via Bayesian Hierarchical Modeling
        </div>
        <div class="hero-subtitle">
            Uncertainty Quantification for Retail Demand Planning and Rigorous Inventory Optimization
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # 4 Highlights KPI cards
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown("""
        <div class="kpi-card">
            <div class="kpi-title">Modeling Engine</div>
            <div class="kpi-value" style="color:#3B82F6;">Bayesian</div>
            <div class="kpi-subtext">PyMC NUTS & ADVI</div>
        </div>
        """, unsafe_allow_html=True)
    with c2:
        st.markdown("""
        <div class="kpi-card">
            <div class="kpi-title">Hierarchy</div>
            <div class="kpi-value" style="color:#8B5CF6;">Partial Pooling</div>
            <div class="kpi-subtext">Stores & Categories</div>
        </div>
        """, unsafe_allow_html=True)
    with c3:
        st.markdown("""
        <div class="kpi-card">
            <div class="kpi-title">Seasonality</div>
            <div class="kpi-value" style="color:#10B981;">Harmonics</div>
            <div class="kpi-subtext">Fourier & Holiday Lifts</div>
        </div>
        """, unsafe_allow_html=True)
    with c4:
        st.markdown("""
        <div class="kpi-card">
            <div class="kpi-title">Inventory Control</div>
            <div class="kpi-value" style="color:#F59E0B;">Empirical ROP</div>
            <div class="kpi-subtext">Posterior Lead-Time VaR</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("### 🎯 Executive Summary & Major Project Highlights")
    st.write(
        """
        In contemporary retail and supply chain ecosystems, demand forecasting is plagued by extreme volatility,
        intermittent stockouts, promotions, and changing seasonal dynamics. Deterministic point forecasts
        routinely fail because they conceal the underlying probability distribution of demand, triggering either
        costly inventory obsolescence or severe lost-sales penalties.
        
        This academic project establishes a production-grade **Bayesian Hierarchical Forecasting Framework** that:
        1. **Quantifies Uncertainty with Posterior Credible Intervals:** Evaluates full predictive distributions rather than single point predictions.
        2. **Applies Bayesian Partial Pooling (Shrinkage):** Borrows statistical strength across retail stores and product categories, preventing overfitting in low-volume stores while preserving distinct local nuances.
        3. **Decomposes Multi-scale Seasonality:** Incorporates Fourier sinusoidal harmonics for annual cycles, monthly spikes, and holiday promotional lifts.
        4. **Transforms Forecast Distributions into Optimal Inventory Decisions:** Computes exact Safety Stock and Reorder Points (ROP) by sampling lead-time demand directly from MCMC posterior predictive trajectories.
        """
    )
    
    st.markdown("---")
    st.markdown("### 🗺️ Project Architecture & End-to-End Workflow")
    st.markdown("""
    ```mermaid
    flowchart LR
        subgraph Data["1. Data Ingestion & Engineering"]
            RawData["Multi-Store Retail Dataset"] --> Cleaning["Validation & Imputation"]
            Cleaning --> Feats["Fourier Harmonics & Calendar Lags"]
        end
        
        subgraph Models["2. Hierarchical Forecasting Engine"]
            Feats --> PyMC["PyMC Bayesian Hierarchical Model"]
            Feats --> Baselines["Ridge, Random Forest, Holt-Winters"]
            PyMC --> PostPred["Posterior Predictive Draws (MCMC)"]
        end
        
        subgraph Optimization["3. Supply Chain Decisioning"]
            PostPred --> ROP["Lead-Time Demand Quantiles (ROP)"]
            PostPred --> SS["Uncertainty-Aware Safety Stock"]
            ROP --> TrajSim["Dynamic Trajectory Simulation"]
            SS --> CostOpt["Total Cost Optimization Curve"]
        end
        
        subgraph UI["4. Streamlit Dashboard"]
            PostPred --> Vis["Interactive Plotly Uncertainty Ribbons"]
            CostOpt --> Vis
        end
    ```
    """)

    st.markdown("### 🚀 Quick-Start Exploration")
    colA, colB, colC = st.columns(3)
    with colA:
        st.info("**1. Explore Data & Trends**\n\nNavigate to **Data Overview** or **Exploratory Analysis** to inspect store distributions, promotional lifts, and correlations.")
    with colB:
        st.info("**2. Run Bayesian Forecast**\n\nGo to **Sales Forecasting** to fit the PyMC model, view 50%/80%/95% credible ribbons, and simulate future promotional scenarios.")
    with colC:
        st.info("**3. Optimize Inventory**\n\nVisit **Inventory Optimization** to adjust service levels, compute Bayesian vs Classical safety stock, and run inventory simulation.")

# ---------------------------------------------------------
# PAGE 2: DATA OVERVIEW
# ---------------------------------------------------------
elif page == "Data Overview":
    st.title("📂 Retail Dataset Management & Schema Overview")
    st.markdown("Inspect, validate, clean, or upload multi-store retail transaction datasets.")
    
    tab1, tab2 = st.tabs(["📊 Current Active Dataset", "⬆️ Upload Custom CSV"])
    
    with tab1:
        df = st.session_state.raw_df
        
        # High-level dataset stats
        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("Total Records", f"{len(df):,}")
        date_span = f"{format_date(df['date'].min())} to {format_date(df['date'].max())}" if not df.empty else "N/A"
        m2.metric("Date Span", date_span)
        m3.metric("Stores", len(df["store_id"].unique()) if "store_id" in df.columns else 1)
        m4.metric("Categories", len(df["category"].unique()) if "category" in df.columns else 1)
        m5.metric("Total Historical Revenue", f"${df['revenue'].sum():,.0f}")
        
        st.markdown("#### Sample Dataset Preview (Top 50 Records)")
        st.dataframe(df.head(50), use_container_width=True)
        
        c_left, c_right = st.columns(2)
        with c_left:
            st.markdown("#### Column Data Types & Non-Null Counts")
            info_df = pd.DataFrame({
                "Column": df.columns,
                "Data Type": [str(t) for t in df.dtypes],
                "Non-Null Count": df.notnull().sum().values,
                "Null Count": df.isnull().sum().values
            })
            st.dataframe(info_df, use_container_width=True)
            
        with c_right:
            st.markdown("#### Numerical Summary Statistics")
            st.dataframe(df.describe().T[["mean", "std", "min", "50%", "max"]], use_container_width=True)
            
        # Download sample CSV button
        st.markdown("---")
        csv_buffer = io.StringIO()
        df.to_csv(csv_buffer, index=False)
        st.download_button(
            label="📥 Download Benchmark Sample Dataset (CSV)",
            data=csv_buffer.getvalue(),
            file_name="retail_sales_sample.csv",
            mime="text/csv",
            help="Download the complete benchmark retail dataset for local testing."
        )

    with tab2:
        st.markdown("#### Upload Your Own Retail Dataset (CSV)")
        st.write("Ensure your CSV includes at least `date`, `store_id`, `category`, `units_sold`, and `revenue`.")
        
        uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"])
        if uploaded_file is not None:
            new_df, report = load_and_validate_data(uploaded_file)
            if report["is_valid"]:
                st.success(f"✅ Successfully validated CSV! Found {report['clean_rows']} records.")
                if report["warnings"]:
                    for w in report["warnings"]:
                        st.warning(f"Notice: {w}")
                if st.button("Apply & Set as Active Dataset"):
                    st.session_state.raw_df = new_df
                    st.session_state.bayes_model = None
                    st.session_state.forecast_results = None
                    st.rerun()
            else:
                st.error("❌ Dataset schema validation failed:")
                for err in report["errors"]:
                    st.write(f"- {err}")

# ---------------------------------------------------------
# PAGE 3: EXPLORATORY ANALYSIS
# ---------------------------------------------------------
elif page == "Exploratory Analysis":
    st.title("🔍 Exploratory Data Analysis & Retail Insights")
    st.markdown("Uncover key empirical patterns, store volume disparities, promotional elasticities, and correlations.")
    
    df = st.session_state.raw_df
    
    # Aggregated weekly series
    weekly_total = aggregate_sales_data(df, freq="W-MON", group_by=None)
    
    # 4 Quick KPIs
    k1, k2, k3, k4 = st.columns(4)
    avg_weekly_rev = weekly_total["revenue"].mean()
    total_rev = df["revenue"].sum()
    promo_rate = (df["on_promotion"].mean() * 100) if "on_promotion" in df.columns else 0.0
    holiday_count = df["holiday_flag"].sum() if "holiday_flag" in df.columns else 0
    
    k1.metric("Total Cumulative Revenue", f"${total_rev:,.0f}")
    k2.metric("Mean Weekly Enterprise Sales", f"${avg_weekly_rev:,.0f}")
    k3.metric("Promotional Penetration", f"{promo_rate:.1f}%")
    k4.metric("Holiday Impact Events", f"{int(holiday_count):,}")
    
    st.markdown("---")
    
    # Time Series Trends
    st.subheader("1. Historical Revenue Trajectory Over Time")
    fig_trend = px.line(
        weekly_total,
        x="date",
        y="revenue",
        title="Enterprise Weekly Revenue (Aggregated Across All Stores & Categories)",
        labels={"date": "Week", "revenue": "Revenue ($)"},
        color_discrete_sequence=["#3B82F6"]
    )
    apply_chart_theme(fig_trend, height=420)
    st.plotly_chart(fig_trend, use_container_width=True)
    
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("2. Store Performance Comparison")
        if "store_name" in df.columns:
            store_agg = df.groupby("store_name")["revenue"].agg(["sum", "mean"]).reset_index()
            fig_store = px.bar(
                store_agg,
                x="store_name",
                y="sum",
                color="store_name",
                title="Total Revenue by Store Location",
                labels={"sum": "Total Revenue ($)", "store_name": "Store"},
                color_discrete_sequence=px.colors.qualitative.Prism
            )
            apply_chart_theme(fig_store, height=400)
            st.plotly_chart(fig_store, use_container_width=True)
            
    with col2:
        st.subheader("3. Category Revenue Distribution")
        if "category" in df.columns:
            cat_agg = df.groupby("category")["revenue"].sum().reset_index()
            fig_cat = px.pie(
                cat_agg,
                names="category",
                values="revenue",
                title="Category Market Share Breakdown",
                hole=0.45,
                color_discrete_sequence=px.colors.qualitative.Safe
            )
            fig_cat.update_layout(
                paper_bgcolor="#1E293B",
                font={"color": "#F8FAFC"},
                height=400,
                legend={"font": {"color": "#F8FAFC"}}
            )
            st.plotly_chart(fig_cat, use_container_width=True)

    st.markdown("---")
    
    col3, col4 = st.columns(2)
    with col3:
        st.subheader("4. Promotional Lift & Demand Elasticity")
        if "on_promotion" in df.columns and "unit_price" in df.columns:
            fig_scatter = px.scatter(
                df.sample(min(800, len(df)), random_state=42),
                x="unit_price",
                y="units_sold",
                color="on_promotion",
                trendline="ols",
                title="Price Elasticity & Promotional Lift (Sampled Transactions)",
                labels={"unit_price": "Unit Price ($)", "units_sold": "Units Sold", "on_promotion": "Promo Flag"},
                color_discrete_map={0: "#94A3B8", 1: "#10B981"}
            )
            apply_chart_theme(fig_scatter, height=420)
            st.plotly_chart(fig_scatter, use_container_width=True)
            
    with col4:
        st.subheader("5. Feature Correlation Heatmap")
        num_cols = ["units_sold", "unit_price", "revenue", "on_promotion", "holiday_flag", "lead_time_days"]
        avail_num = [c for c in num_cols if c in df.columns]
        corr_matrix = df[avail_num].corr()
        
        fig_corr = px.imshow(
            corr_matrix,
            text_auto=".2f",
            color_continuous_scale="Viridis",
            title="Correlation Matrix of Supply Chain & Financial Variables"
        )
        fig_corr.update_layout(
            paper_bgcolor="#1E293B",
            font={"color": "#F8FAFC"},
            height=420
        )
        st.plotly_chart(fig_corr, use_container_width=True)

# ---------------------------------------------------------
# PAGE 4: SALES FORECASTING
# ---------------------------------------------------------
elif page == "Sales Forecasting":
    st.title("📈 Bayesian Posterior Predictive Forecasting")
    st.markdown("Generate probabilistic out-of-sample forecasts with 50%, 80%, and 95% Credible Intervals.")
    
    df = st.session_state.raw_df
    
    # Filtering / Aggregation controls
    f_col1, f_col2, f_col3, f_col4 = st.columns(4)
    with f_col1:
        target_metric = st.selectbox("Target Metric", ["revenue", "units_sold"], index=0)
    with f_col2:
        hierarchy_level = st.selectbox(
            "Hierarchy Level",
            ["Enterprise Total", "By Store", "By Category"],
            index=0
        )
    with f_col3:
        test_weeks = st.slider("Holdout Validation Horizon (Weeks)", min_value=6, max_value=26, value=12)
    with f_col4:
        inference_mode = st.selectbox(
            "Bayesian Inference Method",
            ["ADVI (Fast Variational Inference)", "NUTS (Full MCMC Sampling)"],
            index=0,
            help="ADVI runs in ~2-4 seconds. NUTS runs full Markov Chain Monte Carlo in ~15-25 seconds."
        )

    # Prepare data based on selection
    if hierarchy_level == "Enterprise Total":
        agg_df = aggregate_sales_data(df, freq="W-MON", group_by=None, target_metric=target_metric)
    elif hierarchy_level == "By Store":
        store_names = sorted(df["store_name"].unique())
        selected_store = st.selectbox("Select Store", store_names)
        sub_df = df[df["store_name"] == selected_store]
        agg_df = aggregate_sales_data(sub_df, freq="W-MON", group_by=["store_id"], target_metric=target_metric)
    else:
        cat_names = sorted(df["category"].unique())
        selected_cat = st.selectbox("Select Category", cat_names)
        sub_df = df[df["category"] == selected_cat]
        agg_df = aggregate_sales_data(sub_df, freq="W-MON", group_by=["category"], target_metric=target_metric)
        
    feat_df = engineer_forecasting_features(agg_df, fourier_harmonics=2, target_col=target_metric)
    train_df, test_df = split_train_test(feat_df, test_periods=test_weeks)
    
    st.markdown("---")
    
    # Trigger model fitting button
    btn_col1, btn_col2 = st.columns([2, 5])
    with btn_col1:
        run_model = st.button("🚀 Fit Bayesian Model & Generate Forecast", type="primary")
        
    if run_model or st.session_state.forecast_results is not None:
        if run_model:
            with st.spinner(f"Fitting Bayesian Model using {inference_mode}..."):
                method_code = "advi" if "ADVI" in inference_mode else "nuts"
                forecaster = BayesianHierarchicalForecaster(
                    fourier_harmonics=2,
                    target_col=target_metric,
                    inference_method=method_code,
                    draws=400,
                    tune=400,
                    chains=2 if method_code == "nuts" else 1,
                    random_seed=42
                )
                forecaster.fit(train_df, verbose=False)
                
                # Predict on holdout test set
                pred_df = forecaster.predict(test_df, credible_levels=[0.50, 0.80, 0.95])
                
                st.session_state.bayes_model = forecaster
                st.session_state.forecast_results = pred_df
                st.session_state.last_train_df = train_df
                st.session_state.last_target_metric = target_metric
                
        pred_df = st.session_state.forecast_results
        train_df = st.session_state.last_train_df
        cur_target = st.session_state.last_target_metric
        
        # Uncertainty Ribbons Plot
        st.subheader("1. Posterior Predictive Forecast with Uncertainty Ribbons")
        fig_forecast = plot_forecast_with_credible_intervals(
            train_df=train_df,
            pred_df=pred_df,
            actual_col=cur_target,
            title=f"Bayesian Predictive Distribution vs Actual Ground Truth ({cur_target.capitalize()})"
        )
        st.plotly_chart(fig_forecast, use_container_width=True)
        
        # In-depth interpretation
        st.markdown("""
        <div class="math-box">
            <b>Interpreting Uncertainty Ribbons:</b><br>
            • <b>Dark Blue Center Line:</b> Posterior median prediction $\\hat{y}_{t}$.<br>
            • <b>Inner Ribbon (50% CI / IQR):</b> Represents the central 50% probability mass. There is an even chance of the actual sales landing within this band.<br>
            • <b>Mid Ribbon (80% CI):</b> 80% credible density used for lean inventory reorder points.<br>
            • <b>Outer Ribbon (95% CI):</b> High-density 95% credible interval. Notice how the ribbon widens into holiday peak periods, capturing promotional tail risk.
        </div>
        """, unsafe_allow_html=True)
        
        # Forecast data table & CSV download
        st.subheader("2. Detailed Forecast Credible Intervals Table")
        display_cols = ["date", cur_target, "y_pred_median", "ci_lower_50", "ci_upper_50", "ci_lower_80", "ci_upper_80", "ci_lower_95", "ci_upper_95", "y_pred_std"]
        avail_display = [c for c in display_cols if c in pred_df.columns]
        st.dataframe(pred_df[avail_display].round(2), use_container_width=True)
        
        # Download forecast CSV
        csv_forecast = io.StringIO()
        pred_df[avail_display].to_csv(csv_forecast, index=False)
        st.download_button(
            label="📥 Download Forecast Results (CSV)",
            data=csv_forecast.getvalue(),
            file_name=f"bayesian_forecast_{cur_target}.csv",
            mime="text/csv",
            help="Download the out-of-sample Bayesian forecast predictions and uncertainty bounds."
        )

# ---------------------------------------------------------
# PAGE 5: BAYESIAN ANALYSIS
# ---------------------------------------------------------
elif page == "Bayesian Analysis":
    st.title("🔬 Bayesian Statistical Diagnostics & Hierarchical Shrinkage")
    st.markdown(r"Examine MCMC convergence ($\hat{R}$, ESS), posterior parameter densities, and partial pooling shrinkage.")
    
    if st.session_state.bayes_model is None or st.session_state.bayes_model.idata is None:
        st.warning("⚠️ No Bayesian model has been trained yet. Please navigate to the **Sales Forecasting** page and click **Fit Bayesian Model** first.")
    else:
        model = st.session_state.bayes_model
        idata = model.idata
        
        st.subheader("1. Mathematical Formulation & Hyperprior Structure")
        st.latex(r"""
        y_{s,c,t} \sim \text{StudentT}\left(\nu, \; \mu_{s,c,t}, \; \sigma_{\text{obs}}\right)
        """)
        st.latex(r"""
        \mu_{s,c,t} = \alpha_0 + \alpha_{\text{store}[s]} + \alpha_{\text{cat}[c]} + \beta_{\text{trend}} \cdot t + \beta_{\text{promo}} \cdot \text{Promo}_t + \beta_{\text{holiday}} \cdot \text{Holiday}_t + \sum_{k=1}^2 \left[\gamma_{sk} \sin\left(\frac{2\pi kt}{P}\right) + \gamma_{ck} \cos\left(\frac{2\pi kt}{P}\right)\right]
        """)
        
        st.markdown("---")
        
        st.subheader("2. MCMC Convergence Diagnostics")
        diag = model.get_diagnostics()
        if "summary_df" in diag:
            c1, c2, c3 = st.columns(3)
            if diag.get("max_rhat") is not None and not np.isnan(diag["max_rhat"]):
                c1.metric("Max Gelman-Rubin (R̂)", f"{diag['max_rhat']:.3f}", delta="Optimal < 1.05")
            else:
                c1.metric("Max Gelman-Rubin (R̂)", "N/A", delta="Requires Multi-Chain NUTS")
                
            if diag.get("min_ess_bulk") is not None and not np.isnan(diag["min_ess_bulk"]):
                c2.metric("Min Effective Sample Size (Bulk)", f"{diag['min_ess_bulk']:.0f}", delta="Acceptable > 100")
            else:
                c2.metric("Min Effective Sample Size (Bulk)", "N/A", delta="Target > 100")
                
            if diag.get("min_ess_tail") is not None and not np.isnan(diag["min_ess_tail"]):
                c3.metric("Min Effective Sample Size (Tail)", f"{diag['min_ess_tail']:.0f}", delta="Acceptable > 100")
            else:
                c3.metric("Min Effective Sample Size (Tail)", "N/A", delta="Target > 100")
            
            st.markdown("#### Posterior Parameter Summary Statistics")
            st.dataframe(diag["summary_df"].round(4), use_container_width=True)
            
        st.markdown("---")
        
        st.subheader("3. Posterior Parameter Distributions")
        post = idata.posterior
        param_names = [v for v in ["alpha_0", "beta_trend", "beta_promo", "beta_holiday", "sigma_obs"] if v in post]
        
        sel_param = st.selectbox("Select Parameter to Inspect", param_names)
        vals = post[sel_param].values.flatten()
        
        fig_post = px.histogram(
            x=vals,
            nbins=35,
            marginal="box",
            title=f"Posterior Density of {sel_param} (Mean: {np.mean(vals):.3f}, 95% HDI: [{np.quantile(vals, 0.025):.3f}, {np.quantile(vals, 0.975):.3f}])",
            labels={"x": sel_param},
            color_discrete_sequence=["#8B5CF6"]
        )
        apply_chart_theme(fig_post, height=400)
        st.plotly_chart(fig_post, use_container_width=True)
        
        st.markdown("---")
        
        st.subheader("4. Hierarchical Analysis & Bayesian Partial Pooling")
        
        # Retrieve active multi-store dataset
        raw_df = st.session_state.raw_df
        has_stores = "store_id" in raw_df.columns and raw_df["store_id"].nunique() > 1
        has_cats = "category" in raw_df.columns and raw_df["category"].nunique() > 1
        num_stores = raw_df["store_id"].nunique() if "store_id" in raw_df.columns else 0
        num_cats = raw_df["category"].nunique() if "category" in raw_df.columns else 0
        supports_hierarchy = has_stores or has_cats

        if not supports_hierarchy:
            st.info(
                f"ℹ️ **Hierarchical Structure Information:** The current active dataset contains {num_stores} distinct store(s) "
                f"and {num_cats} distinct product category(ies). Bayesian hierarchical modeling and partial pooling require "
                f"at least 2 distinct stores or product categories to estimate group-level shrinkage parameters. "
                f"To analyze hierarchical partial pooling, please load or upload a multi-store/multi-category dataset from the **Data Overview** page."
            )
        else:
            # Check if active model has hierarchical estimates, otherwise fit or retrieve cached hierarchical model
            hier_forecaster = None
            if model is not None and getattr(model, "is_hierarchical", False) and model.get_hierarchical_shrinkage_summary() is not None:
                hier_forecaster = model
                agg_hier_data = getattr(st.session_state, "last_train_df", raw_df)
            elif "hierarchical_bayes_model" in st.session_state and st.session_state.hierarchical_bayes_model is not None:
                hier_forecaster = st.session_state.hierarchical_bayes_model
                agg_hier_data = st.session_state.get("hierarchical_agg_data", raw_df)
            else:
                cur_target = st.session_state.get("last_target_metric", "revenue")
                group_cols = [c for c in ["store_id", "category"] if c in raw_df.columns and raw_df[c].nunique() > 1]
                agg_hier_data = aggregate_sales_data(raw_df, freq="W-MON", group_by=group_cols, target_metric=cur_target)
                if "store_name" in raw_df.columns and "store_id" in group_cols:
                    store_map_df = raw_df[["store_id", "store_name"]].drop_duplicates()
                    agg_hier_data = agg_hier_data.merge(store_map_df, on="store_id", how="left")
                feat_hier = engineer_forecasting_features(agg_hier_data, fourier_harmonics=2, target_col=cur_target)
                with st.spinner("Fitting Hierarchical Bayesian Model across multi-store and category levels..."):
                    hier_forecaster = BayesianHierarchicalForecaster(
                        fourier_harmonics=2,
                        target_col=cur_target,
                        inference_method="advi",
                        draws=300,
                        random_seed=42
                    )
                    hier_forecaster.fit(feat_hier, verbose=False)
                    st.session_state.hierarchical_bayes_model = hier_forecaster
                    st.session_state.hierarchical_agg_data = agg_hier_data
            
            # Extract shrinkage parameters
            shrink_df = hier_forecaster.get_hierarchical_shrinkage_summary() if hier_forecaster is not None else None
            
            if shrink_df is not None and not shrink_df.empty:
                # Map store ID to descriptive store name
                store_name_map = {}
                if "store_id" in raw_df.columns and "store_name" in raw_df.columns:
                    store_name_map = dict(zip(raw_df["store_id"].astype(str), raw_df["store_name"].astype(str)))
                
                display_entities = []
                for _, r in shrink_df.iterrows():
                    if r["Level"] == "Store":
                        s_id = str(r["Entity"])
                        s_name = store_name_map.get(s_id, s_id)
                        display_entities.append(f"{s_name} ({s_id})" if s_name != s_id else s_id)
                    else:
                        display_entities.append(str(r["Entity"]))
                shrink_df["Display_Entity"] = display_entities
                
                # Compute empirical sample mean for no-pooling comparison
                cur_target = getattr(hier_forecaster, "target_col", "revenue")
                g_mean = float(agg_hier_data[cur_target].mean()) if cur_target in agg_hier_data.columns else 0.0
                g_std = float(agg_hier_data[cur_target].std()) if cur_target in agg_hier_data.columns and agg_hier_data[cur_target].std() > 1e-6 else 1.0
                
                emp_means = []
                emp_offsets = []
                for _, r in shrink_df.iterrows():
                    lvl = r["Level"]
                    ent = r["Entity"]
                    if lvl == "Store" and "store_id" in agg_hier_data.columns:
                        sub = agg_hier_data[agg_hier_data["store_id"] == ent]
                        val = float(sub[cur_target].mean()) if not sub.empty else g_mean
                    elif lvl == "Category" and "category" in agg_hier_data.columns:
                        sub = agg_hier_data[agg_hier_data["category"] == ent]
                        val = float(sub[cur_target].mean()) if not sub.empty else g_mean
                    else:
                        val = g_mean
                    emp_means.append(val)
                    emp_offsets.append((val - g_mean) / g_std)
                    
                shrink_df["Empirical_Mean"] = emp_means
                shrink_df["No_Pooling_Offset"] = emp_offsets
                
                # Selection option for Enterprise Total, individual stores, and product categories
                scope_choices = ["Enterprise Total (All Stores & Categories)"]
                if has_stores:
                    scope_choices.append("Individual Stores")
                if has_cats:
                    scope_choices.append("Product Categories")
                    
                sel_scope = st.radio(
                    "Select Hierarchical Scope View:",
                    scope_choices,
                    horizontal=True
                )
                
                filtered_shrink = shrink_df.copy()
                if sel_scope == "Individual Stores":
                    store_filter_options = ["All Stores"] + sorted(shrink_df[shrink_df["Level"] == "Store"]["Display_Entity"].unique().tolist())
                    chosen_store = st.selectbox("Filter Store Entity:", store_filter_options)
                    if chosen_store != "All Stores":
                        filtered_shrink = filtered_shrink[filtered_shrink["Display_Entity"] == chosen_store]
                    else:
                        filtered_shrink = filtered_shrink[filtered_shrink["Level"] == "Store"]
                elif sel_scope == "Product Categories":
                    cat_filter_options = ["All Categories"] + sorted(shrink_df[shrink_df["Level"] == "Category"]["Display_Entity"].unique().tolist())
                    chosen_cat = st.selectbox("Filter Category Entity:", cat_filter_options)
                    if chosen_cat != "All Categories":
                        filtered_shrink = filtered_shrink[filtered_shrink["Display_Entity"] == chosen_cat]
                    else:
                        filtered_shrink = filtered_shrink[filtered_shrink["Level"] == "Category"]
                
                # Meaningful Partial Pooling Visualization
                fig_partial = go.Figure()
                
                # 1. No-Pooling Empirical Estimates (Raw sample offsets)
                fig_partial.add_trace(go.Scatter(
                    x=filtered_shrink["No_Pooling_Offset"],
                    y=filtered_shrink["Display_Entity"],
                    mode="markers",
                    marker=dict(size=9, color="#94A3B8", symbol="circle-open", line=dict(width=2, color="#94A3B8")),
                    name="No Pooling (Empirical Sample Offset)"
                ))
                
                # 2. Partial Pooling (Bayesian Hierarchical Estimates with Credible Uncertainty)
                store_mask = filtered_shrink["Level"] == "Store"
                cat_mask = filtered_shrink["Level"] == "Category"
                
                if store_mask.any():
                    fig_partial.add_trace(go.Scatter(
                        x=filtered_shrink.loc[store_mask, "Shrinkage_Offset"],
                        y=filtered_shrink.loc[store_mask, "Display_Entity"],
                        error_x=dict(type="data", array=filtered_shrink.loc[store_mask, "Uncertainty_Std"], color="#60A5FA", width=2.5),
                        mode="markers",
                        marker=dict(size=11, color="#3B82F6"),
                        name="Partial Pooling: Store Hierarchy (±1σ)"
                    ))
                    
                if cat_mask.any():
                    fig_partial.add_trace(go.Scatter(
                        x=filtered_shrink.loc[cat_mask, "Shrinkage_Offset"],
                        y=filtered_shrink.loc[cat_mask, "Display_Entity"],
                        error_x=dict(type="data", array=filtered_shrink.loc[cat_mask, "Uncertainty_Std"], color="#C084FC", width=2.5),
                        mode="markers",
                        marker=dict(size=11, color="#8B5CF6"),
                        name="Partial Pooling: Category Hierarchy (±1σ)"
                    ))
                
                # Complete Pooling Reference Line (Global Baseline)
                fig_partial.add_vline(
                    x=0,
                    line_dash="dash",
                    line_color="#10B981",
                    annotation_text="Complete Pooling (Enterprise Baseline)",
                    annotation_position="top left",
                    annotation_font_color="#10B981"
                )
                
                fig_partial.update_layout(
                    xaxis_title="Posterior Effect Offset (Standardized Units from Enterprise Mean)",
                    yaxis_title="Hierarchical Entity",
                    title="<b>Bayesian Partial Pooling Shrinkage: Empirical vs Hierarchical Estimates</b>"
                )
                apply_chart_theme(fig_partial, height=440)
                st.plotly_chart(fig_partial, use_container_width=True)
                
                # Summary Table with actual data metrics
                st.markdown("##### Hierarchical Parameter Estimates & Shrinkage Table")
                summary_view = pd.DataFrame({
                    "Level": filtered_shrink["Level"],
                    "Entity": filtered_shrink["Display_Entity"],
                    f"Empirical Weekly Mean ({cur_target.capitalize()})": [f"${v:,.2f}" if cur_target == "revenue" else f"{v:,.1f} units" for v in filtered_shrink["Empirical_Mean"]],
                    "Raw Sample Offset (No Pooling)": filtered_shrink["No_Pooling_Offset"].round(3),
                    "Bayesian Posterior Offset (Partial Pooling)": filtered_shrink["Shrinkage_Offset"].round(3),
                    "Posterior Uncertainty (±1σ)": filtered_shrink["Uncertainty_Std"].round(3)
                })
                st.dataframe(summary_view, use_container_width=True)
                
                st.markdown("""
                <div class="math-box">
                    <b>Understanding Bayesian Partial Pooling & Shrinkage:</b><br>
                    • <b>No Pooling (Open Circles):</b> Individual group sample means fit each store in complete isolation, leaving smaller/volatile stores vulnerable to noise.<br>
                    • <b>Complete Pooling (Green Dashed Line at 0):</b> Forces all stores and categories into a single identical intercept, eliminating local differentiation.<br>
                    • <b>Partial Pooling (Solid Dots & Credible Bands):</b> Estimates adaptive group hyperpriors $\\sigma_{\\text{store}}$ and $\\sigma_{\\text{cat}}$. Group estimates shrink toward the Enterprise baseline in proportion to their uncertainty, providing optimal statistical regularization without overfitting.
                </div>
                """, unsafe_allow_html=True)

# ---------------------------------------------------------
# PAGE 6: SEASONALITY ANALYSIS
# ---------------------------------------------------------
elif page == "Seasonality Analysis":
    st.title("📅 Multi-Scale Seasonality & Harmonic Decomposition")
    st.markdown("Analyze annual, quarterly, and monthly seasonal profiles alongside Fourier power spectra.")
    
    df = st.session_state.raw_df
    weekly_total = aggregate_sales_data(df, freq="W-MON", group_by=None)
    
    st.subheader("1. Classical 4-Component Time Series Decomposition")
    decomp_dict = perform_seasonal_decomposition(weekly_total, target_col="revenue", period=52, model="additive")
    fig_decomp = plot_seasonality_decomposition(decomp_dict)
    st.plotly_chart(fig_decomp, use_container_width=True)
    
    m_col1, m_col2 = st.columns(2)
    m_col1.metric("Seasonal Component Strength", f"{decomp_dict['seasonal_strength']*100:.1f}%")
    m_col2.metric("Macro Trend Strength", f"{decomp_dict['trend_strength']*100:.1f}%")
    
    st.markdown("---")
    
    colA, colB = st.columns(2)
    with colA:
        st.subheader("2. Monthly Seasonality Index")
        monthly_df = compute_monthly_seasonality(weekly_total, target_col="revenue")
        fig_monthly = plot_monthly_seasonality(monthly_df)
        st.plotly_chart(fig_monthly, use_container_width=True)
        
    with colB:
        st.subheader("3. Quarterly Performance Distribution")
        quarterly_df = compute_quarterly_seasonality(weekly_total, target_col="revenue")
        fig_q = px.bar(
            quarterly_df,
            x="quarter",
            y="revenue_share_pct",
            text=[f"{v:.1f}%" for v in quarterly_df["revenue_share_pct"]],
            title="Quarterly Revenue Share (%)",
            color="quarter",
            color_discrete_sequence=px.colors.sequential.Teal
        )
        apply_chart_theme(fig_q, height=420)
        st.plotly_chart(fig_q, use_container_width=True)

    st.markdown("---")
    
    st.subheader("4. Fourier Harmonic Power Spectral Density (FFT)")
    st.markdown("Identifies dominant cyclical frequencies in retail demand.")
    spectrum_df = compute_fourier_spectrum(weekly_total, target_col="revenue")
    if not spectrum_df.empty:
        fig_fft = px.line(
            spectrum_df,
            x="period_weeks",
            y="power",
            markers=True,
            title="Fast Fourier Transform Power Spectrum (Peak Periodicity)",
            labels={"period_weeks": "Cycle Period (Weeks)", "power": "Spectral Energy"},
            color_discrete_sequence=["#10B981"]
        )
        apply_chart_theme(fig_fft, height=380)
        st.plotly_chart(fig_fft, use_container_width=True)

    st.markdown("---")
    
    st.subheader("5. Holiday & Promotional Demand Lift")
    lift_stats = compute_holiday_promotional_lift(weekly_total, target_col="revenue")
    l1, l2 = st.columns(2)
    if "holiday_lift_pct" in lift_stats:
        l1.metric("Holiday Week Volume Surge", f"+{lift_stats['holiday_lift_pct']:.1f}%", help="Average volume increase during major holiday weeks.")
    if "promo_lift_pct" in lift_stats:
        l2.metric("Promotional Event Volume Surge", f"+{lift_stats['promo_lift_pct']:.1f}%", help="Average lift during active promotional campaigns.")

# ---------------------------------------------------------
# PAGE 7: INVENTORY OPTIMIZATION
# ---------------------------------------------------------
elif page == "Inventory Optimization":
    st.title("📦 Retail Demand Planning & Inventory Optimization")
    st.markdown("Transform Bayesian uncertainty into optimal Safety Stock, Reorder Points, and simulated inventory policies.")
    
    # Check if predictive samples are available
    if st.session_state.bayes_model is None or st.session_state.bayes_model.get_predictive_samples() is None:
        st.info("ℹ️ Using default benchmark posterior draws. Fit the model in 'Sales Forecasting' to simulate live MCMC draws.")
        np.random.seed(42)
        sim_draws = np.random.gamma(shape=15, scale=12, size=(500, 12))
    else:
        sim_draws = st.session_state.bayes_model.get_predictive_samples()

    # Input Parameters
    st.sidebar.markdown("### ⚙️ Inventory Control Panel")
    lead_time_w = st.sidebar.slider("Supplier Lead Time (Weeks)", min_value=1, max_value=6, value=2)
    target_sl = st.sidebar.slider("Target Cycle Service Level (%)", min_value=80.0, max_value=99.5, value=95.0, step=0.5) / 100.0
    unit_cost = st.sidebar.number_input("Unit Purchase Cost ($)", min_value=1.0, max_value=500.0, value=25.0, step=5.0)
    holding_pct = st.sidebar.slider("Annual Holding Cost (% of Unit Cost)", min_value=5.0, max_value=50.0, value=20.0) / 100.0
    order_cost = st.sidebar.number_input("Fixed Order Setup Cost ($)", min_value=5.0, max_value=500.0, value=50.0, step=5.0)
    stockout_penalty = st.sidebar.number_input("Unit Stockout Penalty ($)", min_value=5.0, max_value=200.0, value=20.0, step=5.0)

    # Compute parameters
    mean_w_dem = float(np.mean(sim_draws[:, 0]))
    std_w_dem = float(np.std(sim_draws[:, 0]))
    
    classical_res = calculate_classical_inventory_parameters(
        mean_weekly_demand=mean_w_dem,
        std_weekly_demand=std_w_dem,
        lead_time_weeks=lead_time_w,
        service_level=target_sl,
        unit_cost=unit_cost,
        holding_cost_annual_pct=holding_pct,
        order_cost=order_cost,
        stockout_penalty_per_unit=stockout_penalty
    )
    
    bayesian_res = calculate_bayesian_inventory_parameters(
        predictive_samples=sim_draws,
        lead_time_weeks=lead_time_w,
        service_level=target_sl,
        unit_cost=unit_cost,
        holding_cost_annual_pct=holding_pct,
        order_cost=order_cost,
        stockout_penalty_per_unit=stockout_penalty
    )

    st.subheader("1. Classical Gaussian vs Bayesian Uncertainty-Aware Policy Comparison")
    
    k1, k2, k3, k4 = st.columns(4)
    with k1:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-title">Safety Stock (SS)</div>
            <div class="kpi-value" style="color:#3B82F6;">{bayesian_res['safety_stock']} units</div>
            <div class="kpi-subtext">Classical: {classical_res['safety_stock']} units</div>
        </div>
        """, unsafe_allow_html=True)
    with k2:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-title">Reorder Point (ROP)</div>
            <div class="kpi-value" style="color:#10B981;">{bayesian_res['reorder_point']} units</div>
            <div class="kpi-subtext">Classical: {classical_res['reorder_point']} units</div>
        </div>
        """, unsafe_allow_html=True)
    with k3:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-title">Economic Order Qty (EOQ)</div>
            <div class="kpi-value" style="color:#8B5CF6;">{bayesian_res['eoq']} units</div>
            <div class="kpi-subtext">Optimal Order Batch</div>
        </div>
        """, unsafe_allow_html=True)
    with k4:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-title">Estimated Fill Rate (Type-2)</div>
            <div class="kpi-value" style="color:#F59E0B;">{bayesian_res['fill_rate']}%</div>
            <div class="kpi-subtext">Target CSL: {target_sl*100:.1f}%</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    
    col_hist, col_curve = st.columns(2)
    with col_hist:
        st.subheader("2. Lead-Time Demand Uncertainty & ROP Cutoff")
        fig_dist = plot_lead_time_demand_distribution(
            lead_time_draws=bayesian_res["lead_time_distribution"],
            reorder_point=bayesian_res["reorder_point"],
            service_level=target_sl
        )
        st.plotly_chart(fig_dist, use_container_width=True)
        
    with col_curve:
        st.subheader("3. Service Level vs Total Cost Trade-off Curve")
        tradeoff_df = generate_service_level_tradeoff_curve(
            predictive_samples=sim_draws,
            lead_time_weeks=lead_time_w,
            unit_cost=unit_cost,
            holding_cost_annual_pct=holding_pct,
            order_cost=order_cost,
            stockout_penalty_per_unit=stockout_penalty
        )
        fig_cost = plot_service_level_cost_curve(tradeoff_df)
        st.plotly_chart(fig_cost, use_container_width=True)

    st.markdown("---")
    
    st.subheader("4. Dynamic Inventory Trajectory Simulator (Continuous Review s, Q Policy)")
    sim_horizon = 26
    # Create representative demand sequence
    test_demand = np.mean(sim_draws[:, :min(sim_horizon, sim_draws.shape[1])], axis=0)
    if len(test_demand) < sim_horizon:
        test_demand = np.tile(test_demand, int(np.ceil(sim_horizon / len(test_demand))))[:sim_horizon]
        
    sim_df = simulate_inventory_trajectory(
        demand_series=test_demand,
        reorder_point=bayesian_res["reorder_point"],
        order_quantity=bayesian_res["eoq"],
        lead_time_steps=lead_time_w,
        unit_holding_cost=unit_cost * holding_pct / 52.0,
        unit_stockout_penalty=stockout_penalty
    )
    
    fig_traj = plot_inventory_trajectory(sim_df, reorder_point=bayesian_res["reorder_point"])
    st.plotly_chart(fig_traj, use_container_width=True)

# ---------------------------------------------------------
# PAGE 8: MODEL EVALUATION
# ---------------------------------------------------------
elif page == "Model Evaluation":
    st.title("⚖️ Comprehensive Model Evaluation & Benchmarking")
    st.markdown("Compare the Bayesian Hierarchical model against Ridge, Random Forest, and Holt-Winters baselines.")
    
    df = st.session_state.raw_df
    weekly_total = aggregate_sales_data(df, freq="W-MON", group_by=None, target_metric="revenue")
    feat_df = engineer_forecasting_features(weekly_total, fourier_harmonics=2, target_col="revenue")
    train_df, test_df = split_train_test(feat_df, test_periods=12)
    
    eval_btn = st.button("🏁 Run Comprehensive Multi-Model Benchmark", type="primary")
    
    if eval_btn or st.session_state.baseline_results:
        if eval_btn:
            with st.spinner("Training and benchmarking all candidate forecasting models..."):
                results_dict = {}
                
                # 1. Ridge Baseline
                ridge = RidgeBaselineForecaster(target_col="revenue").fit(train_df)
                results_dict["Ridge Regression"] = ridge.predict(test_df)
                
                # 2. Random Forest Baseline
                rf = RandomForestBaselineForecaster(target_col="revenue", n_estimators=60).fit(train_df)
                results_dict["Random Forest"] = rf.predict(test_df)
                
                # 3. Holt-Winters Baseline
                hw = HoltWintersBaselineForecaster(target_col="revenue", seasonal_periods=52).fit(train_df)
                results_dict["Holt-Winters"] = hw.predict(test_df)
                
                # 4. Bayesian Model
                if (st.session_state.bayes_model is not None and 
                    st.session_state.forecast_results is not None and 
                    len(st.session_state.forecast_results) == len(test_df) and
                    st.session_state.get("last_target_metric") == "revenue"):
                    results_dict["Bayesian Hierarchical"] = st.session_state.forecast_results
                else:
                    bayes = BayesianHierarchicalForecaster(fourier_harmonics=2, target_col="revenue", inference_method="advi", draws=300)
                    bayes.fit(train_df, verbose=False)
                    bayes_pred = bayes.predict(test_df)
                    results_dict["Bayesian Hierarchical"] = bayes_pred
                    st.session_state.bayes_model = bayes
                    st.session_state.forecast_results = bayes_pred
                    st.session_state.last_train_df = train_df
                    st.session_state.last_target_metric = "revenue"
                    
                st.session_state.baseline_results = results_dict
                
        results_dict = st.session_state.baseline_results
        
        # Benchmark Table
        st.subheader("1. Benchmark Performance Comparison Matrix")
        summary_table = compare_multiple_models(results_dict, actual_col="revenue", pred_col="y_pred_mean")
        st.dataframe(summary_table, use_container_width=True)
        
        st.markdown("""
        <div class="math-box">
            <b>Metric Explanations & Zero-Handling:</b><br>
            • <b>MAE (Mean Absolute Error):</b> Average absolute monetary forecast deviation.<br>
            • <b>RMSE:</b> Penalizes large errors heavily, critical for supply chains with stockout risks.<br>
            • <b>wMAPE (Weighted MAPE):</b> Standard metric in retail demand planning ($\\sum |e| / \\sum y$). Completely immune to division-by-zero on slow-moving zero-sales items.<br>
            • <b>Coverage_95 (%):</b> Empirical percentage of actual observations landing inside the model's 95% Credible Interval. A well-calibrated Bayesian model approaches 95%.
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        st.subheader("2. Visual Comparison of Candidate Forecasts on Holdout Test Set")
        fig_comp = plot_model_comparison(results_dict, actual_col="revenue")
        st.plotly_chart(fig_comp, use_container_width=True)

# ---------------------------------------------------------
# PAGE 9: ABOUT PROJECT
# ---------------------------------------------------------
elif page == "About Project":
    st.title("🎓 Final-Year B.Tech Major Project Documentation")
    
    st.markdown("""
    <div class="academic-box">
        <h3 style="color:#F8FAFC; margin-top:0;">Project Title</h3>
        <p style="font-size:16px; color:#E2E8F0; font-weight:600;">
            Sales and Revenue Forecasting with Seasonality via Bayesian Hierarchical Modeling
            for Uncertainty Quantification, Applied to Retail Demand Planning and Inventory Optimization
        </p>
        <hr style="border-color:#475569;">
        <p style="margin:0; font-size:14px; color:#CBD5E1;">
            <b>Project Guide:</b> Dr. P. Sudhakar<br>
            <b>Department:</b> Computer Science & Engineering / Data Science<br>
            <b>Degree:</b> Bachelor of Technology (B.Tech Major Project)
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("### 📚 Mathematical & Algorithmic Foundations")
    st.markdown("""
    #### 1. Why Hierarchical Modeling?
    Retail store sales display a natural two-level nested hierarchy:
    - **Global Level:** Macroeconomic retail trends, holiday shifts, and broad annual seasonality.
    - **Store Level:** Footfall differences, store format variations (e.g. Metro vs Suburban).
    - **Category Level:** Perishability, demand velocity, and price elasticities.
    
    Instead of fitting individual models per store (which suffer from high variance) or pooling all stores into a single model (which ignores local traits), **Bayesian partial pooling** places hyperpriors over store and category random effects:
    $$ \\alpha_{\\text{store}[s]} \\sim \\mathcal{N}(0, \\sigma_{\\text{store}}), \\quad \\sigma_{\\text{store}} \\sim \\text{HalfNormal}(\\tau) $$
    
    #### 2. Robust Student-t Likelihood
    Traditional Gaussian models perform poorly in retail due to heavy-tailed holiday demand bursts. We implement a robust Student-t likelihood:
    $$ y_i \\sim \\text{StudentT}(\\nu, \\mu_i, \\sigma_{\\text{obs}}) $$
    where the degrees-of-freedom $\\nu$ is inferred from the data.
    
    #### 3. Lead-Time Uncertainty & Empirical Safety Stock
    Under traditional inventory control, safety stock is derived assuming normally distributed lead-time demand:
    $$ SS_{\\text{classical}} = Z_{\\alpha} \\cdot \\sigma_{D_L} $$
    Our Bayesian framework instead sums future draws across the lead-time window:
    $$ D_L^{(s)} = \\sum_{t=1}^L y_{t}^{(s)}, \\quad s = 1, \\dots, S $$
    The exact Reorder Point is the empirical quantile:
    $$ ROP_{\\text{Bayesian}} = Q_{\\alpha}\\left(\\{D_L^{(s)}\\}\\right) $$
    """)
    
    st.markdown("---")
    st.markdown("### ⚠️ Known Limitations & Future Scope")
    st.markdown("""
    - **Multi-Echelon Optimization:** Currently assumes single-tier store-level replenishment. Future work could extend to central distribution center (CDC) to regional distribution center (RDC) multi-echelon networks.
    - **Exogenous Signals:** Incorporating local weather forecasts, foot-traffic sensors, and competitive pricing feeds.
    - **Dynamic Lead Times:** Supplying stochastic supplier lead-time distributions alongside demand draws.
    """)
    
    st.markdown("---")
    st.markdown("### 📖 References")
    st.markdown("""
    1. Gelman, A., et al. (2013). *Bayesian Data Analysis* (3rd ed.). CRC Press.
    2. Silver, E. A., Pyke, D. F., & Thomas, D. J. (2016). *Inventory and Production Management in Supply Chains*. CRC Press.
    3. Salvatier, J., Wiecki, T. V., & Fonnesbeck, C. (2016). Probabilistic programming in Python using PyMC. *PeerJ Computer Science*, 2, e55.
    4. Hyndman, R. J., & Athanasopoulos, G. (2018). *Forecasting: Principles and Practice* (2nd ed.). OTexts.
    """)
