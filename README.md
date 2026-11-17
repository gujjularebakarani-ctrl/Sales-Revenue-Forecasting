# Sales & Revenue Forecasting via Bayesian Hierarchical Modeling
### Uncertainty Quantification for Retail Demand Planning and Inventory Optimization

**Final-Year B.Tech Major Project**  
**Project Guide:** Dr. P. Sudhakar  
**Technology Stack:** Python 3.12, PyMC, ArviZ, Scikit-learn, Statsmodels, Pandas, NumPy, Plotly, Streamlit  

---

## 🌟 Executive Summary

Retail demand forecasting is characterized by non-linear annual seasonality, store/category hierarchy, promotional shocks, and extreme demand variability. Standard point-forecasting algorithms (e.g. ARIMA, Point Regression) fail to quantify prediction uncertainty, leading either to excessive carrying costs or catastrophic stockout penalties.

This project implements an end-to-end **Bayesian Hierarchical Forecasting Framework** using **PyMC** and **ArviZ**:
1. **Hierarchical Partial Pooling (Shrinkage):** Borrows statistical strength across multi-store retail footprints and product categories.
2. **Harmonic Seasonality:** Decomposes annual and semi-annual cycles using Fourier sine/cosine harmonics and holiday/promotional intervention flags.
3. **Robust Student-t Likelihood:** Heavy-tailed likelihood to resist distortion from promotional outliers.
4. **Uncertainty Quantification:** Generates full posterior predictive distributions with 50% (IQR), 80%, and 95% Credible Intervals.
5. **Bayesian Inventory Optimization:** Directly derives Safety Stock and Reorder Points (ROP) from empirical quantiles of MCMC lead-time demand draws, avoiding naive Gaussian assumptions.
6. **Multi-Model Benchmarking:** Evaluates against Ridge Regression, Random Forest, and Holt-Winters baselines using zero-resilient metrics (MAE, RMSE, wMAPE, Coverage).
7. **Production Streamlit Dashboard:** 9 interactive pages with responsive Plotly visualizations, CSV upload/download, and trajectory simulators.

---

## 📂 Project Directory Structure

```
Sales_Revenue_Forecasting/
├── .streamlit/
│   └── config.toml             # Streamlit dark modern UI theme & server settings
├── data/
│   ├── generate_dataset.py     # Realistic synthetic benchmark retail dataset generator
│   └── retail_sales_sample.csv # 7,488 records (3 years, 4 stores, 4 categories)
├── docs/
│   ├── ARCHITECTURE.md         # Detailed modular system architecture & data flow
│   ├── MATHEMATICAL_FORMULATION.md # Full probabilistic formulations & proofs
│   └── PROJECT_REPORT.md       # Complete academic B.Tech major project report
├── src/
│   ├── __init__.py
│   ├── data_processing.py      # Schema validation, missing values, Fourier feature pipeline
│   ├── bayesian_model.py       # PyMC hierarchical model, NUTS MCMC & ADVI inference
│   ├── baseline_models.py      # RidgeCV, Random Forest, Holt-Winters Exponential Smoothing
│   ├── seasonality.py          # Classical 4-part decomposition, monthly index, FFT spectrum
│   ├── inventory.py            # Bayesian quantile ROP, Safety Stock, EOQ, Trajectory simulator
│   ├── evaluation.py           # MAE, RMSE, wMAPE, MAPE (zero-handled), 80/95% CI coverage
│   └── visualization.py        # Dark modern Plotly charts with credible ribbons
├── tests/
│   ├── __init__.py
│   ├── test_data_processing.py # Unit tests for data pipeline & features
│   ├── test_models.py          # Unit tests for Bayesian & baseline forecasters
│   ├── test_inventory.py       # Unit tests for inventory logic & trajectory simulation
│   └── test_evaluation.py      # Unit tests for evaluation metrics & zero-handling
├── app.py                      # Main 9-page Streamlit Application
├── run_tests.py                # Automated test runner script
├── requirements.txt            # Production dependencies
├── .gitignore
└── README.md                   # Project documentation (this file)
```

---

## 🚀 Installation & Setup Instructions

### 1. Prerequisites
- Python 3.10, 3.11, or 3.12 installed on your system.
- Git (optional, for cloning).

### 2. Clone or Extract the Project
```bash
cd Sales_Revenue_Forecasting
```

### 3. Create and Activate a Virtual Environment
**On Windows (PowerShell):**
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

**On macOS / Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

### 4. Install Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

---

## 💻 Running the Application

Launch the Streamlit web dashboard with:
```bash
streamlit run app.py
```
The application will automatically start and open in your default web browser at `http://localhost:8501`.

---

## 🧪 Running the Automated Verification Test Suite

To run all unit and integration tests across data processing, models, inventory planning, and metric evaluations:
```bash
python run_tests.py
```
Or directly with pytest:
```bash
pytest -v tests/
```

All 16 test cases verify schema validation, Fourier feature engineering, baseline training, PyMC posterior sampling, credible interval monotonicity, inventory simulation, and metric zero-handling.

---

## 📊 Streamlit Dashboard Pages

1. **Home:** Project abstract, guide credit (Dr. P. Sudhakar), KPI summary cards, architecture diagram, and quick-start guide.
2. **Data Overview:** Dataset inspection, schema validation, summary statistics, custom CSV uploader, and sample dataset download.
3. **Exploratory Analysis:** Enterprise revenue trends, store comparisons, category market share, price elasticity scatter, and correlation heatmaps.
4. **Sales Forecasting:** Interactive Bayesian forecasting with 50%, 80%, and 95% uncertainty ribbons, future scenario simulation, and CSV forecast export.
5. **Bayesian Analysis:** MCMC convergence diagnostics ($\hat{R}$, ESS), posterior density plots, and hierarchical shrinkage visualizations.
6. **Seasonality Analysis:** Classical 4-component decomposition (Trend, Seasonality, Residual), monthly seasonality index, quarterly distributions, and FFT power spectrum.
7. **Inventory Optimization:** Classical vs Bayesian Safety Stock and ROP comparison, lead-time demand histogram, service level cost tradeoff curve, and dynamic inventory trajectory simulator.
8. **Model Evaluation:** Multi-model benchmark matrix (Bayesian vs Ridge vs Random Forest vs Holt-Winters), MAE, RMSE, wMAPE, 95% CI Coverage, and holdout comparison plots.
9. **About Project:** Academic B.Tech major project documentation, mathematical formulations, limitations, and literature references.

---

## ☁️ Streamlit Community Cloud Deployment Guide

This repository is pre-configured for one-click deployment on **Streamlit Community Cloud**:
1. Push this repository to GitHub.
2. Navigate to [share.streamlit.io](https://share.streamlit.io) and log in with GitHub.
3. Click **New app**, select your repository, branch (`main`), and set the main file path to:
   ```
   app.py
   ```
4. Click **Deploy!** The cloud platform reads `requirements.txt` and `.streamlit/config.toml` automatically.

---

## 🎓 Academic Credit & Guidance

- **Student:** B.Tech Major Project Final-Year Candidate
- **Project Guide:** **Dr. P. Sudhakar**
- **Institution:** Department of Computer Science & Engineering / Data Science

---

## 📄 License & Integrity Statement

This project is created for academic final-year major project presentation and research. The synthetic demonstration dataset is clearly documented as synthetic benchmark data designed to test multi-level retail hierarchies without misrepresenting proprietary commercial records.
