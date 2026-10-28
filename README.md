# Energy Consumption Forecasting Using Time-Series Machine Learning Models

A complete, beginner-friendly Python data science and machine learning project that analyzes historical electricity usage and forecasts household power consumption for the next 24 hours.

---

## 📌 Project Overview

Electricity consumption forecasting is a foundational task in modern energy management, smart grid operations, and utility demand-response planning. Accurate forecasts help prevent grid overloads, optimize renewable energy dispatch, and reduce peak electricity costs.

This project implements an end-to-end forecasting pipeline that:
1. Loads and preprocesses over 2 million records from the **UCI Individual Household Electric Power Consumption dataset**.
2. Cleans missing values and resamples data into continuous hourly observations.
3. Performs Exploratory Data Analysis (EDA) on consumption patterns and diurnal daily cycles.
4. Engineers time-series features (lags, rolling statistics, calendar indicators, and cyclical sine/cosine encodings).
5. Trains and benchmarks four distinct forecasting paradigms using **chronological train-test splitting**:
   - **ARIMA** (Statistical Time-Series Model)
   - **SARIMA** (Seasonal Time-Series Model with 24-hour periodicity)
   - **Linear Regression** (Machine Learning with lag and calendar features)
   - **Random Forest Regressor** (Ensemble ML capturing non-linear consumption dynamics)
6. Evaluates all models against the ground truth using **MAE (Mean Absolute Error)** and **RMSE (Root Mean Squared Error)**.
7. Produces and exports 24-hour ahead forecasts to a CSV file.
8. Provides a rich, interactive **Streamlit dashboard** for visual exploration, model comparison, and forecast inspection.

---

## 🛠️ Technologies Used

- **Python 3.10+ / 3.14**
- **Pandas**: Data manipulation, time-series resampling, and interpolation.
- **NumPy**: Numerical operations, array transformations, and trigonometric cyclic encoding.
- **Matplotlib**: Publication-grade static visualizations and comparison plots.
- **Scikit-learn**: Linear Regression, Random Forest Regressor, and evaluation metrics.
- **Statsmodels**: Statistical time-series models (ARIMA & SARIMAX).
- **Streamlit**: Web application dashboard for real-time visualization and interaction.
- **Joblib**: Model serialization and persistence.

---

## 📂 Project Structure

```
energy/
├── data/
│   ├── raw/                              # Raw UCI dataset (household_power_consumption.txt / zip)
│   └── processed/
│       └── hourly_energy_consumption.csv # Cleaned hourly resampled dataset
├── models/                               # Serialized trained model files
│   ├── arima_model.pkl
│   ├── sarima_model.pkl
│   ├── linear_regression.joblib
│   └── random_forest.joblib
├── outputs/                              # Generated CSVs and visualization plots
│   ├── forecasts_24h.csv                 # Next 24-hour forecasts from all 4 models + Actual
│   ├── model_comparison.csv              # MAE, RMSE, and ranking summary table
│   ├── historical_consumption.png        # Historical consumption time series
│   ├── hourly_consumption_profile.png    # 24-hour diurnal consumption curve (mean ± std)
│   ├── actual_vs_predicted_24h.png       # 24-hour ground truth vs model forecasts
│   └── model_comparison.png              # Side-by-side MAE and RMSE bar chart
├── src/
│   ├── __init__.py
│   ├── data_loader.py                    # Dataset download, cleaning, resampling & caching
│   ├── features.py                       # Lag creation, rolling stats, cyclical time encoding
│   ├── models.py                         # ARIMA, SARIMA, Linear Regression & Random Forest classes
│   └── utils.py                          # Metric calculations, plotting helpers, CSV exporters
├── main.py                               # Complete pipeline execution script
├── app.py                                # Interactive Streamlit dashboard
├── requirements.txt                      # Project package dependencies
└── README.md                             # Documentation and user guide
```

---

## 📊 Dataset Details

- **Name**: Individual Household Electric Power Consumption Data Set
- **Source**: [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/235/individual+household+electric+power+consumption)
- **Timeframe**: December 2006 – November 2010 (~47 months, over 2,075,000 observations).
- **Target Variable**: `Global_active_power` (household global minute-averaged active power in kilowatts).
- **Preprocessing**:
  1. Parse semicolon-delimited text and convert `?` markers to `NaN`.
  2. Parse date and time into continuous timestamp index (`Datetime`).
  3. Resample minute records to hourly averages (`1h`), matching hourly utility scheduling horizons.
  4. Fill remaining gaps with time-based linear interpolation.

---

## 🧠 Forecasting Models Explained

### 1. ARIMA (AutoRegressive Integrated Moving Average)
- **Concept**: A classical statistical model combining autoregression (AR), differencing for stationarity (I), and moving average error correction (MA).
- **Order**: `(p=1, d=1, q=1)`
- **Role**: Serves as a fundamental statistical baseline.

### 2. SARIMA (Seasonal ARIMA)
- **Concept**: Extends ARIMA by explicitly modeling recurring seasonal patterns.
- **Seasonality**: `s = 24` hours (daily human circadian cycle).
- **Order**: `(1, 0, 1) x (1, 1, 0, 24)`
- **Role**: Efficiently fits recent diurnal cycles to project daily recurring load peaks and valleys.

### 3. Linear Regression (Lag & Calendar Features)
- **Concept**: Supervised machine learning treating time-series as regression.
- **Features**: Immediate lags (`t-1`, `t-2`, `t-3`), daily lag (`t-24`), 2-day lag (`t-48`), weekly lag (`t-168`), rolling means (6h and 24h), hour of day, and day of week.
- **Role**: Fast, transparent, and highly competitive for short-term persistence.

### 4. Random Forest Regressor
- **Concept**: Non-linear ensemble model combining 100 decision trees.
- **Features**: Historical lag consumption, rolling volatility, sine/cosine cyclical hour encodings, and calendar flags.
- **Role**: Captures complex non-linear consumption surges (such as high-draw appliance activations) without requiring linear assumptions.

---

## 🚀 Installation & Quickstart

### 1. Clone or Open the Workspace
```bash
cd c:\Users\LENOVO\Desktop\energy
```

### 2. Set Up a Virtual Environment (Recommended)
```bash
python -m venv venv
# On Windows PowerShell:
.\venv\Scripts\Activate.ps1
# On Linux/macOS:
source venv/bin/activate
```

### 3. Install Required Packages
```bash
pip install -r requirements.txt
```

---

## ⚡ Execution Instructions

### Run the End-to-End Pipeline
To run the automated data ingestion, preprocessing, model training, evaluation, and artifact export:
```bash
python main.py
```

**What this produces:**
- Cleans and caches the hourly dataset in `data/processed/hourly_energy_consumption.csv`.
- Trains and benchmarks all four forecasting models on chronological data splits.
- Prints the evaluation comparison table directly in the console.
- Saves the next 24-hour forecast predictions to `outputs/forecasts_24h.csv`.
- Saves performance metrics to `outputs/model_comparison.csv`.
- Serializes trained model weights into the `models/` directory.
- Exports visualization charts to `outputs/`.

---

### Launch the SaaS Streamlit Dashboard
To launch the redesigned purple-and-lavender SaaS analytics dashboard:
```bash
streamlit run app.py
```
*Or using Anaconda Python on Windows:*
```bash
C:\Users\LENOVO\anaconda3\python.exe -m streamlit run app.py
```

The application will open at `http://localhost:8501`, featuring:
- **Left Sidebar**: Fixed dark charcoal (`#202124`) navigation with **ENERGY AI** branding and purple gradient active pills.
- **Top Navigation Bar**: Live dataset connection status, test horizon badges, and analyst profile.
- **Top KPI Cards**: Real-time cards for **Latest Consumption (kW)**, **Average Demand (kW)**, **Next 24h Total Forecast (kWh)**, and **Best Forecast Model (Rank 1)**.
- **Main Analytics Section ("Electricity Consumption")**: Interactive time-series line chart with time filters (`24 Hours`, `7 Days`, `30 Days`, `All Data`), purple historical series, and dashed lavender forecast curve.
- **Card A — Consumption Pattern**: Diurnal 24-hour cycle highlighting Morning Peak (7–9 AM), Evening Peak (6–9 PM), and Night Valley (1–5 AM).
- **Card B — Model Accuracy**: Actual vs. Predicted curves for the 24-hour test period with instant MAE/RMSE badges.
- **Consumption Forecast Panel**: Dedicated model selector, summary metrics, hourly forecast comparison table, and one-click CSV export.
- **Model Comparison Panel**: 4-model benchmark with horizontal purple/lavender comparison charts and architectural notes.
- **Historical Data Explorer**: Searchable telemetry records and summary statistics.
- **Settings**: Horizon controls, re-training triggers, and environment diagnostics.

---

## 📈 Evaluation Metrics

All models are evaluated on identical chronological out-of-sample test data using:

$$\text{MAE} = \frac{1}{n} \sum_{t=1}^{n} |y_t - \hat{y}_t|$$

$$\text{RMSE} = \sqrt{\frac{1}{n} \sum_{t=1}^{n} (y_t - \hat{y}_t)^2}$$

- **MAE (Mean Absolute Error)**: Measures average absolute magnitude of forecast errors in kilowatts (kW).
- **RMSE (Root Mean Squared Error)**: Penalizes larger deviations more heavily, essential for grid reliability where large under-forecasts cause shortages.

---

## 📄 License
This project is open-source and available under the MIT License.
Dataset courtesy of the [UCI Machine Learning Repository](https://archive.ics.uci.edu/).
